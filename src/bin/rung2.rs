//! Rung 2 — the reservoir rung. Protocol: docs/03-rung2-preregistration.md.
//!
//! Run from the repository root:
//!   cargo run --release --bin rung2 -- data/text8/text8

use koopman_snn_lm::common::{chol_solve, lr_at, softmax_bits, Adam, Stats};
use koopman_snn_lm::corpus::{load_subset, SUBSET, TRAIN_END, V, VALID_END};
use koopman_snn_lm::reservoir::{run, Raster, ReservoirCfg};
use std::ops::Range;
use std::path::Path;
use std::thread;

const STREAMS: usize = 4096;
const VALID: Range<usize> = TRAIN_END..VALID_END;
const TEST: Range<usize> = VALID_END..SUBSET;
const WINDOW_K: usize = 8;
const CEILING: f64 = 3.077;
const CEILING_LS: f64 = 3.320;

type Feat<'a> = &'a (dyn Fn(usize, &mut Vec<u32>) + Sync);

#[derive(Clone, Debug)]
struct Fit {
    valid: f64,
    test: f64,
    se: f64,
    epochs: usize,
    converged: bool,
}

/// One logistic fit at a fixed epoch count, from zero initialisation.
fn fit_once(c: &[u8], rows: usize, feat: Feat, lr0: f32, epochs: usize, tag: &str) -> Fit {
    let len = TRAIN_END / STREAMS;
    let n = rows * V + V;
    let mut w = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut adam = Adam::new(n);
    let inv = 1.0 / STREAMS as f32;
    let mut f: Vec<u32> = Vec::new();
    let logits = |w: &[f32], f: &[u32]| {
        let mut l = [0f32; V];
        l.copy_from_slice(&w[rows * V..]);
        for &r in f {
            let o = r as usize * V;
            for (a, b) in l.iter_mut().zip(&w[o..o + V]) {
                *a += *b;
            }
        }
        l
    };
    for epoch in 0..epochs {
        let lr = lr_at(epoch, epochs, lr0);
        let mut tr = Stats::default();
        for s in 0..len {
            g.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..STREAMS {
                let t = i * len + s;
                if t == 0 {
                    continue;
                }
                let target = c[t] as usize;
                feat(t, &mut f);
                let mut p = logits(&w, &f);
                tr.add(softmax_bits(&mut p, target));
                p[target] -= 1.0;
                for &r in &f {
                    let o = r as usize * V;
                    for (a, b) in g[o..o + V].iter_mut().zip(&p) {
                        *a += *b * inv;
                    }
                }
                for (a, b) in g[rows * V..].iter_mut().zip(&p) {
                    *a += *b * inv;
                }
            }
            adam.step(&mut w, &g, lr);
        }
        eprintln!("[{tag} e{epochs}] epoch {:>2} lr {lr:.5} train {:.4}", epoch + 1, tr.bpc());
    }
    let eval = |r: Range<usize>| {
        let mut s = Stats::default();
        let mut f = Vec::new();
        for t in r {
            feat(t, &mut f);
            let mut l = logits(&w, &f);
            s.add(softmax_bits(&mut l, c[t] as usize));
        }
        s
    };
    let (v, t) = (eval(VALID), eval(TEST));
    eprintln!("[{tag} e{epochs}] valid {:.4} test {:.4}", v.bpc(), t.bpc());
    Fit { valid: v.bpc(), test: t.bpc(), se: t.se(), epochs, converged: true }
}

/// Registered convergence rule: 8, then 16, then 32 epochs, stopping at the
/// first schedule that improves validation BPC by less than 0.003.
fn fit(c: &[u8], rows: usize, feat: Feat, lr0: f32, tag: &str) -> Fit {
    let mut cur = fit_once(c, rows, feat, lr0, 8, tag);
    for epochs in [16, 32] {
        let next = fit_once(c, rows, feat, lr0, epochs, tag);
        let gain = cur.valid - next.valid;
        cur = next;
        if gain < 0.003 {
            return cur;
        }
    }
    cur.converged = false;
    cur
}

fn window_feat(c: &[u8], base: u32, t: usize, f: &mut Vec<u32>) {
    for lag in 1..=WINDOW_K.min(t) {
        f.push(base + ((lag - 1) * V) as u32 + c[t - lag] as u32);
    }
}

fn fit_s(c: &[u8], r: &Raster, target: f64, tag: &str) -> Fit {
    let feat = |t: usize, f: &mut Vec<u32>| {
        f.clear();
        f.extend(r.at(t - 1).iter().map(|&j| j as u32));
    };
    let lr0 = 0.02 / (r.n as f64 * target).sqrt() as f32;
    fit(c, r.n, &feat, lr0, tag)
}

fn fit_sw(c: &[u8], r: &Raster, target: f64, tag: &str) -> Fit {
    let base = r.n as u32;
    let feat = |t: usize, f: &mut Vec<u32>| {
        f.clear();
        f.extend(r.at(t - 1).iter().map(|&j| j as u32));
        window_feat(c, base, t, f);
    };
    let lr0 = 0.02 / (r.n as f64 * target + WINDOW_K as f64).sqrt() as f32;
    fit(c, r.n + WINDOW_K * V, &feat, lr0, tag)
}

struct Ridge {
    valid: f64,
    test: f64,
    lambda_rel: f64,
    eps: f64,
}

/// Closed-form ridge readout from exact Gram counts of (spikes + constant).
fn fit_ridge(c: &[u8], r: &Raster, tag: &str) -> Option<Ridge> {
    let d = r.n + 1;
    let mut g = vec![0u32; d * d];
    let mut xy = vec![0u32; d * V];
    let mut idx: Vec<usize> = Vec::new();
    for t in 1..TRAIN_END {
        idx.clear();
        idx.extend(r.at(t - 1).iter().map(|&j| j as usize));
        idx.push(r.n);
        let y = c[t] as usize;
        for (a, &i) in idx.iter().enumerate() {
            xy[i * V + y] += 1;
            let row = i * d;
            for &j in &idx[a..] {
                g[row + j] += 1;
            }
        }
    }
    let mut gf = vec![0f64; d * d];
    for i in 0..d {
        for j in i..d {
            let v = g[i * d + j] as f64;
            gf[i * d + j] = v;
            gf[j * d + i] = v;
        }
    }
    drop(g);
    let rhs: Vec<f64> = xy.iter().map(|&x| x as f64).collect();
    let mean_diag = (0..d).map(|i| gf[i * d + i]).sum::<f64>() / d as f64;
    let eval = |coef: &[f64], eps: f64, range: Range<usize>| {
        let mut s = Stats::default();
        for t in range {
            let mut p = [0f64; V];
            p.copy_from_slice(&coef[r.n * V..]);
            for &j in r.at(t - 1) {
                let o = j as usize * V;
                for (a, b) in p.iter_mut().zip(&coef[o..o + V]) {
                    *a += *b;
                }
            }
            let mut z = 0.0;
            for x in p.iter_mut() {
                *x = x.max(eps);
                z += *x;
            }
            s.add(-(p[c[t] as usize] / z).log2());
        }
        s
    };
    let mut best: Option<(f64, f64, f64, Vec<f64>)> = None;
    for lr in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2] {
        let mut a = gf.clone();
        for i in 0..d {
            a[i * d + i] += lr * mean_diag;
        }
        let Some(coef) = chol_solve(&a, d, &rhs, V) else {
            eprintln!("[{tag} ridge] lambda_rel={lr:e}: not positive definite");
            continue;
        };
        for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4] {
            let v = eval(&coef, eps, VALID).bpc();
            eprintln!("[{tag} ridge] lambda_rel={lr:e} eps={eps:e} valid {v:.4}");
            if best.as_ref().map_or(true, |b| v < b.0) {
                best = Some((v, lr, eps, coef.clone()));
            }
        }
    }
    let (v, lr, eps, coef) = best?;
    Some(Ridge { valid: v, test: eval(&coef, eps, TEST).bpc(), lambda_rel: lr, eps })
}

/// Registered rate calibration: 24 steps of geometric bisection on g_in over
/// characters 1,000–20,000 of the training text.
fn calibrate(c: &[u8], n: usize, rho: f64, seed: u64, target: f64) -> ReservoirCfg {
    let (mut lo, mut hi) = (0.1f64, 100.0f64);
    for _ in 0..24 {
        let mid = (lo * hi).sqrt();
        let cfg = ReservoirCfg { n, rho, g_in: mid, seed };
        let rate = run(&cfg, &c[..20_000]).expect("reservoir build").rate(1000..20_000);
        if rate < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ReservoirCfg { n, rho, g_in: (lo * hi).sqrt(), seed }
}

struct Res {
    cfg: ReservoirCfg,
    target: f64,
    rate: f64,
    raster: Raster,
}

fn make(c: &[u8], n: usize, rho: f64, seed: u64, target: f64) -> Res {
    let cfg = calibrate(c, n, rho, seed, target);
    let raster = run(&cfg, c).expect("reservoir run");
    let rate = raster.rate(0..TRAIN_END);
    eprintln!("[res n={n} rho={rho} r={target} seed={seed}] g_in={:.4} rate={rate:.4}", cfg.g_in);
    Res { cfg, target, rate, raster }
}

fn tag(r: &Res) -> String {
    format!("n{} rho{} r{} s{}", r.cfg.n, r.cfg.rho, r.target, r.cfg.seed)
}

fn flag(f: &Fit) -> &'static str {
    if f.converged {
        ""
    } else {
        " NOT CONVERGED"
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: rung2 <path to text8>");
    let c = load_subset(Path::new(&path)).expect("failed to load the text8 subset");
    let c = &c[..];

    let mut grid = Vec::new();
    for n in [512usize, 1024] {
        for rho in [0.0, 1.0, 3.0] {
            for target in [0.05, 0.2] {
                grid.push((n, rho, target));
            }
        }
    }

    // Gate G1 and stage 1, all in parallel.
    let (gate, stage1) = thread::scope(|sc| {
        let hg = sc.spawn(move || {
            let feat = |t: usize, f: &mut Vec<u32>| {
                f.clear();
                window_feat(c, 0, t, f);
            };
            fit(c, WINDOW_K * V, &feat, 0.02 / (WINDOW_K as f32).sqrt(), "gate window K=8")
        });
        let hs: Vec<_> = grid
            .iter()
            .map(|&(n, rho, target)| {
                sc.spawn(move || {
                    let res = make(c, n, rho, 0, target);
                    let tg = tag(&res);
                    let (s, sw, ridge) = thread::scope(|s2| {
                        let (r, tg) = (&res, &tg);
                        let a = s2.spawn(move || fit_s(c, &r.raster, r.target, &format!("S {tg}")));
                        let b =
                            s2.spawn(move || fit_sw(c, &r.raster, r.target, &format!("SW {tg}")));
                        let d = s2.spawn(move || fit_ridge(c, &r.raster, tg));
                        (a.join().unwrap(), b.join().unwrap(), d.join().unwrap())
                    });
                    (res.cfg, res.target, res.rate, s, sw, ridge)
                })
            })
            .collect();
        (hg.join().unwrap(), hs.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>())
    });

    println!("# Rung 2 — the reservoir rung\n");
    println!("## Gate\n");
    let gdiff = (gate.valid - 3.1214).abs();
    println!(
        "G1 window-only K=8: valid {:.4} (rung 1: 3.1214), |diff| = {gdiff:.4}, {} epochs  {}",
        gate.valid,
        gate.epochs,
        if gdiff <= 0.003 { "PASS" } else { "FAIL" }
    );

    println!("\n## Stage 1 (reservoir seed 0)\n");
    println!("| N | rho | target rate | g_in | realised rate | S valid | S test | S epochs | SW valid | SW test | SW epochs | ridge valid | ridge test | ridge (lambda_rel, eps) |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (cfg, target, rate, s, sw, ridge) in &stage1 {
        let (rv, rt, rn) = match ridge {
            Some(r) => (
                format!("{:.4}", r.valid),
                format!("{:.4}", r.test),
                format!("{:e}, {:e}", r.lambda_rel, r.eps),
            ),
            None => ("-".into(), "-".into(), "unsolvable".into()),
        };
        println!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {}{} | {:.4} | {:.4} | {}{} | {rv} | {rt} | {rn} |",
            cfg.n, cfg.rho, target, cfg.g_in, rate, s.valid, s.test, s.epochs, flag(s),
            sw.valid, sw.test, sw.epochs, flag(sw)
        );
    }

    // Stage 2: seed band for the selected configuration of each readout.
    let pick = |key: &dyn Fn(&(ReservoirCfg, f64, f64, Fit, Fit, Option<Ridge>)) -> f64| {
        stage1.iter().min_by(|a, b| key(a).partial_cmp(&key(b)).unwrap()).unwrap()
    };
    let best_s = pick(&|x| x.3.valid);
    let best_sw = pick(&|x| x.4.valid);
    let (stage2_s, stage2_sw) = thread::scope(|sc| {
        let hs: Vec<_> = [1u64, 2]
            .into_iter()
            .map(|seed| {
                sc.spawn(move || {
                    let r = make(c, best_s.0.n, best_s.0.rho, seed, best_s.1);
                    (r.rate, fit_s(c, &r.raster, r.target, &format!("S {}", tag(&r))))
                })
            })
            .collect();
        let hw: Vec<_> = [1u64, 2]
            .into_iter()
            .map(|seed| {
                sc.spawn(move || {
                    let r = make(c, best_sw.0.n, best_sw.0.rho, seed, best_sw.1);
                    (r.rate, fit_sw(c, &r.raster, r.target, &format!("SW {}", tag(&r))))
                })
            })
            .collect();
        (
            hs.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>(),
            hw.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>(),
        )
    });

    println!("\n## Stage 2 (seed band) and registered result\n");
    let report = |name: &str,
                  best: &(ReservoirCfg, f64, f64, Fit, Fit, Option<Ridge>),
                  seed0: &Fit,
                  more: &[(f64, Fit)]| {
        let mut tests = vec![seed0.test];
        tests.extend(more.iter().map(|m| m.1.test));
        let mean = tests.iter().sum::<f64>() / tests.len() as f64;
        let (mn, mx) = tests
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        let lift = CEILING - mean;
        let range = mx - mn;
        println!(
            "{name}: selected N={} rho={} target rate={} (seed-0 valid {:.4})",
            best.0.n, best.0.rho, best.1, seed0.valid
        );
        println!("  seed 0: test {:.4} (SE {:.4}){}", seed0.test, seed0.se, flag(seed0));
        for (i, (rate, f)) in more.iter().enumerate() {
            println!(
                "  seed {}: test {:.4} (SE {:.4}), valid {:.4}, rate {rate:.4}, {} epochs{}",
                i + 1, f.test, f.se, f.valid, f.epochs, flag(f)
            );
        }
        println!("  mean {mean:.4}  min {mn:.4}  max {mx:.4}  seed range {range:.4}");
        println!(
            "  lift over the linear ceiling ({CEILING}) = {lift:+.4}  =>  {}\n",
            if lift > 0.01 && lift > range { "LIFTS above the ceiling" } else { "does NOT lift" }
        );
    };
    report("S-logistic", best_s, &best_s.3, &stage2_s);
    report("SW-logistic", best_sw, &best_sw.4, &stage2_sw);

    let best_ridge = stage1
        .iter()
        .filter_map(|x| x.5.as_ref().map(|r| (x, r)))
        .min_by(|a, b| a.1.valid.partial_cmp(&b.1.valid).unwrap());
    if let Some((x, r)) = best_ridge {
        println!(
            "S-ridge (seed 0, descriptive): best N={} rho={} target rate={}: test {:.4} (valid {:.4}) vs closed-form ceiling {CEILING_LS}: {:+.4}",
            x.0.n, x.0.rho, x.1, r.test, r.valid, CEILING_LS - r.test
        );
    }
}
