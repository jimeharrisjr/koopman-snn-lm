//! Rung 1 — the linear ceiling. Protocol: docs/01-rung1-preregistration.md.
//!
//! Run from the repository root:
//!   cargo run --release --bin rung1 -- data/text8/text8

use koopman_snn_lm::corpus::{load_subset, SUBSET, TRAIN_END, V, VALID_END};
use std::ops::Range;
use std::path::Path;
use std::thread;

const STREAMS: usize = 4096;
const WARM: usize = 512;
const VALID: Range<usize> = TRAIN_END..VALID_END;
const TEST: Range<usize> = VALID_END..SUBSET;

#[derive(Default, Clone, Copy)]
struct Stats {
    sum: f64,
    sumsq: f64,
    n: f64,
}
impl Stats {
    fn add(&mut self, bits: f64) {
        self.sum += bits;
        self.sumsq += bits * bits;
        self.n += 1.0;
    }
    fn bpc(&self) -> f64 {
        self.sum / self.n
    }
    fn se(&self) -> f64 {
        let m = self.bpc();
        ((self.sumsq / self.n - m * m).max(0.0) / self.n).sqrt()
    }
}

#[derive(Clone)]
struct Row {
    arm: String,
    valid: f64,
    test: f64,
    se: f64,
    note: String,
}

/// Softmax in place; returns −log2 p(target).
fn softmax_bits(logits: &mut [f32; V], target: usize) -> f64 {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - mx).exp();
        z += *l;
    }
    for l in logits.iter_mut() {
        *l /= z;
    }
    -(logits[target].max(1e-30) as f64).log2()
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}
impl Adam {
    fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    fn step(&mut self, w: &mut [f32], g: &[f32], lr: f32) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        for i in 0..w.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            w[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

/// Registered schedule: lr0 for epochs 1–4, lr0/4 for 5–6, lr0/16 for 7–8;
/// `mult` doubles every phase for the convergence check.
fn lr_at(epoch: usize, mult: usize, lr0: f32) -> f32 {
    match epoch / mult {
        0..=3 => lr0,
        4..=5 => lr0 / 4.0,
        _ => lr0 / 16.0,
    }
}

// ---------------------------------------------------------------- arm A

fn arm_a(c: &[u8]) -> (Row, Row) {
    let mut uni = [0f64; V];
    let mut bi = vec![0f64; V * V];
    for t in 0..TRAIN_END {
        uni[c[t] as usize] += 1.0;
        if t > 0 {
            bi[c[t - 1] as usize * V + c[t] as usize] += 1.0;
        }
    }
    let ctx: Vec<f64> = (0..V).map(|a| bi[a * V..(a + 1) * V].iter().sum()).collect();
    let total: f64 = uni.iter().sum();
    let eval_uni = |d: f64, r: Range<usize>| {
        let mut s = Stats::default();
        for t in r {
            s.add(-((uni[c[t] as usize] + d) / (total + V as f64 * d)).log2());
        }
        s
    };
    let eval_bi = |d: f64, r: Range<usize>| {
        let mut s = Stats::default();
        for t in r {
            let (a, b) = (c[t - 1] as usize, c[t] as usize);
            s.add(-((bi[a * V + b] + d) / (ctx[a] + V as f64 * d)).log2());
        }
        s
    };
    let deltas = [1e-3, 1e-2, 1e-1, 1.0];
    let pick = |f: &dyn Fn(f64, Range<usize>) -> Stats, name: &str| {
        let (d, v) = deltas
            .iter()
            .map(|&d| (d, f(d, VALID).bpc()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let t = f(d, TEST);
        Row { arm: name.into(), valid: v, test: t.bpc(), se: t.se(), note: format!("delta={d}") }
    };
    (pick(&eval_uni, "A unigram"), pick(&eval_bi, "A bigram"))
}

// ---------------------------------------------------------------- arm B

struct Window {
    k: usize,
    w: Vec<f32>,
    b: [f32; V],
}
impl Window {
    fn logits(&self, c: &[u8], t: usize) -> [f32; V] {
        let mut l = self.b;
        for lag in 1..=self.k.min(t) {
            let o = ((lag - 1) * V + c[t - lag] as usize) * V;
            for (x, y) in l.iter_mut().zip(&self.w[o..o + V]) {
                *x += *y;
            }
        }
        l
    }
    fn eval(&self, c: &[u8], r: Range<usize>) -> Stats {
        let mut s = Stats::default();
        for t in r {
            let mut l = self.logits(c, t);
            s.add(softmax_bits(&mut l, c[t] as usize));
        }
        s
    }
}

fn arm_b(c: &[u8], k: usize, mult: usize) -> Row {
    let len = TRAIN_END / STREAMS;
    let mut m = Window { k, w: vec![0.0; k * V * V], b: [0.0; V] };
    let n = k * V * V + V;
    let mut params = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut adam = Adam::new(n);
    let lr0 = 0.02 / (k as f32).sqrt();
    let inv = 1.0 / STREAMS as f32;
    let mut last_valid = f64::NAN;
    for epoch in 0..8 * mult {
        let lr = lr_at(epoch, mult, lr0);
        let mut tr = Stats::default();
        for s in 0..len {
            g.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..STREAMS {
                let t = i * len + s;
                let target = c[t] as usize;
                let mut p = m.logits(c, t);
                tr.add(softmax_bits(&mut p, target));
                p[target] -= 1.0;
                for lag in 1..=k.min(t) {
                    let o = ((lag - 1) * V + c[t - lag] as usize) * V;
                    for (x, y) in g[o..o + V].iter_mut().zip(&p) {
                        *x += *y * inv;
                    }
                }
                for (x, y) in g[k * V * V..].iter_mut().zip(&p) {
                    *x += *y * inv;
                }
            }
            params[..k * V * V].copy_from_slice(&m.w);
            params[k * V * V..].copy_from_slice(&m.b);
            adam.step(&mut params, &g, lr);
            m.w.copy_from_slice(&params[..k * V * V]);
            m.b.copy_from_slice(&params[k * V * V..]);
        }
        last_valid = m.eval(c, VALID).bpc();
        eprintln!(
            "[B K={k} x{mult}] epoch {:>2} lr {:.5} train {:.4} valid {:.4}",
            epoch + 1,
            lr,
            tr.bpc(),
            last_valid
        );
    }
    let t = m.eval(c, TEST);
    Row {
        arm: format!("B window logistic K={k}"),
        valid: last_valid,
        test: t.bpc(),
        se: t.se(),
        note: format!("schedule x{mult}"),
    }
}

// ---------------------------------------------------------------- arm D

struct Traces {
    alphas: Vec<f32>,
    w: Vec<f32>,
    b: [f32; V],
}
impl Traces {
    fn feed(&self, x: &mut [f32], ch: usize) {
        for (j, &a) in self.alphas.iter().enumerate() {
            let xs = &mut x[j * V..(j + 1) * V];
            for v in xs.iter_mut() {
                *v *= a;
            }
            xs[ch] += 1.0 - a;
        }
    }
    fn logits(&self, x: &[f32]) -> [f32; V] {
        let mut l = self.b;
        for (f, &xv) in x.iter().enumerate() {
            if xv != 0.0 {
                for (a, b) in l.iter_mut().zip(&self.w[f * V..(f + 1) * V]) {
                    *a += xv * *b;
                }
            }
        }
        l
    }
    fn eval(&self, c: &[u8], r: Range<usize>) -> Stats {
        let mut x = vec![0f32; self.alphas.len() * V];
        for u in r.start.saturating_sub(4 * WARM)..r.start {
            self.feed(&mut x, c[u] as usize);
        }
        let mut s = Stats::default();
        for t in r {
            let mut l = self.logits(&x);
            s.add(softmax_bits(&mut l, c[t] as usize));
            self.feed(&mut x, c[t] as usize);
        }
        s
    }
}

fn arm_d(c: &[u8], name: &str, taus: &[f64], mult: usize) -> Row {
    let nb = taus.len();
    let f = nb * V;
    let alphas: Vec<f32> =
        taus.iter().map(|&t| if t == 0.0 { 0.0 } else { (-1.0 / t).exp() as f32 }).collect();
    let mut m = Traces { alphas, w: vec![0.0; f * V], b: [0.0; V] };
    let n = f * V + V;
    let mut params = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut adam = Adam::new(n);
    let lr0 = 0.02 / (nb as f32).sqrt();
    let len = TRAIN_END / STREAMS;
    let inv = 1.0 / STREAMS as f32;
    let mut state = vec![0f32; STREAMS * f];
    let mut last_valid = f64::NAN;
    for epoch in 0..8 * mult {
        let lr = lr_at(epoch, mult, lr0);
        state.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..STREAMS {
            let start = i * len;
            let x = &mut state[i * f..(i + 1) * f];
            for u in start.saturating_sub(WARM)..start {
                m.feed(x, c[u] as usize);
            }
        }
        let mut tr = Stats::default();
        for s in 0..len {
            g.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..STREAMS {
                let t = i * len + s;
                let x = &mut state[i * f..(i + 1) * f];
                if s > 0 {
                    m.feed(x, c[t - 1] as usize);
                }
                let target = c[t] as usize;
                let mut p = m.logits(x);
                tr.add(softmax_bits(&mut p, target));
                p[target] -= 1.0;
                for (fi, &xv) in x.iter().enumerate() {
                    if xv != 0.0 {
                        let s = xv * inv;
                        for (a, b) in g[fi * V..(fi + 1) * V].iter_mut().zip(&p) {
                            *a += s * *b;
                        }
                    }
                }
                for (a, b) in g[f * V..].iter_mut().zip(&p) {
                    *a += *b * inv;
                }
            }
            params[..f * V].copy_from_slice(&m.w);
            params[f * V..].copy_from_slice(&m.b);
            adam.step(&mut params, &g, lr);
            m.w.copy_from_slice(&params[..f * V]);
            m.b.copy_from_slice(&params[f * V..]);
        }
        last_valid = m.eval(c, VALID).bpc();
        eprintln!(
            "[{name} x{mult}] epoch {:>2} lr {:.5} train {:.4} valid {:.4}",
            epoch + 1,
            lr,
            tr.bpc(),
            last_valid
        );
    }
    let t = m.eval(c, TEST);
    Row {
        arm: format!("{name} trace logistic ({nb} taus)"),
        valid: last_valid,
        test: t.bpc(),
        se: t.se(),
        note: format!("schedule x{mult}"),
    }
}

// ---------------------------------------------------------------- arm C

const KMAX: usize = 64;

/// Lower-triangular Cholesky solve of (A) X = B, A given dense row-major d×d.
/// Returns None if A is not positive definite.
fn chol_solve(a: &[f64], d: usize, rhs: &[f64], nrhs: usize) -> Option<Vec<f64>> {
    let mut l = vec![0f64; d * d];
    for j in 0..d {
        let (head, tail) = l.split_at_mut(j * d + j);
        let rowj = &head[j * d..j * d + j];
        let s = a[j * d + j] - rowj.iter().map(|x| x * x).sum::<f64>();
        if !(s > 0.0) {
            return None;
        }
        let ljj = s.sqrt();
        tail[0] = ljj;
        let rowj: Vec<f64> = rowj.to_vec();
        for i in j + 1..d {
            let dot: f64 = l[i * d..i * d + j].iter().zip(&rowj).map(|(x, y)| x * y).sum();
            l[i * d + j] = (a[i * d + j] - dot) / ljj;
        }
    }
    let mut x = rhs.to_vec();
    for r in 0..nrhs {
        for i in 0..d {
            let mut s = x[i * nrhs + r];
            for k in 0..i {
                s -= l[i * d + k] * x[k * nrhs + r];
            }
            x[i * nrhs + r] = s / l[i * d + i];
        }
        for i in (0..d).rev() {
            let mut s = x[i * nrhs + r];
            for k in i + 1..d {
                s -= l[k * d + i] * x[k * nrhs + r];
            }
            x[i * nrhs + r] = s / l[i * d + i];
        }
    }
    Some(x)
}

fn arm_c(c: &[u8], ks: &[usize]) -> Vec<Row> {
    let dim = KMAX * V;
    let nthreads = 16;
    let span = (TRAIN_END - KMAX) / nthreads + 1;
    // Gram matrices of lagged one-hot features, as exact counts.
    let parts: Vec<(Vec<u32>, Vec<u32>)> = thread::scope(|sc| {
        let hs: Vec<_> = (0..nthreads)
            .map(|th| {
                sc.spawn(move || {
                    let mut g = vec![0u32; dim * dim];
                    let mut xy = vec![0u32; dim * V];
                    let lo = KMAX + th * span;
                    let hi = (lo + span).min(TRAIN_END);
                    let mut idx = [0usize; KMAX];
                    for t in lo..hi {
                        for k in 0..KMAX {
                            idx[k] = k * V + c[t - k - 1] as usize;
                        }
                        let y = c[t] as usize;
                        for i in 0..KMAX {
                            xy[idx[i] * V + y] += 1;
                            let row = idx[i] * dim;
                            for &j in &idx[i..] {
                                g[row + j] += 1;
                            }
                        }
                    }
                    (g, xy)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut g = vec![0f64; dim * dim];
    let mut xy = vec![0f64; dim * V];
    for (pg, pxy) in &parts {
        for (a, b) in g.iter_mut().zip(pg) {
            *a += *b as f64;
        }
        for (a, b) in xy.iter_mut().zip(pxy) {
            *a += *b as f64;
        }
    }
    drop(parts);
    for i in 0..dim {
        for j in 0..i {
            g[i * dim + j] = g[j * dim + i];
        }
    }
    let lambdas = [1e-8, 1e-6, 1e-4, 1e-2];
    let epss = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4];
    let (g, xy) = (&g, &xy);
    thread::scope(|sc| {
        let hs: Vec<_> = ks
            .iter()
            .map(|&k| {
                sc.spawn(move || {
                    let d = k * V;
                    let mean_diag = (0..d).map(|i| g[i * dim + i]).sum::<f64>() / d as f64;
                    let rhs: Vec<f64> = xy[..d * V].to_vec();
                    let predict = |coef: &[f64], t: usize, out: &mut [f64; V]| {
                        out.iter_mut().for_each(|x| *x = 0.0);
                        for lag in 0..k {
                            let o = (lag * V + c[t - lag - 1] as usize) * V;
                            for (a, b) in out.iter_mut().zip(&coef[o..o + V]) {
                                *a += *b;
                            }
                        }
                    };
                    let eval = |coef: &[f64], eps: f64, r: Range<usize>| {
                        let mut s = Stats::default();
                        let mut p = [0f64; V];
                        for t in r {
                            predict(coef, t, &mut p);
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
                    for &lr in &lambdas {
                        let mut a = vec![0f64; d * d];
                        for i in 0..d {
                            a[i * d..(i + 1) * d].copy_from_slice(&g[i * dim..i * dim + d]);
                            a[i * d + i] += lr * mean_diag;
                        }
                        let Some(coef) = chol_solve(&a, d, &rhs, V) else {
                            eprintln!("[C K={k}] lambda_rel={lr:e}: not positive definite");
                            continue;
                        };
                        for &eps in &epss {
                            let v = eval(&coef, eps, VALID).bpc();
                            eprintln!("[C K={k}] lambda_rel={lr:e} eps={eps:e} valid {v:.4}");
                            if best.as_ref().map_or(true, |b| v < b.0) {
                                best = Some((v, lr, eps, coef.clone()));
                            }
                        }
                    }
                    let (v, lr, eps, coef) = best.expect("no ridge value was solvable");
                    let t = eval(&coef, eps, TEST);
                    Row {
                        arm: format!("C window least squares K={k}"),
                        valid: v,
                        test: t.bpc(),
                        se: t.se(),
                        note: format!("lambda_rel={lr:e} eps={eps:e}"),
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

// ---------------------------------------------------------------- main

fn log_spaced(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64)).collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: rung1 <path to text8>");
    let c = load_subset(Path::new(&path)).expect("failed to load the text8 subset");
    let c = &c[..];
    let ks = [1usize, 2, 4, 8, 16, 32, 64];
    let mut d16 = vec![0.0];
    d16.extend(log_spaced(0.5, 128.0, 15));
    let banks: Vec<(&str, Vec<f64>)> = vec![
        ("D4", vec![0.0, 2.0, 8.0, 32.0]),
        ("D8", vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        ("D16", d16),
    ];
    let banks = &banks;

    let run_bd = |mult: usize, with_checks: bool| -> (Vec<Row>, Vec<Row>, Option<(Row, Row)>) {
        thread::scope(|sc| {
            let hb: Vec<_> = ks.iter().map(|&k| sc.spawn(move || arm_b(c, k, mult))).collect();
            let hd: Vec<_> = banks
                .iter()
                .map(|(n, t)| sc.spawn(move || arm_d(c, n, t, mult)))
                .collect();
            let checks = with_checks.then(|| {
                (
                    sc.spawn(move || arm_b(c, 16, 2 * mult)),
                    sc.spawn(move || arm_d(c, "D8", &banks[1].1, 2 * mult)),
                )
            });
            (
                hb.into_iter().map(|h| h.join().unwrap()).collect(),
                hd.into_iter().map(|h| h.join().unwrap()).collect(),
                checks.map(|(a, b)| (a.join().unwrap(), b.join().unwrap())),
            )
        })
    };

    let (a_rows, c_rows, (mut b_rows, mut d_rows, checks)) = thread::scope(|sc| {
        let ha = sc.spawn(move || arm_a(c));
        let hc = sc.spawn(move || arm_c(c, &ks));
        let bd = run_bd(1, true);
        (ha.join().unwrap(), hc.join().unwrap(), bd)
    });

    let (cb, cd) = checks.unwrap();
    let b16 = b_rows.iter().find(|r| r.arm.ends_with("K=16")).unwrap().valid;
    let d8 = d_rows.iter().find(|r| r.arm.starts_with("D8")).unwrap().valid;
    let (gain_b, gain_d) = (b16 - cb.valid, d8 - cd.valid);
    println!("# Rung 1 — linear ceiling\n");
    println!("## Convergence check (validation BPC)\n");
    println!("B K=16: x1 {b16:.4}  x2 {:.4}  gain {gain_b:+.4}", cb.valid);
    println!("D8    : x1 {d8:.4}  x2 {:.4}  gain {gain_d:+.4}", cd.valid);
    let doubled = gain_b > 0.003 || gain_d > 0.003;
    if doubled {
        println!("=> gain exceeds 0.003: all B and D arms re-run on the doubled schedule\n");
        let (b2, d2, _) = run_bd(2, false);
        b_rows = b2;
        d_rows = d2;
    } else {
        println!("=> within 0.003: the registered schedule stands\n");
    }

    println!("## Results\n");
    println!("| arm | valid BPC | test BPC | test SE | note |");
    println!("|---|---|---|---|---|");
    let (uni, bi) = a_rows;
    let all: Vec<&Row> =
        [&uni, &bi].into_iter().chain(&b_rows).chain(&c_rows).chain(&d_rows).collect();
    for r in &all {
        println!("| {} | {:.4} | {:.4} | {:.4} | {} |", r.arm, r.valid, r.test, r.se, r.note);
    }

    println!("\n## Gates\n");
    let g1 = (c_rows[0].test - bi.test).abs();
    let g2 = (b_rows[0].test - bi.test).abs();
    println!("G1 |C(K=1) - bigram| = {g1:.4}  {}", if g1 <= 0.01 { "PASS" } else { "FAIL" });
    println!("G2 |B(K=1) - bigram| = {g2:.4}  {}", if g2 <= 0.01 { "PASS" } else { "FAIL" });

    let ceil = b_rows
        .iter()
        .chain(&d_rows)
        .min_by(|a, b| a.valid.partial_cmp(&b.valid).unwrap())
        .unwrap();
    let best_c = c_rows.iter().min_by(|a, b| a.valid.partial_cmp(&b.valid).unwrap()).unwrap();
    println!("\n## Registered result\n");
    println!(
        "Linear ceiling: {:.4} test BPC (+/- {:.4}), arm \"{}\" (valid {:.4})",
        ceil.test, ceil.se, ceil.arm, ceil.valid
    );
    println!(
        "Closed-form value: {:.4} test BPC, arm \"{}\" (valid {:.4})",
        best_c.test, best_c.arm, best_c.valid
    );
}
