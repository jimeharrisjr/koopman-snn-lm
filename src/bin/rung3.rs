//! Rung 3 — the registered character probe.
//! Protocol: docs/05-rung3-preregistration.md (frozen before this file existed).
//!
//! usage: rung3 <text8 path>
//! Markdown summary on stdout, progress on stderr, one line per finished arm
//! appended to results/rung3-arms.tsv (arms already there are skipped).

use koopman_snn_lm::common::lr_at;
use koopman_snn_lm::corpus::{load_subset, SUBSET, TRAIN_END, VALID_END, V};
use koopman_snn_lm::encoding::Encoding;
use koopman_snn_lm::snn::{calibrated, eval_crate, mirror_matches_crate, Model, ModelCfg, Trainer};
use std::io::Write;
use std::time::Instant;

const N: usize = 512;
const EPOCHS: usize = 10;
const LR0: f32 = 0.01;
const STREAMS: usize = 64;
const T_LEN: usize = 64;
const THREADS: usize = 12;
const TSV: &str = "results/rung3-arms.tsv";

#[derive(Clone, Debug)]
struct Arm {
    name: String,
    enc: Encoding,
    seed: u64,
    k: usize,
    rho0: f64,
}

#[derive(Clone, Debug)]
struct Res {
    name: String,
    params: usize,
    g_in: f64,
    valid: f64,
    test: f64,
    se: f64,
    rate: f64,
    active_in: f64,
    last_gain: f64,
    curve: String,
}

fn enc_tag(e: Encoding) -> &'static str {
    match e {
        Encoding::OneHot => "onehot",
        Encoding::Sparse => "sparse",
        Encoding::Artic => "artic",
    }
}

fn load_done() -> Vec<Res> {
    let Ok(s) = std::fs::read_to_string(TSV) else { return Vec::new() };
    s.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Res {
                name: f[0].into(),
                params: f[1].parse().unwrap(),
                g_in: f[2].parse().unwrap(),
                valid: f[3].parse().unwrap(),
                test: f[4].parse().unwrap(),
                se: f[5].parse().unwrap(),
                rate: f[6].parse().unwrap(),
                active_in: f[7].parse().unwrap(),
                last_gain: f[8].parse().unwrap(),
                curve: f[9].into(),
            }
        })
        .collect()
}

fn run_arm(arm: &Arm, chars: &[u8], done: &mut Vec<Res>) -> Res {
    if let Some(r) = done.iter().find(|r| r.name == arm.name) {
        eprintln!("[{}] already done, skipping", arm.name);
        return r.clone();
    }
    let t0 = Instant::now();
    let cfg = ModelCfg { n: N, enc: arm.enc, rho0: arm.rho0, g_in: 1.0, k: arm.k, seed: arm.seed };
    let (mut m, rate0) = calibrated(cfg, &chars[..20_000], 1000, 0.1).expect("build");
    eprintln!("[{}] params {} g_in {:.4} initial rate {:.4}", arm.name, m.n_params(), m.cfg.g_in, rate0);
    let mut tr = Trainer::new(&m, STREAMS, T_LEN, THREADS);
    let mut curve: Vec<f64> = Vec::new();
    for e in 0..EPOCHS {
        let lr = lr_at(e, EPOCHS, LR0) as f64;
        let (bpc, rate) = tr.epoch(&mut m, &chars[..TRAIN_END], lr);
        let v = eval_crate(&m, chars, 8_900_000, &[TRAIN_END..VALID_END]).expect("eval")[0];
        eprintln!(
            "[{}] epoch {:2} lr {:.5} train {:.4} rate {:.4} | monitor valid {:.4} | {:.0} s",
            arm.name, e + 1, lr, bpc, rate, v.bpc, t0.elapsed().as_secs_f64()
        );
        curve.push(v.bpc);
    }
    assert!(
        mirror_matches_crate(&m, &chars[..20_000]).expect("gate"),
        "GATE G2 FAILED for {}", arm.name
    );
    let sc = eval_crate(&m, chars, 0, &[TRAIN_END..VALID_END, VALID_END..SUBSET]).expect("eval");
    let mut cnt = [0f64; V];
    for &c in &chars[VALID_END..SUBSET] {
        cnt[c as usize] += 1.0;
    }
    let active_in = (0..V).map(|c| cnt[c] * m.codes[c].len() as f64).sum::<f64>()
        / (SUBSET - VALID_END) as f64;
    let res = Res {
        name: arm.name.clone(),
        params: m.n_params(),
        g_in: m.cfg.g_in,
        valid: sc[0].bpc,
        test: sc[1].bpc,
        se: sc[1].se,
        rate: sc[1].rate,
        active_in,
        last_gain: curve[EPOCHS - 2] - curve[EPOCHS - 1],
        curve: curve.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(","),
    };
    eprintln!(
        "[{}] FINAL valid {:.4} test {:.4} (SE {:.4}) rate {:.4} | {:.0} s",
        arm.name, res.valid, res.test, res.se, res.rate, t0.elapsed().as_secs_f64()
    );
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(TSV).unwrap();
    writeln!(
        f,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        res.name, res.params, res.g_in, res.valid, res.test, res.se, res.rate, res.active_in,
        res.last_gain, res.curve
    )
    .unwrap();
    done.push(res.clone());
    res
}

fn row(r: &Res) -> String {
    format!(
        "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2} | {} |",
        r.name,
        r.params,
        r.g_in,
        r.valid,
        r.test,
        r.se,
        r.rate,
        r.active_in,
        if r.last_gain >= 0.003 { format!("NOT CONVERGED ({:.4})", r.last_gain) } else { format!("{:.4}", r.last_gain) }
    )
}

fn band(label: &str, rs: &[Res]) {
    let t: Vec<f64> = rs.iter().map(|r| r.test).collect();
    let mean = t.iter().sum::<f64>() / t.len() as f64;
    let (mn, mx) = (t.iter().cloned().fold(f64::MAX, f64::min), t.iter().cloned().fold(f64::MIN, f64::max));
    let rate = rs.iter().map(|r| r.rate).sum::<f64>() / rs.len() as f64;
    let lift = 3.077 - mean;
    println!(
        "{label}: tests {:?}\n  mean {:.4}  min {:.4}  max {:.4}  seed range {:.4}  mean rate {:.4}\n  lift over the linear ceiling (3.077) = {:+.4}  =>  {}\n",
        t.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>(),
        mean, mn, mx, mx - mn, rate, lift,
        if lift > 0.01 && lift > mx - mn { "BEATS the ceiling" } else { "does NOT beat the ceiling" }
    );
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: rung3 <text8 path>");
    let chars = load_subset(std::path::Path::new(&path)).expect("corpus");
    std::fs::create_dir_all("results").unwrap();
    let mut done = load_done();

    // Gate G1.
    let m0 = Model::new(ModelCfg { n: N, enc: Encoding::OneHot, rho0: 0.5, g_in: 4.0, k: 0, seed: 0 }).unwrap();
    let g1 = eval_crate(&m0, &chars, 0, &[1..50_000]).unwrap()[0].bpc;
    let ok = (g1 - (V as f64).log2()).abs() < 1e-9;
    println!("# Rung 3 — the registered character probe\n\n## Gates\n");
    println!("G1 untrained, zero readout: {:.4} BPC (log2 27 = {:.4})  {}", g1, (V as f64).log2(), if ok { "PASS" } else { "FAIL" });
    assert!(ok, "GATE G1 FAILED");

    let arm = |tag: &str, enc: Encoding, seed: u64, k: usize, rho0: f64| Arm {
        name: format!("{tag}-{}-s{seed}", enc_tag(enc)),
        enc, seed, k, rho0,
    };
    // Stage 1.
    let s1: Vec<Res> = [Encoding::OneHot, Encoding::Sparse, Encoding::Artic]
        .iter()
        .map(|&e| run_arm(&arm("S", e, 0, 0, 0.5), &chars, &mut done))
        .collect();
    let encs = [Encoding::OneHot, Encoding::Sparse, Encoding::Artic];
    let best = (0..3).min_by(|&a, &b| s1[a].valid.total_cmp(&s1[b].valid)).unwrap();
    let enc = encs[best];
    // Stage 2.
    let mut s2 = vec![s1[best].clone()];
    for seed in [1, 2] {
        s2.push(run_arm(&arm("S", enc, seed, 0, 0.5), &chars, &mut done));
    }
    // Stage 3.
    let w8: Vec<Res> = (0..3).map(|s| run_arm(&arm("W8", enc, s, 8, 0.5), &chars, &mut done)).collect();
    let nr = run_arm(&arm("NR", enc, 0, 0, 0.0), &chars, &mut done);

    println!("G2 mirror == crate spikes: PASS for every trained model (the run aborts otherwise)\n");
    let head = "| arm | params | g_in | valid | test | test SE | test spike rate | active input lines | epoch 9→10 gain |\n|---|---|---|---|---|---|---|---|---|";
    println!("## Stage 1 — encoding arms (seed 0)\n\n{head}");
    s1.iter().for_each(|r| println!("{}", row(r)));
    println!("\nSelected encoding: {}\n\n## Stage 2 — seed band and kill criterion\n\n{head}", enc.name());
    s2.iter().for_each(|r| println!("{}", row(r)));
    println!();
    band("Spikes-only, selected encoding", &s2);
    println!("## Stage 3 — descriptive\n\n{head}");
    w8.iter().for_each(|r| println!("{}", row(r)));
    println!("{}\n", row(&nr));
    band("W8 (spikes + 8-character window)", &w8);
    println!("NR (no recurrence), seed 0: test {:.4} against {:.4} recurrent (difference {:+.4})", nr.test, s2[0].test, nr.test - s2[0].test);
}
