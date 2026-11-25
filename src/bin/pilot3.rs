//! Rung-3 pilot on a **synthetic** source. Reads no corpus text.
//!
//! The source is a random order-2 Markov chain over 27 symbols, so its next
//! symbol depends on a *pair* of previous symbols — structure an additive
//! model cannot represent. The pilot answers two questions before the
//! rung-3 registration: does the trainer learn non-additive structure at
//! all, and how long does an epoch take?
//!
//! usage: pilot3 <n> <lr> <epochs> <streams> <t_len> <rho0> <enc> [k]

use koopman_snn_lm::corpus::V;
use koopman_snn_lm::encoding::Encoding;
use koopman_snn_lm::reservoir::Rng;
use koopman_snn_lm::snn::{calibrated, eval_crate, mirror_matches_crate, ModelCfg, Trainer};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a[1].parse().unwrap();
    let lr0: f64 = a[2].parse().unwrap();
    let epochs: usize = a[3].parse().unwrap();
    let streams: usize = a[4].parse().unwrap();
    let t_len: usize = a[5].parse().unwrap();
    let rho0: f64 = a[6].parse().unwrap();
    let enc = match a[7].as_str() {
        "onehot" => Encoding::OneHot,
        "sparse" => Encoding::Sparse,
        "artic" => Encoding::Artic,
        e => panic!("unknown encoding {e}"),
    };
    let k: usize = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(0);

    // Order-2 Markov source.
    let mut rng = Rng::new(20261124);
    let mut table = vec![0f64; V * V * V];
    for ctx in 0..V * V {
        let row = &mut table[ctx * V..(ctx + 1) * V];
        let mut z = 0.0;
        for p in row.iter_mut() {
            *p = (2.0 * rng.normal()).exp();
            z += *p;
        }
        row.iter_mut().for_each(|p| *p /= z);
    }
    let (n_train, n_valid) = (2_000_000usize, 200_000usize);
    let mut chars = vec![0u8; n_train + n_valid];
    for t in 2..chars.len() {
        let ctx = chars[t - 2] as usize * V + chars[t - 1] as usize;
        let u = rng.uniform();
        let (mut acc, mut c) = (0.0, 0usize);
        while c < V - 1 {
            acc += table[ctx * V + c];
            if u < acc {
                break;
            }
            c += 1;
        }
        chars[t] = c as u8;
    }
    // Reference values on the validation part.
    let mut big = vec![1f64; V * V];
    for t in 1..n_train {
        big[chars[t - 1] as usize * V + chars[t] as usize] += 1.0;
    }
    let (mut h_true, mut h_bigram) = (0.0, 0.0);
    for t in n_train..n_train + n_valid {
        let (p2, p1, c) = (chars[t - 2] as usize, chars[t - 1] as usize, chars[t] as usize);
        h_true -= table[(p2 * V + p1) * V + c].log2();
        let z: f64 = big[p1 * V..(p1 + 1) * V].iter().sum();
        h_bigram -= (big[p1 * V + c] / z).log2();
    }
    println!(
        "source entropy {:.4} bits/symbol; best previous-symbol-only model {:.4}",
        h_true / n_valid as f64,
        h_bigram / n_valid as f64
    );

    let cfg = ModelCfg { n, enc, rho0, g_in: 1.0, k, seed: 0 };
    let (mut m, rate0) = calibrated(cfg, &chars[..20_000], 1000, 0.1).unwrap();
    println!(
        "n={n} enc={} rho0={rho0} k={k} params={} g_in={:.4} initial rate {:.4}",
        enc.name(),
        m.n_params(),
        m.cfg.g_in,
        rate0
    );
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).min(16);
    let mut tr = Trainer::new(&m, streams, t_len, threads);
    for e in 0..epochs {
        let lr = koopman_snn_lm::common::lr_at(e, epochs, lr0 as f32) as f64;
        let t0 = Instant::now();
        let (bpc, rate) = tr.epoch(&mut m, &chars[..n_train], lr);
        let secs = t0.elapsed().as_secs_f64();
        let v = eval_crate(&m, &chars, n_train - 20_000, &[n_train..n_train + n_valid]).unwrap()[0];
        println!(
            "epoch {:2} lr {:.5} train {:.4} rate {:.4} | valid {:.4} (rate {:.4}) | {:.1} s, {:.0} chars/s",
            e + 1, lr, bpc, rate, v.bpc, v.rate, secs, n_train as f64 / secs
        );
    }
    println!("mirror == crate after training: {}", mirror_matches_crate(&m, &chars[..20_000]).unwrap());
}
