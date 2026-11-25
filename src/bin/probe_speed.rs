//! Unregistered feasibility probe: reservoir simulation speed and spike rate
//! on SYNTHETIC uniform-random characters. Reads no corpus text.

use koopman_snn_lm::reservoir::{run, ReservoirCfg, Rng};
use std::time::Instant;

fn main() {
    let steps = 50_000;
    let mut rng = Rng::new(1);
    let chars: Vec<u8> = (0..steps).map(|_| (rng.next_u64() % 27) as u8).collect();
    for &n in &[512usize, 1024, 2048] {
        for &rho in &[0.0, 1.0] {
            for &g_in in &[0.5, 1.0, 2.0] {
                let cfg = ReservoirCfg { n, rho, g_in, seed: 7 };
                let t0 = Instant::now();
                let r = run(&cfg, &chars).unwrap();
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "n={n:5} rho={rho} g_in={g_in}: rate {:.4}  {:.0} steps/s  (10M steps = {:.1} min)",
                    r.rate(1000..steps),
                    steps as f64 / dt,
                    1e7 / (steps as f64 / dt) / 60.0
                );
            }
        }
    }
}
