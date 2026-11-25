//! A frozen random Koopman SNN used as a feature generator.
//!
//! One `kdmd_snn` LIF layer with per-neuron (heterogeneous) membrane time
//! constants, random input weights from the 27 one-hot character lines, and
//! optional random recurrence. Nothing in it is trained. Every state step
//! goes through the crate's closed-form propagator (`KoopmanLayer::lif_hetero`).
//! Time is measured in characters: one simulation step per character, dt = 1.

use crate::corpus::V;
use faer::Mat;
use kdmd_snn::{KoopmanLayer, Network, SnnError, SpikeVec};

/// SplitMix64 — a fixed, dependency-free generator so reservoirs are
/// reproducible independent of any crate version.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Reservoir definition. Weights are drawn in a fixed order from `seed`:
/// input weights row-major, then recurrent weights row-major.
#[derive(Debug, Clone, Copy)]
pub struct ReservoirCfg {
    pub n: usize,
    /// Recurrent scale: W_rec = rho · G / sqrt(n), G standard normal. 0 = none.
    pub rho: f64,
    /// Input scale: W_in = g_in · G.
    pub g_in: f64,
    pub seed: u64,
}

/// Membrane time constants: log-spaced from 1 to 64 characters.
pub const TAU_M_LO: f64 = 1.0;
pub const TAU_M_HI: f64 = 64.0;
/// Synaptic time constant, in characters.
pub const TAU_S: f64 = 0.5;

pub fn build(cfg: &ReservoirCfg) -> Result<Network, SnnError> {
    let n = cfg.n;
    let mut rng = Rng::new(cfg.seed);
    let win: Vec<f64> = (0..n * V).map(|_| cfg.g_in * rng.normal()).collect();
    let taus_m: Vec<f64> = (0..n)
        .map(|i| TAU_M_LO * (TAU_M_HI / TAU_M_LO).powf(i as f64 / (n - 1) as f64))
        .collect();
    let taus_s = vec![TAU_S; n];
    let w_in = Mat::from_fn(n, V, |i, j| win[i * V + j]);
    let mut layer = KoopmanLayer::lif_hetero(&taus_m, &taus_s, 1.0, 1.0, 1.0, w_in, 1)?;
    if cfg.rho != 0.0 {
        let s = cfg.rho / (n as f64).sqrt();
        let wrec: Vec<f64> = (0..n * n).map(|_| s * rng.normal()).collect();
        layer = layer.with_recurrent(Mat::from_fn(n, n, |i, j| wrec[i * n + j]))?;
    }
    Network::new(vec![layer], 1)
}

/// Spike raster in compressed form: the neurons that spiked at step `t` are
/// `idx[off[t]..off[t + 1]]`. The spikes at step `t` have seen characters
/// `0..=t`, so they are the features for predicting character `t + 1`.
pub struct Raster {
    pub n: usize,
    pub off: Vec<u64>,
    pub idx: Vec<u16>,
}
impl Raster {
    pub fn at(&self, t: usize) -> &[u16] {
        &self.idx[self.off[t] as usize..self.off[t + 1] as usize]
    }
    pub fn rate(&self, steps: std::ops::Range<usize>) -> f64 {
        (self.off[steps.end] - self.off[steps.start]) as f64 / (steps.len() * self.n) as f64
    }
}

/// Drive the reservoir with the character sequence, from a zero state.
pub fn run(cfg: &ReservoirCfg, chars: &[u8]) -> Result<Raster, SnnError> {
    assert!(cfg.n <= u16::MAX as usize + 1);
    let mut net = build(cfg)?;
    let mut off = Vec::with_capacity(chars.len() + 1);
    let mut idx: Vec<u16> = Vec::new();
    off.push(0u64);
    let mut input = SpikeVec::new(V);
    for &ch in chars {
        input.clear();
        input.push(ch as u32);
        let out = net.step(&input)?;
        idx.extend(out.active().iter().map(|&j| j as u16));
        off.push(idx.len() as u64);
    }
    Ok(Raster { n: cfg.n, off, idx })
}
