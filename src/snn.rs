//! A trainable one-layer recurrent Koopman SNN language model.
//!
//! State step (per neuron j, drive d = W_in·x_t + W_rec·s_{t−1}):
//!
//! ```text
//! v' = α v + γ i + δ d        i' = β i + (1 − β) d
//! s  = [v' ≥ θ]               v  = v' − θ s
//! logits_t = b + R·s_t  (+ additive window over the last K characters)
//! ```
//!
//! α, β, γ, δ are the closed-form propagator entries, read out of a
//! `kdmd_snn::KoopmanLayer::lif_hetero` layer — never recomputed here.
//!
//! **Two forward paths.** Training uses the in-repo mirror below (truncated
//! BPTT with a per-character loss, which the crate's whole-sequence
//! `Trainer` does not offer). Every reported number comes from
//! [`Model::to_network`] stepped by the crate itself. The mirror reproduces
//! the crate's arithmetic in the same order, and `mirror_matches_crate`
//! (test and run-time gate) checks the two give identical spikes.

use crate::corpus::V;
use crate::encoding::Encoding;
use crate::reservoir::{Rng, TAU_M_HI, TAU_M_LO, TAU_S};
use faer::Mat;
use kdmd_snn::{KoopmanLayer, Network, Operator, SnnError, SpikeVec};

#[derive(Debug, Clone, Copy)]
pub struct ModelCfg {
    pub n: usize,
    pub enc: Encoding,
    /// Initial recurrent scale: W_rec = rho0 · G / sqrt(n). 0 = no recurrence
    /// (and none is trained).
    pub rho0: f64,
    /// Input scale: W_in = g_in · G / sqrt(mean active lines per character).
    pub g_in: f64,
    /// Additive window lags on the readout (0 = spikes only).
    pub k: usize,
    pub seed: u64,
}

#[derive(Clone)]
pub struct Model {
    pub cfg: ModelCfg,
    pub d_in: usize,
    pub codes: Vec<Vec<u32>>,
    /// Column-major: win[j_in * n + i].
    pub win: Vec<f64>,
    /// Column-major: wrec[j * n + i]; empty if rho0 == 0.
    pub wrec: Vec<f64>,
    /// r[j * V + c].
    pub r: Vec<f64>,
    pub b: Vec<f64>,
    /// ww[(lag * V + ch) * V + c]; empty if k == 0.
    pub ww: Vec<f64>,
    pub alpha: Vec<f64>,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub delta: Vec<f64>,
    pub c1: Vec<f64>,
    pub theta: f64,
}

fn taus(n: usize) -> (Vec<f64>, Vec<f64>) {
    let tm = (0..n)
        .map(|i| TAU_M_LO * (TAU_M_HI / TAU_M_LO).powf(i as f64 / (n - 1) as f64))
        .collect();
    (tm, vec![TAU_S; n])
}

impl Model {
    pub fn new(cfg: ModelCfg) -> Result<Self, SnnError> {
        let n = cfg.n;
        let codes = cfg.enc.codes();
        let d_in = cfg.enc.d_in();
        let mean_active = codes.iter().map(|c| c.len()).sum::<usize>() as f64 / V as f64;
        let mut rng = Rng::new(cfg.seed);
        let s_in = cfg.g_in / mean_active.sqrt();
        let win: Vec<f64> = (0..n * d_in).map(|_| s_in * rng.normal()).collect();
        let wrec: Vec<f64> = if cfg.rho0 != 0.0 {
            let s = cfg.rho0 / (n as f64).sqrt();
            (0..n * n).map(|_| s * rng.normal()).collect()
        } else {
            Vec::new()
        };
        // Propagator entries from the crate's closed-form construction.
        let (tm, ts) = taus(n);
        let probe = KoopmanLayer::lif_hetero(&tm, &ts, 1.0, 1.0, 1.0, Mat::zeros(n, d_in), 1)?;
        let (mut alpha, mut gamma, mut beta) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        match probe.operator() {
            Operator::PerNeuron { blocks, .. } => {
                for j in 0..n {
                    alpha[j] = blocks[(0, j)];
                    gamma[j] = blocks[(1, j)];
                    assert_eq!(blocks[(2, j)], 0.0);
                    beta[j] = blocks[(3, j)];
                }
            }
            _ => unreachable!("lif_hetero builds a PerNeuron operator"),
        }
        let delta = (0..n).map(|j| probe.coupling(0, j)).collect();
        let c1 = (0..n).map(|j| probe.coupling(1, j)).collect();
        assert_eq!(probe.jumps(), &[-probe.theta(), 0.0]);
        Ok(Model {
            cfg,
            d_in,
            codes,
            win,
            wrec,
            r: vec![0.0; n * V],
            b: vec![0.0; V],
            ww: vec![0.0; cfg.k * V * V],
            alpha,
            gamma,
            beta,
            delta,
            c1,
            theta: probe.theta(),
        })
    }

    pub fn n_params(&self) -> usize {
        self.win.len() + self.wrec.len() + self.r.len() + self.b.len() + self.ww.len()
    }

    /// The same network as a `kdmd_snn::Network` (batch 1).
    pub fn to_network(&self) -> Result<Network, SnnError> {
        let n = self.cfg.n;
        let (tm, ts) = taus(n);
        let w_in = Mat::from_fn(n, self.d_in, |i, j| self.win[j * n + i]);
        let mut layer = KoopmanLayer::lif_hetero(&tm, &ts, 1.0, 1.0, 1.0, w_in, 1)?;
        if !self.wrec.is_empty() {
            layer = layer.with_recurrent(Mat::from_fn(n, n, |i, j| self.wrec[j * n + i]))?;
        }
        Network::new(vec![layer], 1)
    }

    /// Logits for the character after position `t`, given the spikes at `t`.
    pub fn logits(&self, spikes: &[u32], chars: &[u8], t: usize, out: &mut [f64; V]) {
        out.copy_from_slice(&self.b);
        for &j in spikes {
            let row = &self.r[j as usize * V..j as usize * V + V];
            for c in 0..V {
                out[c] += row[c];
            }
        }
        for lag in 0..self.cfg.k.min(t + 1) {
            let o = (lag * V + chars[t - lag] as usize) * V;
            for c in 0..V {
                out[c] += self.ww[o + c];
            }
        }
    }
}

/// Softmax in place; returns −log2 p(target).
pub fn softmax_bits64(l: &mut [f64; V], target: usize) -> f64 {
    let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut z = 0.0;
    for x in l.iter_mut() {
        *x = (*x - mx).exp();
        z += *x;
    }
    for x in l.iter_mut() {
        *x /= z;
    }
    -l[target].max(1e-300).log2()
}

/// Per-stream recurrent state carried across truncation windows.
#[derive(Clone)]
pub struct StreamState {
    pub v: Vec<f64>,
    pub i: Vec<f64>,
    pub prev: Vec<u32>,
}
impl StreamState {
    pub fn zeros(n: usize) -> Self {
        StreamState { v: vec![0.0; n], i: vec![0.0; n], prev: Vec::new() }
    }
}

/// One mirror step. Writes the pre-reset potentials into `vpre` and the
/// spikes into `out` (ascending). Arithmetic order matches the crate.
#[inline]
pub fn mirror_step(
    m: &Model,
    st: &mut StreamState,
    ch: u8,
    drive: &mut [f64],
    vpre: &mut [f64],
    out: &mut Vec<u32>,
) {
    let n = m.cfg.n;
    drive.fill(0.0);
    for &j in &m.codes[ch as usize] {
        let col = &m.win[j as usize * n..(j as usize + 1) * n];
        for i in 0..n {
            drive[i] += col[i];
        }
    }
    if !m.wrec.is_empty() {
        for &j in &st.prev {
            let col = &m.wrec[j as usize * n..(j as usize + 1) * n];
            for i in 0..n {
                drive[i] += col[i];
            }
        }
    }
    out.clear();
    for j in 0..n {
        let mut y0 = 0.0;
        y0 += m.alpha[j] * st.v[j];
        y0 += m.gamma[j] * st.i[j];
        y0 += m.delta[j] * drive[j];
        let mut y1 = 0.0;
        y1 += 0.0 * st.v[j];
        y1 += m.beta[j] * st.i[j];
        y1 += m.c1[j] * drive[j];
        vpre[j] = y0;
        if y0 >= m.theta {
            y0 += -m.theta;
            out.push(j as u32);
        }
        st.v[j] = y0;
        st.i[j] = y1;
    }
    st.prev.clear();
    st.prev.extend_from_slice(out);
}

/// Gradient accumulator (same layouts as the model).
pub struct Grads {
    pub win: Vec<f64>,
    pub wrec: Vec<f64>,
    pub r: Vec<f64>,
    pub b: Vec<f64>,
    pub ww: Vec<f64>,
    pub bits: f64,
    pub spikes: u64,
}
impl Grads {
    pub fn zeros(m: &Model) -> Self {
        Grads {
            win: vec![0.0; m.win.len()],
            wrec: vec![0.0; m.wrec.len()],
            r: vec![0.0; m.r.len()],
            b: vec![0.0; V],
            ww: vec![0.0; m.ww.len()],
            bits: 0.0,
            spikes: 0,
        }
    }
    fn clear(&mut self) {
        for v in [&mut self.win, &mut self.wrec, &mut self.r, &mut self.b, &mut self.ww] {
            v.fill(0.0);
        }
        self.bits = 0.0;
        self.spikes = 0;
    }
    fn add(&mut self, o: &Grads) {
        for (a, b) in [
            (&mut self.win, &o.win),
            (&mut self.wrec, &o.wrec),
            (&mut self.r, &o.r),
            (&mut self.b, &o.b),
            (&mut self.ww, &o.ww),
        ] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
        self.bits += o.bits;
        self.spikes += o.spikes;
    }
}

/// Fast-sigmoid surrogate derivative, 1 / (1 + β|x|)² (the crate's default).
pub const SURROGATE_BETA: f64 = 5.0;

struct Scratch {
    drive: Vec<f64>,
    vpre: Vec<f64>,       // t_len × n
    spikes: Vec<Vec<u32>>, // t_len
    dlog: Vec<[f64; V]>,
    lam_v: Vec<f64>,
    lam_i: Vec<f64>,
    dldd: Vec<f64>,
    gs: Vec<f64>,
}

/// Forward and backward over one truncation window of one stream:
/// characters `chars[pos..pos + t_len]`, targets `chars[pos + 1..=pos + t_len]`.
/// Gradients are of the mean loss in nats over `denom` predictions.
fn window_pass(
    m: &Model,
    chars: &[u8],
    pos: usize,
    t_len: usize,
    st: &mut StreamState,
    g: &mut Grads,
    sc: &mut Scratch,
    denom: f64,
) {
    let n = m.cfg.n;
    let prev0 = st.prev.clone();
    // Forward.
    for t in 0..t_len {
        let (vp, sp) = (&mut sc.vpre[t * n..(t + 1) * n], &mut sc.spikes[t]);
        mirror_step(m, st, chars[pos + t], &mut sc.drive, vp, sp);
        let mut l = [0.0; V];
        m.logits(sp, chars, pos + t, &mut l);
        let target = chars[pos + t + 1] as usize;
        g.bits += softmax_bits64(&mut l, target);
        g.spikes += sp.len() as u64;
        l[target] -= 1.0;
        for x in l.iter_mut() {
            *x /= denom;
        }
        sc.dlog[t] = l;
    }
    // Backward.
    sc.lam_v.fill(0.0);
    sc.lam_i.fill(0.0);
    let rec = !m.wrec.is_empty();
    for t in (0..t_len).rev() {
        let dl = sc.dlog[t];
        // Readout and window gradients.
        for c in 0..V {
            g.b[c] += dl[c];
        }
        for &j in &sc.spikes[t] {
            let row = &mut g.r[j as usize * V..j as usize * V + V];
            for c in 0..V {
                row[c] += dl[c];
            }
        }
        for lag in 0..m.cfg.k.min(pos + t + 1) {
            let o = (lag * V + chars[pos + t - lag] as usize) * V;
            for c in 0..V {
                g.ww[o + c] += dl[c];
            }
        }
        // g_s = Rᵀ·dlogits + W_recᵀ·dldd(t+1) − θ·λ_v
        let last = t + 1 == t_len;
        for j in 0..n {
            let row = &m.r[j * V..j * V + V];
            let mut acc = 0.0;
            for c in 0..V {
                acc += row[c] * dl[c];
            }
            if rec && !last {
                let col = &m.wrec[j * n..(j + 1) * n];
                let mut dot = 0.0;
                for i in 0..n {
                    dot += col[i] * sc.dldd[i];
                }
                acc += dot;
            }
            sc.gs[j] = acc - m.theta * sc.lam_v[j];
        }
        let vp = &sc.vpre[t * n..(t + 1) * n];
        for j in 0..n {
            let d = 1.0 + SURROGATE_BETA * (vp[j] - m.theta).abs();
            let dy_v = sc.lam_v[j] + sc.gs[j] / (d * d);
            let dy_i = sc.lam_i[j];
            sc.dldd[j] = m.delta[j] * dy_v + m.c1[j] * dy_i;
            sc.lam_v[j] = m.alpha[j] * dy_v;
            sc.lam_i[j] = m.gamma[j] * dy_v + m.beta[j] * dy_i;
        }
        for &j in &m.codes[chars[pos + t] as usize] {
            let col = &mut g.win[j as usize * n..(j as usize + 1) * n];
            for i in 0..n {
                col[i] += sc.dldd[i];
            }
        }
        if rec {
            let sprev: &[u32] = if t == 0 { &prev0 } else { &sc.spikes[t - 1] };
            for &j in sprev {
                let col = &mut g.wrec[j as usize * n..(j as usize + 1) * n];
                for i in 0..n {
                    col[i] += sc.dldd[i];
                }
            }
        }
    }
}

pub struct Adam64 {
    m: Vec<f64>,
    v: Vec<f64>,
}
impl Adam64 {
    pub fn new(n: usize) -> Self {
        Adam64 { m: vec![0.0; n], v: vec![0.0; n] }
    }
    pub fn step(&mut self, w: &mut [f64], g: &[f64], lr: f64, scale: f64, t: i32) {
        let (b1, b2) = (0.9f64, 0.999f64);
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        for i in 0..w.len() {
            let gi = g[i] * scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            w[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

/// Stateful truncated-BPTT trainer over `streams` contiguous streams.
pub struct Trainer {
    pub streams: usize,
    pub t_len: usize,
    pub threads: usize,
    pub clip: f64,
    states: Vec<StreamState>,
    grads: Vec<Grads>,
    scratch: Vec<Scratch>,
    opt: [Adam64; 5],
    step: i32,
}

impl Trainer {
    pub fn new(m: &Model, streams: usize, t_len: usize, threads: usize) -> Self {
        let n = m.cfg.n;
        let threads = threads.min(streams).max(1);
        Trainer {
            streams,
            t_len,
            threads,
            clip: 1.0,
            states: vec![StreamState::zeros(n); streams],
            grads: (0..threads).map(|_| Grads::zeros(m)).collect(),
            scratch: (0..threads)
                .map(|_| Scratch {
                    drive: vec![0.0; n],
                    vpre: vec![0.0; t_len * n],
                    spikes: vec![Vec::new(); t_len],
                    dlog: vec![[0.0; V]; t_len],
                    lam_v: vec![0.0; n],
                    lam_i: vec![0.0; n],
                    dldd: vec![0.0; n],
                    gs: vec![0.0; n],
                })
                .collect(),
            opt: [
                Adam64::new(m.win.len()),
                Adam64::new(m.wrec.len()),
                Adam64::new(m.r.len()),
                Adam64::new(V),
                Adam64::new(m.ww.len()),
            ],
            step: 0,
        }
    }

    /// One epoch over `chars` (the training split). Stream `s` covers
    /// `[s·len, (s+1)·len)`; states start from zero each epoch. Returns
    /// (train BPC, mean spike rate).
    pub fn epoch(&mut self, m: &mut Model, chars: &[u8], lr: f64) -> (f64, f64) {
        let n = m.cfg.n;
        let len = chars.len() / self.streams;
        let windows = (len - 1) / self.t_len;
        for st in self.states.iter_mut() {
            *st = StreamState::zeros(n);
        }
        let denom = (self.streams * self.t_len) as f64;
        let per = self.streams.div_ceil(self.threads);
        let (mut bits, mut spikes) = (0.0, 0u64);
        for w in 0..windows {
            {
                let model: &Model = m;
                let t_len = self.t_len;
                std::thread::scope(|scope| {
                    for (c, ((sts, g), sc)) in self
                        .states
                        .chunks_mut(per)
                        .zip(self.grads.iter_mut())
                        .zip(self.scratch.iter_mut())
                        .enumerate()
                    {
                        scope.spawn(move || {
                            g.clear();
                            for (k, st) in sts.iter_mut().enumerate() {
                                let s = c * per + k;
                                let pos = s * len + w * t_len;
                                window_pass(model, chars, pos, t_len, st, g, sc, denom);
                            }
                        });
                    }
                });
            }
            let (head, tail) = self.grads.split_at_mut(1);
            let g = &mut head[0];
            for o in tail.iter() {
                g.add(o);
            }
            bits += g.bits;
            spikes += g.spikes;
            let norm = [&g.win, &g.wrec, &g.r, &g.b, &g.ww]
                .iter()
                .map(|v| v.iter().map(|x| x * x).sum::<f64>())
                .sum::<f64>()
                .sqrt();
            let scale = if norm > self.clip { self.clip / norm } else { 1.0 };
            self.step += 1;
            let t = self.step;
            self.opt[0].step(&mut m.win, &g.win, lr, scale, t);
            self.opt[1].step(&mut m.wrec, &g.wrec, lr, scale, t);
            self.opt[2].step(&mut m.r, &g.r, lr, scale, t);
            self.opt[3].step(&mut m.b, &g.b, lr, scale, t);
            self.opt[4].step(&mut m.ww, &g.ww, lr, scale, t);
        }
        let preds = (windows * self.t_len * self.streams) as f64;
        (bits / preds, spikes as f64 / (preds * n as f64))
    }
}

/// Result of scoring one range through the crate network.
#[derive(Debug, Clone, Copy, Default)]
pub struct Score {
    pub bpc: f64,
    pub se: f64,
    /// Mean hidden spikes per step, as a fraction of neurons.
    pub rate: f64,
}

/// Step the model **through the crate** over `chars[start..]` from a zero
/// state, and score next-character prediction on each `[a, b)` in `ranges`
/// (positions of the *predicted* character; each `a > start`).
pub fn eval_crate(
    m: &Model,
    chars: &[u8],
    start: usize,
    ranges: &[std::ops::Range<usize>],
) -> Result<Vec<Score>, SnnError> {
    let mut net = m.to_network()?;
    let end = ranges.iter().map(|r| r.end).max().unwrap_or(start);
    let mut sums = vec![(0.0f64, 0.0f64, 0u64, 0u64); ranges.len()];
    let mut input = SpikeVec::new(m.d_in);
    for t in start..end - 1 {
        input.clear();
        for &j in &m.codes[chars[t] as usize] {
            input.push(j);
        }
        let out = net.step(&input)?;
        let target = t + 1;
        for (k, r) in ranges.iter().enumerate() {
            if r.contains(&target) {
                let mut l = [0.0; V];
                m.logits(out.active(), chars, t, &mut l);
                let bits = softmax_bits64(&mut l, chars[target] as usize);
                let s = &mut sums[k];
                s.0 += bits;
                s.1 += bits * bits;
                s.2 += 1;
                s.3 += out.active().len() as u64;
            }
        }
    }
    Ok(sums
        .iter()
        .map(|&(s, ss, cnt, sp)| {
            let nn = cnt as f64;
            let mean = s / nn;
            Score {
                bpc: mean,
                se: ((ss / nn - mean * mean).max(0.0) / nn).sqrt(),
                rate: sp as f64 / (nn * m.cfg.n as f64),
            }
        })
        .collect())
}

/// Mean spike rate of the mirror over `chars` from a zero state.
pub fn mirror_rate(m: &Model, chars: &[u8], skip: usize) -> f64 {
    let n = m.cfg.n;
    let mut st = StreamState::zeros(n);
    let (mut drive, mut vpre, mut out) = (vec![0.0; n], vec![0.0; n], Vec::new());
    let mut total = 0u64;
    for (t, &ch) in chars.iter().enumerate() {
        mirror_step(m, &mut st, ch, &mut drive, &mut vpre, &mut out);
        if t >= skip {
            total += out.len() as u64;
        }
    }
    total as f64 / ((chars.len() - skip) * n) as f64
}

/// Build a model whose initial spike rate on `calib` (after `skip` warm-up
/// characters) meets `target`, by 24 steps of geometric bisection on g_in.
pub fn calibrated(
    mut cfg: ModelCfg,
    calib: &[u8],
    skip: usize,
    target: f64,
) -> Result<(Model, f64), SnnError> {
    let (mut lo, mut hi) = (0.1f64, 100.0f64);
    for _ in 0..24 {
        cfg.g_in = (lo * hi).sqrt();
        let m = Model::new(cfg)?;
        if mirror_rate(&m, calib, skip) < target {
            lo = cfg.g_in;
        } else {
            hi = cfg.g_in;
        }
    }
    cfg.g_in = (lo * hi).sqrt();
    let m = Model::new(cfg)?;
    let rate = mirror_rate(&m, calib, skip);
    Ok((m, rate))
}

/// Run-time gate: the mirror and the crate must emit identical spikes.
/// Returns the number of steps compared.
pub fn mirror_matches_crate(m: &Model, chars: &[u8]) -> Result<bool, SnnError> {
    let n = m.cfg.n;
    let mut net = m.to_network()?;
    let mut st = StreamState::zeros(n);
    let (mut drive, mut vpre, mut out) = (vec![0.0; n], vec![0.0; n], Vec::new());
    let mut input = SpikeVec::new(m.d_in);
    let mut any = false;
    for &ch in chars {
        mirror_step(m, &mut st, ch, &mut drive, &mut vpre, &mut out);
        input.clear();
        for &j in &m.codes[ch as usize] {
            input.push(j);
        }
        let c = net.step(&input)?;
        if c.active() != out.as_slice() {
            return Ok(false);
        }
        any |= !out.is_empty();
    }
    Ok(any)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| (rng.next_u64() % V as u64) as u8).collect()
    }

    #[test]
    fn mirror_and_crate_agree_for_every_encoding() {
        let chars = synth(3000, 7);
        for enc in [Encoding::OneHot, Encoding::Sparse, Encoding::Artic] {
            for rho0 in [0.0, 1.0] {
                let cfg = ModelCfg { n: 48, enc, rho0, g_in: 3.0, k: 0, seed: 11 };
                let m = Model::new(cfg).unwrap();
                assert!(mirror_matches_crate(&m, &chars).unwrap(), "{enc:?} rho0={rho0}");
            }
        }
    }

    /// The readout, bias and window gradients are exact (no surrogate on
    /// that path): check them against central finite differences.
    #[test]
    fn smooth_gradients_match_finite_differences() {
        let chars = synth(200, 3);
        let cfg = ModelCfg { n: 24, enc: Encoding::OneHot, rho0: 1.0, g_in: 3.0, k: 3, seed: 5 };
        let mut m = Model::new(cfg).unwrap();
        let mut rng = Rng::new(99);
        for x in m.r.iter_mut().chain(m.b.iter_mut()).chain(m.ww.iter_mut()) {
            *x = 0.3 * rng.normal();
        }
        let (pos, t_len) = (5, 40);
        let loss = |m: &Model| {
            let mut tr = Trainer::new(m, 1, t_len, 1);
            let mut g = Grads::zeros(m);
            let mut st = StreamState::zeros(24);
            window_pass(m, &chars, pos, t_len, &mut st, &mut g, &mut tr.scratch[0], t_len as f64);
            (g.bits * std::f64::consts::LN_2 / t_len as f64, g)
        };
        let (_, g) = loss(&m);
        let h = 1e-6;
        for (which, idx) in [(0, 7usize), (0, 200), (1, 4), (2, 30), (2, 100)] {
            let (an, mut plus, mut minus) = match which {
                0 => (g.r[idx], m.clone(), m.clone()),
                1 => (g.b[idx], m.clone(), m.clone()),
                _ => (g.ww[idx], m.clone(), m.clone()),
            };
            match which {
                0 => { plus.r[idx] += h; minus.r[idx] -= h; }
                1 => { plus.b[idx] += h; minus.b[idx] -= h; }
                _ => { plus.ww[idx] += h; minus.ww[idx] -= h; }
            }
            let fd = (loss(&plus).0 - loss(&minus).0) / (2.0 * h);
            assert!((fd - an).abs() < 1e-7, "param {which}/{idx}: fd {fd} vs {an}");
        }
    }
}
