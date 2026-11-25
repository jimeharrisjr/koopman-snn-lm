//! Shared numerics for rung 2 onward (rung 1's binary keeps its own as-run
//! copies): BPC statistics, softmax, Adam, and a dense Cholesky solve.

use crate::corpus::V;

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub sum: f64,
    pub sumsq: f64,
    pub n: f64,
}
impl Stats {
    pub fn add(&mut self, bits: f64) {
        self.sum += bits;
        self.sumsq += bits * bits;
        self.n += 1.0;
    }
    pub fn bpc(&self) -> f64 {
        self.sum / self.n
    }
    /// Naive standard error (ignores serial correlation).
    pub fn se(&self) -> f64 {
        let m = self.bpc();
        ((self.sumsq / self.n - m * m).max(0.0) / self.n).sqrt()
    }
}

/// Softmax in place; returns −log2 p(target).
pub fn softmax_bits(logits: &mut [f32; V], target: usize) -> f64 {
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

pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}
impl Adam {
    pub fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    pub fn step(&mut self, w: &mut [f32], g: &[f32], lr: f32) {
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

/// lr0 for the first half of the epochs, lr0/4 for the next quarter, lr0/16
/// for the last quarter (rung 1's schedule, stated for any epoch count).
pub fn lr_at(epoch: usize, epochs: usize, lr0: f32) -> f32 {
    if epoch < epochs / 2 {
        lr0
    } else if epoch < 3 * epochs / 4 {
        lr0 / 4.0
    } else {
        lr0 / 16.0
    }
}

/// Solve A X = B for symmetric positive-definite A (dense row-major d×d) by
/// Cholesky. `rhs` is d×nrhs row-major. None if A is not positive definite.
pub fn chol_solve(a: &[f64], d: usize, rhs: &[f64], nrhs: usize) -> Option<Vec<f64>> {
    let mut l = vec![0f64; d * d];
    for j in 0..d {
        let rowj: Vec<f64> = l[j * d..j * d + j].to_vec();
        let s = a[j * d + j] - rowj.iter().map(|x| x * x).sum::<f64>();
        if !(s > 0.0) {
            return None;
        }
        let ljj = s.sqrt();
        l[j * d + j] = ljj;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chol_solve_recovers_known_solution() {
        // A = M Mᵀ + I for a fixed M; X chosen; B = A X.
        let d = 5;
        let m: Vec<f64> = (0..d * d).map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0).collect();
        let mut a = vec![0f64; d * d];
        for i in 0..d {
            for j in 0..d {
                a[i * d + j] = (0..d).map(|k| m[i * d + k] * m[j * d + k]).sum::<f64>()
                    + if i == j { 1.0 } else { 0.0 };
            }
        }
        let x: Vec<f64> = (0..d * 2).map(|i| i as f64 * 0.3 - 1.0).collect();
        let mut b = vec![0f64; d * 2];
        for i in 0..d {
            for r in 0..2 {
                b[i * 2 + r] = (0..d).map(|k| a[i * d + k] * x[k * 2 + r]).sum();
            }
        }
        let got = chol_solve(&a, d, &b, 2).unwrap();
        for (g, w) in got.iter().zip(&x) {
            assert!((g - w).abs() < 1e-10, "{g} vs {w}");
        }
    }

    #[test]
    fn chol_solve_rejects_indefinite() {
        assert!(chol_solve(&[1.0, 2.0, 2.0, 1.0], 2, &[1.0, 1.0], 1).is_none());
    }
}
