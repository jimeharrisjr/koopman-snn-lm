# 01 — Rung 1 pre-registration: the linear ceiling

**Registered:** 2026-11-24 · **Status:** frozen before any rung-1 code
exists or any number has been computed. Run authorized by the owner on
2026-11-24 ("run rung 1, the linear ceiling"). Governed by
`docs/00-frozen-decisions.md`.

## Question

How many bits per character (BPC) of next-character prediction are
reachable by a model whose logits are *additive over past characters* —
that is, by linear memory `x' = Ax + Bu` with one-hot characters as `u`
and a linear readout, with no threshold and no spikes? The answer is the
**linear ceiling**: the number every spiking rung must beat (rung 3 kill
criterion).

## Corpus and split (fixes the "exact text8 subset and split" item)

- Source: `text8` from `http://mattmahoney.net/dc/text8.zip`;
  SHA-256 of the 100,000,000-byte file:
  `6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7`.
- Subset: the first 10,000,000 bytes; SHA-256
  `e5bc9580d197612032534e743716f44813367eaabd73165f4691b7a9d3490eb0`.
  27 symbols (space, a–z).
- Split, contiguous: train `[0, 9,000,000)`, validation
  `[9,000,000, 9,500,000)`, test `[9,500,000, 10,000,000)`.
- Context may reach back across a split boundary (history only; the
  predicted character always lies inside its own split).
- This subset and split govern rungs 1 to 6.

## Metric

BPC = mean of −log2 p(true next character). Reported on validation and
test, with the naive standard error (per-character standard deviation ÷
√n). Every hyperparameter choice below is made on **validation**; test is
read once per arm, after selection.

## Arms

**A — count references (sanity anchors).** Unigram and bigram, add-δ
smoothed, δ ∈ {1e-3, 1e-2, 1e-1, 1} chosen on validation.

**B — window-additive logistic regression (the ceiling proper).**
`logits_t = b + Σ_{k=1..K} W_k[c_{t−k}]`, softmax, cross-entropy.
K ∈ {1, 2, 4, 8, 16, 32, 64}. Convex, so the optimum is unique; zero
initialisation, no seed.

**C — window least squares (the closed-form, DMD-style fit).** The same
K-lag one-hot features regressed onto the one-hot next character by ridge
least squares: the delay-embedded (Hankel) DMD one-step operator, formed
from Gram matrices of lagged co-occurrence counts because the snapshot
matrix is too large to hold. Same K list. Ridge
λ = λ_rel × mean diagonal of the Gram matrix, λ_rel ∈ {1e-8, 1e-6, 1e-4,
1e-2}. Outputs become probabilities by clipping at ε and renormalising,
ε ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4}. (λ_rel, ε) chosen on validation.
*This is where DMD does work in this rung* (frozen Koopman rule).

**D — decayed-trace state with logistic readout (a threshold-free Koopman
layer).** State = a bank of exponentially decayed one-hot traces,
`x_j ← α_j x_j + (1 − α_j) u`, α_j = exp(−1/τ_j) (τ in characters; τ = 0
means α = 0, the previous character itself). Diagonal `A`, exact
propagator. `logits = b + Σ_j W_j x_j`. Banks:
- D4: τ ∈ {0, 2, 8, 32}
- D8: τ ∈ {0, 1, 2, 4, 8, 16, 32, 64}
- D16: τ = 0 plus 15 values log-spaced from 0.5 to 128.

## Optimiser for B and D (fixed; not a scientific degree of freedom)

Adam (β 0.9 / 0.999, ε 1e-8), no weight decay. 4,096 parallel contiguous
streams over the training text (stream length ⌊9,000,000 / 4,096⌋ = 2,197
steps; one step = one minibatch of 4,096 characters). 8 epochs: learning
rate lr0 for epochs 1–4, lr0/4 for 5–6, lr0/16 for 7–8, with
lr0 = 0.02 / √K (arm B) or 0.02 / √(bank size) (arm D). Arm D traces are
warmed on the 512 characters preceding each stream start.

**Convergence check.** B at K = 16 and D8 are re-run with every phase
doubled (16 epochs). If either validation BPC improves by more than 0.003,
all B and D arms are re-run on the doubled schedule and those numbers are
the registered ones. Because the problems are convex, more optimisation
only approaches the unique optimum; this is not tuning.

## Gates (the run is void if either fails)

- **G1.** C at K = 1 must match the bigram reference: |ΔBPC| ≤ 0.01 on
  test (least squares from one one-hot to one-hot is the conditional
  frequency table).
- **G2.** B at K = 1 must match the bigram reference: |ΔBPC| ≤ 0.01 on
  test (the convex optimum of a one-lag softmax is the bigram MLE).

## The registered result

**Linear ceiling** := the test BPC of the arm among B and D with the
lowest validation BPC. Arm C is reported beside it as the closed-form
value. Also reported descriptively: BPC against K for B and C, and
arithmetic per character for each arm.

## Predictions (stated before running; misses will be recorded)

- Unigram 4.0–4.2 BPC; bigram 3.3–3.6.
- Arm B plateaus by K ≈ 8–16, with best test BPC in 2.6–3.0.
- Arm C is worse than arm B at the same K by 0.05–0.3 BPC.
- D16 lands within 0.05 BPC of the best arm-B value; D4 is clearly worse.

## Not in this rung

No spiking network, no GRU, no energy measurement, no seeds (every arm
is deterministic). Nothing here is compared against a published number.
