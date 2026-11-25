# 02 — Rung 1 results: the linear ceiling

**Run:** 2026-11-24, Apple M5 Max, `cargo run --release --bin rung1`,
390 s wall-clock. Protocol: `docs/01-rung1-preregistration.md` (commit
`e3e0e4c`, pushed before this code existed). Raw output:
`results/rung1-results.md`; per-epoch log: `results/rung1-log.txt`.

## Registered result

**The linear ceiling is 3.077 test BPC (± 0.003)** — arm B, window-additive
logistic regression at K = 64 (validation 3.116). The closed-form
least-squares value (arm C, K = 64) is 3.320 test BPC.

Both gates passed: G1 |C(K=1) − bigram| = 0.0000, G2 |B(K=1) − bigram| =
0.0002 (limit 0.01).

## All arms

| arm | valid BPC | test BPC | note |
|---|---|---|---|
| A unigram | 4.117 | 4.116 | δ = 1 |
| A bigram | 3.437 | 3.431 | δ = 1 |
| B window logistic K=1 | 3.437 | 3.431 | |
| B K=2 | 3.245 | 3.221 | |
| B K=4 | 3.149 | 3.116 | |
| B K=8 | 3.121 | 3.083 | |
| B K=16 | 3.118 | 3.079 | |
| B K=32 | 3.117 | 3.078 | |
| **B K=64** | **3.116** | **3.077** | the ceiling |
| C window least squares K=1 | 3.437 | 3.431 | |
| C K=2 | 3.420 | 3.396 | |
| C K=4 | 3.373 | 3.342 | |
| C K=8 | 3.357 | 3.322 | |
| C K=16 | 3.356 | 3.321 | |
| C K=32 | 3.355 | 3.321 | |
| C K=64 | 3.354 | 3.320 | closed-form value |
| D4 trace logistic | 3.217 | 3.189 | upper bound, see below |
| D8 trace logistic | 3.175 | 3.144 | upper bound |
| D16 trace logistic | 3.162 | 3.131 | upper bound |

Naive test standard errors are 0.002–0.003 for every arm.

## What the protocol's convergence check did

B at K = 16 did not move on the doubled schedule (3.1175 → 3.1176). D8
improved by 0.0076, above the 0.003 limit, so — as registered — every B
and D arm was re-run on the doubled schedule and those are the numbers
above.

**Caveat on arm D.** The protocol registered one doubling, not a
convergence guarantee. D16 still improved by 0.009 between the two
schedules, so the arm-D values are upper bounds on their convex optima,
not the optima. Overlapping exponential traces are badly conditioned
features and Adam closes on them slowly. This does not touch the
registered ceiling: arm B is converged, and arm D is the more constrained
model.

## Predictions, scored

| Prediction | Outcome |
|---|---|
| Unigram 4.0–4.2, bigram 3.3–3.6 | Hit (4.116, 3.431) |
| Arm B plateaus by K ≈ 8–16 | Hit (3.083 at K=8, 3.077 at K=64) |
| Arm B best test BPC in 2.6–3.0 | **Miss** — 3.077; additive memory is weaker than predicted |
| Arm C worse than B by 0.05–0.3 at the same K | Hit for K ≥ 2 (0.18–0.24); trivially zero at K=1 |
| D16 within 0.05 of best B | **Miss**, narrowly — 0.054 (and D16 is not converged) |
| D4 clearly worse | Hit (0.11 behind) |

## What it means

- **Additive memory buys little.** Going from the previous character alone
  (3.431) to 64 characters of additive history (3.077) gains 0.35 BPC, and
  0.348 of that is in by K = 8. A linear state with a linear readout
  cannot use long context, however long its memory.
- **The closed-form fit is not the ceiling.** Least squares onto one-hot
  targets loses 0.24 BPC to the logistic fit of the same features. The DMD
  operator is the right *state* model here but the wrong *loss* for
  probabilities. Selection also sat at the edge of the ridge grid
  (λ_rel = 1e-8 for every K), so a smaller ridge may do marginally better;
  the 0.24 gap is far larger than that could close.
- **A threshold-free Koopman layer sits just above the ceiling.** Sixteen
  decay rates reach 3.131 against 3.077 for the unconstrained window — the
  diagonal-`A` trace bank is nearly as good as free lag weights, as it
  should be.
- **The bar for rung 3 is now a number.** A spiking network must beat
  3.077 test BPC by more than its seed band, or the ladder stops. It is a
  low bar by design: it tests only whether the threshold adds
  composition. The competitive test remains the matched GRU.

## Arithmetic per character (descriptive)

| arm | arithmetic per predicted character |
|---|---|
| B K=8 | 8 table lookups, 216 additions, one 27-way softmax |
| B K=64 | 64 lookups, 1,728 additions, one softmax |
| D16 | 432 trace updates, up to about 11,000 multiply-accumulates, one softmax |

## Deviations from the registration

None in procedure. Two things to carry forward: selection hit the edge of
a registered grid in arm C (λ_rel) and in arm A (δ = 1), and arm D's
convergence was not established.
