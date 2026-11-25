# 04 — Rung 2 results: the reservoir rung

**Run:** 2026-11-24, Apple M5 Max, `cargo run --release --bin rung2`,
about 56 minutes wall-clock (20:46:58–21:42:29 EST). Protocol:
`docs/03-rung2-preregistration.md` (commit `8aedc0b`, pushed before the
experiment binary existed). Raw output: `results/rung2-results.md`;
per-epoch log: `results/rung2-log.txt`.

## Registered result

**Frozen, untrained spikes lift above the linear ceiling.**

| readout | selected reservoir | 3-seed mean test BPC | seed range | lift over 3.077 | verdict |
|---|---|---|---|---|---|
| S-logistic (spikes only) | N=1024, ρ=0, rate 0.2 | 3.019 | 0.007 | +0.058 | lifts |
| SW-logistic (spikes + 8-character window) | N=1024, ρ=0, rate 0.2 | 2.743 | 0.002 | +0.334 | lifts |

Both lifts exceed 0.01 and the seed range, as registered. Naive test
standard errors are 0.003 per run.

Gate G1 passed: the window-only K = 8 model reproduced rung 1's validation
BPC of 3.1214 exactly (|diff| = 0.0000; test 3.083).

**S-ridge does not beat the closed-form ceiling.** The best of the twelve
reservoirs reached 3.349 test BPC (N=1024, ρ=0, rate 0.05) against 3.320
for rung 1's closed-form window fit. All twelve are above 3.320.

## Stage 1 (reservoir seed 0), test BPC

| N | ρ | target rate | S-logistic | SW-logistic | S-ridge |
|---|---|---|---|---|---|
| 512 | 0 | 0.05 | 3.254 | 2.887 | 3.496 |
| 512 | 0 | 0.2 | 3.212 | 2.844 | 3.495 |
| 512 | 1 | 0.05 | 3.256 | 2.890 | 3.492 |
| 512 | 1 | 0.2 | 3.215 | 2.846 | 3.497 |
| 512 | 3 | 0.05 | 3.324 | 2.925 | 3.534 |
| 512 | 3 | 0.2 | 3.231 | 2.865 | 3.505 |
| 1024 | 0 | 0.05 | 3.036 | 2.778 | 3.349 |
| **1024** | **0** | **0.2** | **3.022** | **2.742** | 3.359 |
| 1024 | 1 | 0.05 | 3.057 | 2.797 | 3.364 |
| 1024 | 1 | 0.2 | 3.027 | 2.746 | 3.363 |
| 1024 | 3 | 0.05 | 3.134 | 2.852 | 3.418 |
| 1024 | 3 | 0.2 | 3.049 | 2.767 | 3.376 |

Validation values, calibrated `g_in`, realised rates and selected ridge
hyperparameters are in `results/rung2-results.md`. Realised spike rates
over the training split were within 0.001 of target in every reservoir.

## Convergence

No arm was flagged "not converged". Every S-logistic arm and the low-rate
and N=512 SW arms stopped at 16 epochs (gain from 8 to 16 below 0.003).
The three N=1024, rate-0.2 SW arms went to 32 epochs, where the gain from
16 to 32 was 0.001–0.002. The stage-2 reseeds behaved the same way.

## Predictions, scored

| Prediction | Outcome |
|---|---|
| Best S-logistic 3-seed mean in 2.7–3.0 | **Miss**, narrowly — 3.019. It beats the ceiling but by less than predicted |
| N=1024 beats N=512, and rate 0.2 beats 0.05, at every ρ | Hit for both logistic readouts (not for S-ridge, where rate has no consistent effect) |
| Recurrence changes S-logistic by less than 0.05 | **Miss** — true at ρ=1 (at most 0.021) and at rate 0.2 (at most 0.027), but ρ=3 at rate 0.05 costs 0.070 (N=512) and 0.098 (N=1024) |
| SW beats S for the same reservoir by 0.02–0.15 | **Miss** — the gap is 0.26–0.40, far larger |
| Best S-ridge in 2.9–3.25 | **Miss** — 3.349, worse than the closed-form window fit |
| Seed range below 0.02 | Hit (0.007 and 0.002) |

Four of six predictions missed. The registered verdict went the predicted
way; the sizes of the effects did not.

## What it means

- **The threshold adds information additive memory does not have.** With
  the 8-character window supplying the additive part, 1,024 untrained
  spiking neurons take the model from 3.083 to 2.743 test BPC, a gain of
  0.34. That is as large as the whole gain from one character of context
  to 64 in rung 1 (0.35). No gradient passed through the network.
- **Spikes alone are a poor code for the additive part.** S-logistic lifts
  only 0.058 over the ceiling, while the same spikes add 0.34 on top of
  the window. Random thresholded projections carry the non-additive
  information well and the plain identity of recent characters badly.
  A design that keeps a cheap additive path beside the spiking path is
  suggested by this, and is not yet tested as a trained network.
- **Width is still paying.** Doubling N from 512 to 1,024 gained 0.19
  (S) and 0.10 (SW). Neither curve has flattened, so these numbers are
  not the reservoir limit. N=2048 was left out for run time, not because
  it was expected to stop helping.
- **Random recurrence did not help anywhere.** ρ=1 is indistinguishable
  from none, and ρ=3 is worse in all eight comparisons. The lift here
  comes from thresholding a decaying sum of inputs, not from recurrent
  dynamics. Whether *trained* recurrence helps is rung 3's question.
- **Least squares onto one-hot targets is the wrong loss, again.** The
  ridge readout loses 0.33 BPC to the logistic readout of the same
  spikes. This repeats rung 1's finding with a larger gap. The closed-form
  DMD-style fit did real work in this rung and lost; a closed-form
  readout will need a different target than one-hot least squares.

## Limits of this result

- The comparison is against *additive* models only. No non-additive
  baseline (a trigram count model, a small GRU) was registered, so this
  rung does not say the reservoir is competitive, only that it is not
  linear. The matched GRU is the competitive test and comes later.
- The lift may come from short-range conjunctions (pairs and triples of
  recent characters) and not from long context. Nothing here separates
  the two.
- Stage 1 used one reservoir seed; only the selected configuration has a
  seed band.

## Arithmetic per character (descriptive)

For the selected reservoir (N=1024, ρ=0, rate 0.2, about 205 spikes per
step):

| part | arithmetic per character |
|---|---|
| Input | 1 input spike: 1,024 additions |
| State step | closed-form propagator on a 2-state neuron: about 3 multiplies each, about 3,100 in all; 1,024 threshold comparisons |
| S readout | about 205 spikes × 27 = about 5,500 additions, one 27-way softmax |
| SW readout | the above plus 8 lookups and 216 additions |

For comparison, rung 1's ceiling model (window K=64) costs 1,728
additions and one softmax. The reservoir buys its 0.33 BPC with roughly
four times the additions plus about 3,100 multiplies. These are counts
from the model definition, not measurements, and no energy claim is made
in this rung.

## Deviations from the registration

None in procedure. To carry forward:

- **Ridge selection sat on grid edges again.** λ_rel was chosen at 1e-2
  (upper edge) or 1e-10 (lower edge) in eleven of twelve reservoirs, and ε
  at 1e-2 (upper edge) in five. The validation surface is evidently flat
  in λ; the 0.33 gap to the logistic readout is far larger than a wider
  grid could close.
- **`src/common.rs` duplicates rung 1's helpers** (Adam, softmax, Cholesky)
  so that `src/bin/rung1.rs` stays frozen as run. Gate G1 confirms the
  shared code path reproduces rung 1 to four decimals.
