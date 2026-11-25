# 06 — Rung 3 results: the registered character probe

**Run:** 2026-11-24 to 2026-11-25, Apple M5 Max. SNN:
`cargo run --release --bin rung3`, 12 threads, 3 h 12 min
(22:11:45–01:23:45 EST). GRUs: `baselines/run_gru.sh`, PyTorch 2.15 on the
CPU, about 1 h, concurrent with the SNN. Protocol:
`docs/05-rung3-preregistration.md` (commit `4ba7500`, pushed before the run
code, which is commit `49ccfbf`). Raw output: `results/rung3-results.md`,
`results/rung3-arms.tsv`, `results/rung3-gru.tsv`; logs alongside.

## Registered result: the kill criterion is passed

**The trained SNN beats the linear ceiling.** One-hot input, spikes-only
readout, N = 512, three seeds: **2.342 test BPC** (2.340, 2.353, 2.335;
seed range 0.019) against 3.077. Lift +0.735, far above 0.01 and above the
seed range. The ladder may continue.

Both gates passed. G1: the untrained model with a zero readout scores
4.7549 = log2 27 through the evaluation path. G2: for every one of the
nine trained models, the training mirror and `kdmd_snn::Network` emitted
identical spikes. Every number below was computed by stepping the crate's
network.

## All SNN arms

| arm | seeds | test BPC (mean) | seed range | test spike rate | parameters |
|---|---|---|---|---|---|
| One-hot, spikes only **(selected)** | 3 | **2.342** | 0.019 | 0.089 | 289,819 |
| Sparse 4-of-32 | 1 | 2.355 | n/a | 0.088 | 292,379 |
| Articulatory features | 1 | 2.423 | n/a | 0.085 | 287,259 |
| One-hot + 8-character window (W8) | 3 | 2.316 | 0.012 | 0.079 | 295,651 |
| One-hot, no recurrence (NR) | 1 | 2.618 | n/a | 0.123 | 27,675 |

Naive test standard errors are 0.003 per run. No arm was flagged "not
converged" (largest epoch 9→10 gain 0.0028, limit 0.003).

## GRU baselines

lr0 = 0.003 was selected for every size (seed-0 validation).

| GRU hidden | parameters | test BPC (3-seed mean) | seed range |
|---|---|---|---|
| 64 | 19,611 | 2.236 | 0.005 |
| 128 | 63,771 | 2.018 | 0.009 |
| 292 (parameter-matched) | 289,107 | 1.850 | 0.012 |

## Operations ledger

Per predicted character, from the model definitions and the measured mean
spike count (45.4 of 512 neurons per step for the selected SNN). Energy at
4.5 pJ per multiply-accumulate and 0.9 pJ per addition.

| model | test BPC | MACs | additions | energy, nJ per character |
|---|---|---|---|---|
| SNN, registered ledger (readout charged dense) | 2.342 | 16,384 | 23,737 | **95.1** |
| SNN, spike-driven readout | 2.342 | 2,560 | 24,962 | 34.0 |
| SNN W8, registered ledger | 2.316 | 16,384 | 21,490 | 93.1 |
| SNN NR, registered ledger | 2.618 | 16,384 | 512 | 74.2 |
| GRU-64 | 2.236 | 14,208 | 192 | **64.1** |
| GRU-128 | 2.018 | 52,992 | 384 | 238.8 |
| GRU-292 | 1.850 | 264,552 | 876 | 1,191.3 |

**No quality-per-operation advantage at this rung.** On the registered
ledger the SNN is worse than GRU-64 on both axes: 0.107 BPC behind and
about 1.5 times the energy. The SNN's BPC is outside the range the GRUs
span, so, as registered, no equal-BPC energy ratio is extrapolated.

Two things the table also shows, neither of which is a win:

- **The readout dominates the registered SNN ledger.** 13,824 of its
  16,384 MACs are the readout charged dense, as the frozen Output decision
  requires. Counted as spike-driven additions instead, the SNN costs 34 nJ,
  about half of GRU-64. It is still 0.107 BPC worse, and a GRU small
  enough to cost 34 nJ was not run, so this does not show an advantage
  either.
- **Against the parameter-matched GRU the trade is steep.** GRU-292 is
  0.49 BPC better at 12.5 times the registered energy. Parameter matching
  is the wrong axis for the goal; matching on cost, the small GRU wins.

## Predictions, scored

| Prediction | Outcome |
|---|---|
| Kill criterion passed; 3-seed mean in 2.2–2.6 | Hit (2.342) |
| One-hot best or within 0.02 of best; articulatory worst | Hit (one-hot best by 0.015 on test; articulatory 0.084 behind) |
| W8 improves on spikes-only by less than 0.05 | Hit (0.027) |
| NR worse than recurrent by more than 0.1 | Hit (0.278) |
| Seed range below 0.03 | Hit (0.019) |
| Test spike rate in 0.03–0.2 | Hit (0.089) |
| GRU-292 in 1.75–1.95; GRU-64 in 2.0–2.25 | Hit (1.850; 2.236) |
| The SNN loses to GRU-64 on BPC while costing more on the registered ledger | Hit |

All eight held, including the prediction of a loss to the GRU.

## What it means

- **Training is worth 0.68 BPC over the best untrained reservoir**
  (2.342 at N = 512 against rung 2's 3.019 at N = 1024), and clears the
  linear ceiling by 0.73. Surrogate-gradient training through the
  closed-form propagator works on text at one step per character.
- **Trained recurrence is where the gain is.** Removing it costs 0.28 BPC.
  Rung 2 found random recurrence useless; trained recurrence is not.
- **Training removes most of the additive-path deficit.** In rung 2 the
  window added 0.33 to untrained spikes. On the trained network it adds
  0.027. The window lowers the spike rate a little (0.089 to 0.079), so it
  is nearly free on the ledger, but it does not change the picture.
- **Input encoding is not a lever.** One-hot is best; the sparse code is
  within a seed range of it and costs four times the input additions;
  articulatory features are clearly worse.
- **The SNN is behind a GRU of similar cost.** This is the rung's most
  important number for the project's goal. The gap to close is 0.107 BPC
  at 1.5 times the energy on the registered ledger.

## Limits of this result

- **One SNN size.** The frozen win-margin test needs SNN sizes matched to
  three quality levels. This rung places a single SNN on the GRU curve.
- **The SNN is probably under-trained by its schedule.** Validation fell
  at every learning-rate drop (for example 2.499 → 2.417 at epoch 6) and
  was still inching down at the end. The convergence flag only tests the
  last epoch. More epochs or a different schedule might gain something;
  none was tried, as registered.
- **The GRU grid selected its edge.** lr0 = 0.003 won at every size and
  was the largest value offered. The GRUs may be under-tuned, which would
  make the true gap larger, not smaller.
- **The SNN's learning rate came from a synthetic pilot**, not from text.
- **Stage 1 used one seed.** The one-hot/sparse difference (0.015) is
  inside the seed range (0.019), so "one-hot is best" is not established
  against the sparse code; "articulatory is worst" is.
- **No measured energy.** The Raspberry Pi measurement the frozen
  decisions call for from rung 3 was not made (no board access, no named
  power instrument). The ledger is arithmetic from model definitions.

## Deviations from the registration

None in procedure. Recorded for the record:

- The training forward pass is an in-repo mirror of the crate's layer
  step, as registered; a per-character loss in `kdmd-snn` itself is owed
  upstream if the ladder continues.
- GRUs ran on the Mac's CPU instead of the batcomputer's GPU, as
  registered, because that GPU's driver needs a reboot.
