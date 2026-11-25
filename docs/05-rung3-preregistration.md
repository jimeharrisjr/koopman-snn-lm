# 05 — Rung 3 pre-registration: the registered character probe

**Registered:** 2026-11-24 · **Status:** frozen before the rung-3
experiment binary or the GRU baseline script exists, and before any trained
network has seen corpus text. Run authorized by the owner on 2026-11-24
("run rung 3 when rung 2 finishes"). Governed by
`docs/00-frozen-decisions.md`; corpus, split and metric exactly as
`docs/01-rung1-preregistration.md`.

## Question

Does a **trained** Koopman SNN, at one step per character, beat the linear
ceiling (3.077 test BPC, rung 1)? That is the frozen kill criterion. Beside
it, descriptively: how far is it from GRUs of known cost?

## What existed before this registration

Committed with this file, written first so the run could be sized:

- `src/encoding.rs` — the three input encodings, including the complete
  articulatory feature table.
- `src/snn.rs` — the model, a truncated-BPTT trainer with a per-character
  loss, and evaluation through the crate.
- `src/bin/pilot3.rs` — a pilot on a **synthetic** random order-2 Markov
  source. It reads no corpus text.

The pilot fixed the learning rate, the stream count and the epoch time,
and nothing else. Its output (N=512, one-hot, 6 epochs, validation bits per
symbol; source entropy 2.875, best previous-symbol-only model 4.575):

| lr0 | streams | recurrence | encoding | valid | final rate |
|---|---|---|---|---|---|
| 0.0003 | 128 | trained | one-hot | 4.103 | 0.249 |
| 0.001 (3 epochs) | 128 | trained | one-hot | 4.047 | 0.212 |
| 0.003 | 128 | trained | one-hot | 3.642 | 0.173 |
| 0.003 | 64 | trained | one-hot | 3.585 | 0.159 |
| 0.01 | 64 | trained | one-hot | 3.484 | 0.098 |
| 0.03 | 64 | trained | one-hot | 3.465 | 0.045 |
| 0.003 | 64 | none | one-hot | 3.941 | 0.115 |
| 0.003 | 64 | trained | articulatory | 3.799 | 0.115 |

About 100,000 characters per second on 16 threads, so about 90 s per epoch
of the 9M-character training split. The trainer learns pair structure that
no additive model can (below 4.575) but stays far from the source entropy.

## Why the trainer lives in this repository

The crate's `Trainer` computes one loss per whole sequence. A language
model needs one loss per character. Rather than change the crate mid-ladder,
`src/snn.rs` holds an in-repo mirror of the layer step for training only:

- The propagator entries (α, β, γ, δ, 1−β) are read out of a
  `kdmd_snn::KoopmanLayer::lif_hetero` layer, never recomputed.
- **Every reported number is computed by stepping `kdmd_snn::Network`
  itself** with the trained weights loaded (`Model::to_network`).
- **Gate G2** requires the mirror and the crate to emit identical spikes.

Contributing a per-step loss upstream is recorded as owed if the ladder
continues past this rung.

## The model

One recurrent LIF layer, N = 512, stepped once per character.

- Neurons exactly as rung 2: membrane time constants log-spaced 1–64
  characters, synaptic 0.5, threshold 1, subtractive reset. Time constants
  are **not** trained (the spectrum is rung 4's question).
- Input: the character's encoding lines spike. `W_in = g_in·G/√(mean
  active lines)`, with `g_in` calibrated on training characters
  1,000–20,000 to an initial spike rate of 0.1 (24-step geometric
  bisection, as rung 2). Trained.
- Recurrence: `W_rec = 0.5·G/√N` initially. Trained.
- Readout: `logits = b + R·s_t`, softmax, from the spikes at step t. Zero
  initialisation. Trained.
- Parameters (one-hot): 289,819.

**Where DMD does work in this rung:** only in the state step, which is the
closed-form propagator. Nothing is fitted by DMD or DMDc here, and no
DMDc compression is applied; rung 6 owns that.

## Training

Truncated BPTT, window 64 characters, 64 contiguous streams with state
carried across windows and reset to zero each epoch. Loss: mean
cross-entropy over all predictions in the window. Surrogate derivative:
fast sigmoid, `1/(1+5|v−θ|)²` (the crate's default). Adam (0.9, 0.999),
global gradient-norm clip 1.0, lr0 = 0.01 for the first half of the epochs,
lr0/4 for the next quarter, lr0/16 for the last quarter. **10 epochs.**
12 threads on the Apple M5 Max. Weights after the last epoch are the model;
there is no early stopping and no checkpoint selection.

An arm is flagged "not converged" if its validation BPC improved by 0.003
or more from epoch 9 to epoch 10. No arm is extended.

**Resume.** Each finished arm appends its result to
`results/rung3-arms.tsv`; a restarted run skips arms already there. This
arm-level resume is the checkpoint mechanism for this rung (arms take
about 20–30 minutes).

## Arms

Stage 1, seed 0 — the registered encoding arms:

| arm | encoding | input lines |
|---|---|---|
| E1 | one-hot | 27, 1 active |
| E2 | sparse 4-of-32, fixed random codes | 32, 4 active |
| E3 | articulatory features (table in `src/encoding.rs`) | 22, 1–7 active |

Stage 2 — the encoding with the lowest stage-1 validation BPC is retrained
with seeds 1 and 2 (new initial weights, re-calibrated `g_in`).

Stage 3, descriptive, on the selected encoding:

- **W8**: the readout also gets rung 1's 8-character additive window;
  seeds 0, 1, 2. Rung 2 found untrained spikes carried the additive part
  badly; this asks whether training removes that.
- **NR**: no recurrence (`W_rec` absent); seed 0.

## Evaluation

Trained weights are loaded into `kdmd_snn::Network` and stepped from a zero
state continuously through all 10,000,000 characters. Reported: validation
and test BPC, naive standard error, and the mean spike rate on the test
split. (Per-epoch validation during training starts from a zero state at
character 8,900,000; it is a monitor only.)

## Gates (the run is void if either fails)

- **G1.** The untrained model with `R = b = 0` scores exactly log2(27) =
  4.7549 BPC through the evaluation path.
- **G2.** For every trained model, the mirror and the crate emit identical
  spikes over training characters 0–20,000.

## Kill criterion (frozen)

**lift** := 3.077 − (3-seed mean test BPC of the selected stage-2
configuration, spikes-only readout). The trained SNN **beats the linear
ceiling** iff lift > 0.01 and lift > the seed range. If it does not, the
ladder stops here.

## GRU baselines (descriptive; fixes the "three GRU sizes" item)

One-layer GRU on one-hot characters with a linear softmax readout, hidden
sizes **64, 128 and 292** (19,611 / 63,771 / 289,107 parameters; 292 is
parameter-matched to the SNN). Same split, same 64 streams × 64-character
truncation with carried state, same 10 epochs and learning-rate schedule
shape, Adam, gradient-norm clip 1.0. lr0 ∈ {0.001, 0.003} chosen per size
on seed-0 validation, then seeds 1 and 2 at the chosen value. Evaluated
from a zero state continuously through the whole subset, as the SNN.

**Placement, changed from the idea note.** The note put GRU baselines on
the batcomputer's GPU. On 2026-11-24 its probe reports "Failed to
initialize NVML: Driver/library version mismatch", which needs a reboot
this project may not perform. The GRUs therefore run in PyTorch 2.15 on the
Mac's CPU (6 threads), concurrently with the SNN. All seeds of both models
run on the Mac.

## Operations ledger

Per predicted character, from the model definition and the measured mean
spike count s̄ on the test split (energy = 4.5 pJ × MACs + 0.9 pJ ×
additions; softmax and gate nonlinearities are reported but not charged,
for either model):

| model | multiply-accumulates | additions |
|---|---|---|
| SNN, registered (readout charged dense, per the frozen Output decision) | 5N + 27N | (active input lines + s̄)·N |
| SNN, spike-driven readout (reported beside it) | 5N | (active input lines + s̄)·N + 27·s̄ |
| SNN W8 | as above | + 216 |
| GRU, hidden H | 3H² + 3H + 27H | 3H |

Reported: both SNN ledgers, the three GRU points, and the energy ratio at
equal BPC obtained by interpolating log-energy linearly in BPC between GRU
sizes. If the SNN's BPC lies outside the GRU range, that is stated and no
ratio is extrapolated. **This is not the frozen win-margin test**, which
needs SNN sizes matched to three quality levels; rung 3 only places one
SNN on the GRU curve.

## Not done in this rung: measured energy

The frozen decisions call for joules per character on a Raspberry Pi from
rung 3. That needs a board this session cannot reach and a power
instrument the owner has not yet named. It is **outstanding**, recorded
here as a known gap and not silently dropped. It is confirmatory; the
ledger above is the registered measure.

## Predictions (stated before running; misses will be recorded)

- The kill criterion is passed: selected spikes-only 3-seed mean test BPC
  in 2.2–2.6.
- One-hot is best or within 0.02 of best; articulatory is worst.
- W8 improves on spikes-only by less than 0.05.
- NR is worse than the recurrent model by more than 0.1.
- Seed range of the selected configuration below 0.03.
- Final test-split spike rate between 0.03 and 0.2.
- GRU-292 reaches 1.75–1.95; GRU-64 reaches 2.0–2.25.
- **The SNN loses to GRU-64 on BPC** while costing more on the registered
  ledger (about 100 nJ against about 64 nJ per character). That is, no
  quality-per-operation advantage at this rung.

## Not in this rung

No trained time constants, no second layer, no extra simulation steps per
character, no distillation, no DMDc compression, no tuning of anything
above after seeing a number on text.
