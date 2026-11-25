# 03 — Rung 2 pre-registration: the reservoir rung

**Registered:** 2026-11-24 · **Status:** frozen before the rung-2
experiment binary exists or any rung-2 number on text has been computed.
Run authorized by the owner on 2026-11-24 ("run rung 2, the reservoir
rung"). Governed by `docs/00-frozen-decisions.md`; corpus, split and
metric exactly as `docs/01-rung1-preregistration.md`.

## Question

Do the spikes of a **frozen, untrained** Koopman SNN carry information
about the next character that additive linear memory cannot — with no
backpropagation through the network at all? Rung 1 measured the additive
ceiling at **3.077** test BPC (logistic readout) and **3.320** (closed-form
least squares). Rung 2 asks whether thresholded features get below those.

## What existed before this registration

`src/reservoir.rs` (the reservoir builder) and `src/bin/probe_speed.rs`
were written first, to size the run. The probe drives reservoirs with
**synthetic uniform-random characters** and reads no corpus text. Its
output (steps per second and spike rate) set the sizes below and nothing
else:

```
n=  512 rho=1 g_in=2: rate 0.0291  123531 steps/s
n= 1024 rho=1 g_in=2: rate 0.0302   38179 steps/s
n= 2048 rho=1 g_in=2: rate 0.0314   10446 steps/s
```

## The reservoir (Koopman rule: every state step is the closed-form propagator)

One `kdmd_snn::KoopmanLayer::lif_hetero` layer, batch 1, stepped once per
character (dt = 1 character), from a zero state at character 0 and run
continuously through the whole 10,000,000-character subset.

- N neurons; membrane time constants log-spaced from 1 to 64 characters
  across neurons; synaptic time constant 0.5 characters; R = 1; threshold 1;
  subtractive reset.
- Input: character `c` is one spike on input line `c` of 27.
  `W_in = g_in · G`, G standard normal.
- Recurrence: `W_rec = ρ · G / √N`; ρ = 0 means none.
- Weights from SplitMix64 + Box–Muller with a stated seed, drawn in a fixed
  order. **Nothing in the reservoir is trained.**
- **Rate calibration (training text only).** `g_in` is set by 24 steps of
  geometric bisection on [0.1, 100] so the mean spike rate over characters
  1,000–20,000 of the training text meets a target rate. The realised rate
  over the full training split is reported.

Features for predicting character t+1: the set of neurons that spiked at
step t (after seeing characters 0..t).

## Grid (stage 1, reservoir seed 0)

N ∈ {512, 1024} × ρ ∈ {0, 1, 3} × target rate ∈ {0.05, 0.2} = 12 reservoirs.

## Readouts

- **S-logistic.** `logits = b + Σ_{spiking n} W[n]`, softmax, cross-entropy.
  Convex in the readout; zero initialisation. The frozen "softmax off
  membrane potential" output: the readout unit's potential is the sum of
  its synaptic weights from the neurons that spiked.
- **SW-logistic.** The same plus the 8-lag one-hot window of rung 1 arm B
  (K = 8). Tests what spikes add *on top of* the additive features.
- **S-ridge.** Spikes (plus a constant) regressed onto the one-hot next
  character by ridge least squares, from exact Gram counts; probabilities by
  clip-and-renormalise. *This is where the DMD-style closed-form fit does
  work in this rung.* λ_rel ∈ {1e-10, 1e-8, 1e-6, 1e-4, 1e-2} (× mean Gram
  diagonal), ε ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4}, chosen on validation. The
  grid extends one decade below rung 1's because rung 1 selected its edge.

Optimiser for the logistic readouts: as rung 1 (Adam, 4,096 contiguous
streams, lr0 for the first half of the epochs, lr0/4 for the next quarter,
lr0/16 for the last quarter), lr0 = 0.02 / √(expected active features) =
0.02 / √(N · target rate [+ 8 for SW]). **Convergence rule, tightened after
rung 1:** each arm is trained at 8 epochs, then 16, then 32, stopping at the
first schedule whose validation BPC improves on the previous one by less
than 0.003. The last value trained is reported; an arm that still improved
by 0.003 or more at 32 epochs is flagged "not converged".

## Stage 2 — seed band

For each of S-logistic and SW-logistic, the stage-1 configuration with the
lowest validation BPC is rebuilt with reservoir seeds 1 and 2 (new weights,
re-calibrated `g_in`) and trained the same way. Reported: mean, minimum and
maximum test BPC over the three seeds.

## Gate (the run is void if it fails)

- **G1.** The window-only K = 8 logistic model, trained by this binary's
  code path, must reproduce rung 1's arm B K = 8 validation BPC of 3.1214 to
  within 0.003.

## The registered result

For S-logistic: **lift** := 3.077 − (3-seed mean test BPC of the selected
configuration). Spiking features are declared to **lift above the linear
ceiling** iff lift > 0.01 and lift > the seed range (max − min). The same
statistic is reported for SW-logistic. S-ridge is reported for all twelve
stage-1 reservoirs against 3.320 (single seed, descriptive).

Also reported, descriptively: the full stage-1 table, realised spike rates,
and arithmetic per character.

## Predictions (stated before running; misses will be recorded)

- The best S-logistic reservoir beats the ceiling: 3-seed mean test BPC in
  2.7–3.0.
- N = 1024 beats N = 512, and target rate 0.2 beats 0.05, at every ρ.
- Recurrence changes S-logistic by less than 0.05 BPC in either direction.
- SW-logistic beats S-logistic for the same reservoir by 0.02–0.15 BPC.
- The best S-ridge value is in 2.9–3.25 (below 3.320).
- Seed range of the selected configuration is below 0.02 BPC.

## Not in this rung

No training of the reservoir, no surrogate gradients, no GRU, no energy
measurement. No tuning of the time-constant range, synaptic constant,
threshold or weight distributions after seeing results.
