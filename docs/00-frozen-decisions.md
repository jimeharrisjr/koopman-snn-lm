# 00 — Frozen decisions

**Frozen:** 2026-11-24 · **Status:** decisions on record; **no runs
authorized**. This file is committed before any model, training or
evaluation code exists, so that the protocol-before-results order is
verifiable from the commit history.

Source: the discussion document "Toward a Koopman-SNN Language Model"
(the plan of record) and, for the character-probe rung, the earlier idea
note `docs/ideas/snn-char-lm-probe.md` in the `kdmd-SNN` checkout. Where the
two disagree, the plan of record wins.

## Decisions

| Topic | Frozen decision |
|---|---|
| Goal | Beat a matched GRU on quality per operation. "Matched" means the same parameter count, training text and context length. |
| Stretch goal | Match SpikeGPT's enwik8 result (1.262 bits per character at 46M parameters) at a smaller size. |
| Win margin | At most half the GRU's energy-weighted operations per character at equal bits per character; measured at three quality levels set by a small, a medium and a large GRU; the gap must exceed the seed band of both models over 3 to 5 seeds; raw operation counts and spike rates are reported alongside. |
| Energy model | 4.5 pJ per multiply-accumulate; 0.9 pJ per addition on a synapse whose input spiked (SpikeGPT's accounting, for comparability). |
| Measured energy | From rung 3 onward, joules per character on one Raspberry Pi, CPU only, both models: (load power − idle power) × run time ÷ characters. Confirmatory; the operations ledger is the registered criterion for early rungs. |
| Koopman rule | Every neuron-state step goes through the closed-form propagator. Each rung names where DMD or DMDc does work. Efficiency is reported with and without the DMDc-derived compression. |
| Output | Softmax off membrane potential; its dense arithmetic is charged to the SNN's side of the ledger as multiply-accumulates. |
| Simulation steps | One step per token is the default; any extra step must buy measurable quality. |
| Teacher | None through rung 6. At rung 7, the matched GRU first, then a small transformer trained on the same corpus. No externally pretrained teacher. |
| Corpus | A 5 to 10 MB text8 subset for rungs 1 to 6; TinyStories at rung 7; full enwik8 at rung 8. Baselines are recomputed on the identical subset and split; published numbers are never quoted for a subset. |
| Home | This repository, depending on `kdmd-snn`; changes the crate needs go upstream. |

## The ladder

Order of testing if and when runs are authorized. A failure stops the climb.

1. **Linear ceiling.** Closed-form additive model (DMDc / reduced-rank
   regression on decayed token history); bits per character with no SNN.
2. **Reservoir rung.** Frozen random Koopman SNN with log-spaced tau,
   one-hot characters, ridge-regression readout.
3. **Registered character probe.** Trained SNN at one step per token, with
   encoding arms (one-hot, sparse k-of-n, articulatory features) and the
   operations ledger beside bits per character. Kill criterion: the SNN
   must beat the rung-1 linear ceiling by more than the seed band.
4. **Spectrum arms.** Default, log-spaced, resonator and Legendre-operator
   initialisations.
5. **Token-dependent decay.** Diagonal `A(u)` against threshold-only.
6. **Reduced-order inference.** DMDc compression of trained recurrent
   layers; quality against operations per token as rank and spike rate fall.
7. **Small-vocabulary words with a teacher.** Subword vocabulary on
   TinyStories, hierarchical output, distillation.
8. **The scale decision.** f32 or GPU port and SpikeGPT's enwik8 setting,
   only if rungs 3 to 7 show a quality-per-operation advantage.

Optional side task, gating nothing: spoken digit sequences built from SHD
recordings, predicting the next digit.

## Still to be fixed before a registered run

Each of these must be committed here before the run it governs:

- the exact text8 subset and train/validation/test split;
- the three GRU sizes;
- the seed count within 3 to 5, and the machine per seed;
- the instrument for measuring board power;
- checkpoint and resume, which the crate does not yet have.

Fixed since the freeze:

- 2026-11-24: `kdmd-snn` pinned to revision `e8480f5` of
  `koopman-based-SNN` (replacing the local path dependency). Any bump is a
  commit of its own, made before the run it governs.
