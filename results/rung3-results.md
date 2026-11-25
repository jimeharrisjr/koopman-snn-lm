# Rung 3 — the registered character probe

## Gates

G1 untrained, zero readout: 4.7549 BPC (log2 27 = 4.7549)  PASS
G2 mirror == crate spikes: PASS for every trained model (the run aborts otherwise)

## Stage 1 — encoding arms (seed 0)

| arm | params | g_in | valid | test | test SE | test spike rate | active input lines | epoch 9→10 gain |
|---|---|---|---|---|---|---|---|---|
| S-onehot-s0 | 289819 | 4.1812 | 2.3827 | 2.3396 | 0.0031 | 0.0879 | 1.00 | 0.0025 |
| S-sparse-s0 | 292379 | 3.5263 | 2.4002 | 2.3548 | 0.0030 | 0.0881 | 4.00 | 0.0019 |
| S-artic-s0 | 287259 | 3.0951 | 2.4663 | 2.4234 | 0.0030 | 0.0850 | 4.13 | 0.0018 |

Selected encoding: one-hot

## Stage 2 — seed band and kill criterion

| arm | params | g_in | valid | test | test SE | test spike rate | active input lines | epoch 9→10 gain |
|---|---|---|---|---|---|---|---|---|
| S-onehot-s0 | 289819 | 4.1812 | 2.3827 | 2.3396 | 0.0031 | 0.0879 | 1.00 | 0.0025 |
| S-onehot-s1 | 289819 | 4.0882 | 2.3947 | 2.3534 | 0.0030 | 0.0916 | 1.00 | 0.0028 |
| S-onehot-s2 | 289819 | 4.4575 | 2.3774 | 2.3346 | 0.0031 | 0.0864 | 1.00 | 0.0005 |

Spikes-only, selected encoding: tests ["2.3396", "2.3534", "2.3346"]
  mean 2.3425  min 2.3346  max 2.3534  seed range 0.0188  mean rate 0.0886
  lift over the linear ceiling (3.077) = +0.7345  =>  BEATS the ceiling

## Stage 3 — descriptive

| arm | params | g_in | valid | test | test SE | test spike rate | active input lines | epoch 9→10 gain |
|---|---|---|---|---|---|---|---|---|
| W8-onehot-s0 | 295651 | 4.1812 | 2.3514 | 2.3198 | 0.0031 | 0.0787 | 1.00 | 0.0023 |
| W8-onehot-s1 | 295651 | 4.0882 | 2.3570 | 2.3199 | 0.0031 | 0.0793 | 1.00 | 0.0017 |
| W8-onehot-s2 | 295651 | 4.4575 | 2.3447 | 2.3082 | 0.0031 | 0.0796 | 1.00 | 0.0004 |
| NR-onehot-s0 | 27675 | 4.1960 | 2.6743 | 2.6175 | 0.0030 | 0.1234 | 1.00 | 0.0010 |

W8 (spikes + 8-character window): tests ["2.3198", "2.3199", "2.3082"]
  mean 2.3159  min 2.3082  max 2.3199  seed range 0.0117  mean rate 0.0792
  lift over the linear ceiling (3.077) = +0.7611  =>  BEATS the ceiling

NR (no recurrence), seed 0: test 2.6175 against 2.3396 recurrent (difference +0.2779)
