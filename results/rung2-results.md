# Rung 2 — the reservoir rung

## Gate

G1 window-only K=8: valid 3.1214 (rung 1: 3.1214), |diff| = 0.0000, 16 epochs  PASS

## Stage 1 (reservoir seed 0)

| N | rho | target rate | g_in | realised rate | S valid | S test | S epochs | SW valid | SW test | SW epochs | ridge valid | ridge test | ridge (lambda_rel, eps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 0 | 0.05 | 2.4827 | 0.0501 | 3.3121 | 3.2540 | 16 | 2.9410 | 2.8870 | 16 | 3.5447 | 3.4964 | 1e-2, 3e-3 |
| 512 | 0 | 0.2 | 11.0575 | 0.2001 | 3.2790 | 3.2121 | 16 | 2.8987 | 2.8438 | 16 | 3.5441 | 3.4951 | 1e-10, 1e-2 |
| 512 | 1 | 0.05 | 2.4483 | 0.0500 | 3.3130 | 3.2560 | 16 | 2.9411 | 2.8902 | 16 | 3.5376 | 3.4918 | 1e-2, 3e-3 |
| 512 | 1 | 0.2 | 10.9987 | 0.2002 | 3.2806 | 3.2145 | 16 | 2.9003 | 2.8461 | 16 | 3.5451 | 3.4967 | 1e-4, 1e-2 |
| 512 | 3 | 0.05 | 2.2733 | 0.0501 | 3.3805 | 3.3240 | 16 | 2.9736 | 2.9251 | 16 | 3.5789 | 3.5343 | 1e-2, 3e-3 |
| 512 | 3 | 0.2 | 10.6695 | 0.2002 | 3.2934 | 3.2311 | 16 | 2.9136 | 2.8646 | 16 | 3.5531 | 3.5054 | 1e-2, 3e-3 |
| 1024 | 0 | 0.05 | 2.4958 | 0.0502 | 3.1032 | 3.0360 | 16 | 2.8345 | 2.7775 | 16 | 3.4052 | 3.3490 | 1e-2, 3e-3 |
| 1024 | 0 | 0.2 | 11.2408 | 0.2007 | 3.0945 | 3.0221 | 16 | 2.7983 | 2.7424 | 32 | 3.4166 | 3.3591 | 1e-10, 1e-2 |
| 1024 | 1 | 0.05 | 2.4676 | 0.0502 | 3.1214 | 3.0568 | 16 | 2.8496 | 2.7966 | 16 | 3.4170 | 3.3643 | 1e-2, 3e-3 |
| 1024 | 1 | 0.2 | 11.1664 | 0.2007 | 3.0972 | 3.0269 | 16 | 2.7999 | 2.7456 | 32 | 3.4174 | 3.3627 | 1e-10, 1e-2 |
| 1024 | 3 | 0.05 | 2.3060 | 0.0503 | 3.1909 | 3.1342 | 16 | 2.9034 | 2.8515 | 16 | 3.4634 | 3.4176 | 1e-2, 3e-3 |
| 1024 | 3 | 0.2 | 10.8186 | 0.2007 | 3.1175 | 3.0490 | 16 | 2.8206 | 2.7665 | 32 | 3.4318 | 3.3764 | 1e-10, 1e-2 |

## Stage 2 (seed band) and registered result

S-logistic: selected N=1024 rho=0 target rate=0.2 (seed-0 valid 3.0945)
  seed 0: test 3.0221 (SE 0.0028)
  seed 1: test 3.0198 (SE 0.0028), valid 3.0874, rate 0.2006, 16 epochs
  seed 2: test 3.0148 (SE 0.0028), valid 3.0791, rate 0.2006, 16 epochs
  mean 3.0189  min 3.0148  max 3.0221  seed range 0.0073
  lift over the linear ceiling (3.077) = +0.0581  =>  LIFTS above the ceiling

SW-logistic: selected N=1024 rho=0 target rate=0.2 (seed-0 valid 2.7983)
  seed 0: test 2.7424 (SE 0.0029)
  seed 1: test 2.7427 (SE 0.0029), valid 2.8009, rate 0.2006, 32 epochs
  seed 2: test 2.7443 (SE 0.0029), valid 2.7981, rate 0.2006, 32 epochs
  mean 2.7431  min 2.7424  max 2.7443  seed range 0.0019
  lift over the linear ceiling (3.077) = +0.3339  =>  LIFTS above the ceiling

S-ridge (seed 0, descriptive): best N=1024 rho=0 target rate=0.05: test 3.3490 (valid 3.4052) vs closed-form ceiling 3.32: -0.0290
