# Rung 1 — linear ceiling

## Convergence check (validation BPC)

B K=16: x1 3.1175  x2 3.1176  gain -0.0000
D8    : x1 3.1820  x2 3.1745  gain +0.0076
=> gain exceeds 0.003: all B and D arms re-run on the doubled schedule

## Results

| arm | valid BPC | test BPC | test SE | note |
|---|---|---|---|---|
| A unigram | 4.1174 | 4.1159 | 0.0017 | delta=1 |
| A bigram | 3.4371 | 3.4307 | 0.0022 | delta=1 |
| B window logistic K=1 | 3.4372 | 3.4309 | 0.0022 | schedule x2 |
| B window logistic K=2 | 3.2451 | 3.2210 | 0.0025 | schedule x2 |
| B window logistic K=4 | 3.1487 | 3.1159 | 0.0026 | schedule x2 |
| B window logistic K=8 | 3.1214 | 3.0834 | 0.0026 | schedule x2 |
| B window logistic K=16 | 3.1176 | 3.0786 | 0.0026 | schedule x2 |
| B window logistic K=32 | 3.1167 | 3.0777 | 0.0026 | schedule x2 |
| B window logistic K=64 | 3.1161 | 3.0769 | 0.0026 | schedule x2 |
| C window least squares K=1 | 3.4371 | 3.4307 | 0.0022 | lambda_rel=1e-8 eps=1e-4 |
| C window least squares K=2 | 3.4200 | 3.3962 | 0.0023 | lambda_rel=1e-8 eps=3e-3 |
| C window least squares K=4 | 3.3730 | 3.3423 | 0.0024 | lambda_rel=1e-8 eps=3e-3 |
| C window least squares K=8 | 3.3566 | 3.3224 | 0.0024 | lambda_rel=1e-8 eps=3e-3 |
| C window least squares K=16 | 3.3555 | 3.3208 | 0.0024 | lambda_rel=1e-8 eps=3e-3 |
| C window least squares K=32 | 3.3550 | 3.3206 | 0.0024 | lambda_rel=1e-8 eps=3e-3 |
| C window least squares K=64 | 3.3543 | 3.3201 | 0.0024 | lambda_rel=1e-8 eps=3e-3 |
| D4 trace logistic (4 taus) | 3.2165 | 3.1892 | 0.0025 | schedule x2 |
| D8 trace logistic (8 taus) | 3.1745 | 3.1435 | 0.0025 | schedule x2 |
| D16 trace logistic (16 taus) | 3.1623 | 3.1308 | 0.0025 | schedule x2 |

## Gates

G1 |C(K=1) - bigram| = 0.0000  PASS
G2 |B(K=1) - bigram| = 0.0002  PASS

## Registered result

Linear ceiling: 3.0769 test BPC (+/- 0.0026), arm "B window logistic K=64" (valid 3.1161)
Closed-form value: 3.3201 test BPC, arm "C window least squares K=64" (valid 3.3543)
