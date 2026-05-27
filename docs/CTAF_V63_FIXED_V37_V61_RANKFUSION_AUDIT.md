# CTAF V63 Fixed V37/V61 Rank-Fusion Audit

## 1. V63A Summary

- V63A established that fixed 0.5/0.5 rank-normalized fusion is the only allowed fusion candidate.
- No weight tuning, no per-machine selection, no oracle candidate selection.
- V63 materializes that fixed route into frozen per-seed/per-machine score files.

## 2. Alignment Confirmation

- V37 files: present
- V61 files: present
- y_test alignment: PASS
- domain alignment: PASS
- sample order alignment: PASS

## 3. Fixed Fusion Formula

`C2 = 0.5 * rank_norm(V37_score) + 0.5 * rank_norm(V61_score)`

- rank_norm is computed independently per machine and per seed
- no weight tuning
- no per-machine selection
- no oracle selection

## 4. 3-Seed Comparison

| route | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V37 | 0.553898 | 0.006537 | 0.545689 | 0.003628 | 0.305714 | 0.008411 | 0.152857 | 0.080000 |
| V61 | 0.562946 | 0.007941 | 0.549323 | 0.004589 | 0.262857 | 0.018516 | 0.161429 | 0.078571 |
| V63_fixed_rankfusion | 0.562753 | 0.007588 | 0.549762 | 0.002258 | 0.294286 | 0.011664 | 0.152381 | 0.089524 |

## 5. Per-Machine Comparison

| machine | route | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@99n | std TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bearing | V37 | 0.640740 | 0.018146 | 0.625614 | 0.016980 | 0.166667 | 0.033993 | 0.213333 | 0.054365 |
| bearing | V61 | 0.631871 | 0.013329 | 0.619474 | 0.007959 | 0.146667 | 0.024944 | 0.180000 | 0.016330 |
| bearing | V63_fixed_rankfusion | 0.635347 | 0.005952 | 0.624211 | 0.007735 | 0.153333 | 0.033993 | 0.206667 | 0.028674 |
| fan | V37 | 0.578161 | 0.002774 | 0.565965 | 0.015568 | 0.993333 | 0.009428 | 0.126667 | 0.049216 |
| fan | V61 | 0.593863 | 0.027561 | 0.556316 | 0.023066 | 1.000000 | 0.000000 | 0.096667 | 0.054365 |
| fan | V63_fixed_rankfusion | 0.593493 | 0.014492 | 0.579474 | 0.005035 | 1.000000 | 0.000000 | 0.156667 | 0.009428 |
| gearbox | V37 | 0.540506 | 0.019606 | 0.523333 | 0.003655 | 0.180000 | 0.065320 | 0.070000 | 0.014142 |
| gearbox | V61 | 0.528223 | 0.006378 | 0.520175 | 0.003338 | 0.133333 | 0.024944 | 0.050000 | 0.008165 |
| gearbox | V63_fixed_rankfusion | 0.541192 | 0.008937 | 0.522632 | 0.000985 | 0.160000 | 0.058878 | 0.050000 | 0.008165 |
| slider | V37 | 0.512138 | 0.024512 | 0.526667 | 0.005018 | 0.186667 | 0.047140 | 0.016667 | 0.016997 |
| slider | V61 | 0.543858 | 0.011113 | 0.528246 | 0.011613 | 0.113333 | 0.024944 | 0.043333 | 0.033993 |
| slider | V63_fixed_rankfusion | 0.530093 | 0.017434 | 0.528772 | 0.007147 | 0.166667 | 0.024944 | 0.030000 | 0.014142 |
| toycar | V37 | 0.541548 | 0.019190 | 0.515088 | 0.012248 | 0.213333 | 0.033993 | 0.030000 | 0.014142 |
| toycar | V61 | 0.512443 | 0.012902 | 0.497368 | 0.009912 | 0.180000 | 0.074833 | 0.016667 | 0.016997 |
| toycar | V63_fixed_rankfusion | 0.536633 | 0.004106 | 0.510965 | 0.005909 | 0.193333 | 0.067987 | 0.013333 | 0.018856 |
| toytrain | V37 | 0.518576 | 0.005448 | 0.514912 | 0.009220 | 0.200000 | 0.028284 | 0.033333 | 0.004714 |
| toytrain | V61 | 0.517540 | 0.027137 | 0.503509 | 0.012050 | 0.113333 | 0.033993 | 0.016667 | 0.016997 |
| toytrain | V63_fixed_rankfusion | 0.520799 | 0.014566 | 0.499123 | 0.003225 | 0.226667 | 0.047140 | 0.013333 | 0.009428 |
| valve | V37 | 0.545615 | 0.029034 | 0.548246 | 0.011469 | 0.200000 | 0.056569 | 0.070000 | 0.021602 |
| valve | V61 | 0.612828 | 0.013157 | 0.620175 | 0.019816 | 0.153333 | 0.047140 | 0.146667 | 0.044969 |
| valve | V63_fixed_rankfusion | 0.581714 | 0.024680 | 0.583158 | 0.006535 | 0.160000 | 0.028284 | 0.156667 | 0.012472 |

## 6. TPR@99n Comparison

- V37 mean TPR@99n: `0.080000`
- V61 mean TPR@99n: `0.078571`
- V63 fixed rank fusion mean TPR@99n: `0.089524`
- bottleneck machines improved over V61: `2/5`

## 7. pAUC / hmean Comparison

- V37: `hmean=0.553898`, `pAUC=0.545689`
- V61: `hmean=0.562946`, `pAUC=0.549323`
- V63: `hmean=0.562753`, `pAUC=0.549762`

## 8. q99 Note

V63 uses rank-normalized fusion scores.
Therefore V63 q95/q99 values live on the rank-normalized score scale and must not be directly compared against the raw-score q95/q99 values of V37 or V61.

## 9. Complementarity Summary

| machine | mean spearman_r | mean top-q99 normal jaccard | mean missed-anomaly jaccard | sum(V37-detect/V61-miss) | sum(V61-detect/V37-miss) |
|---|---:|---:|---:|---:|---:|
| bearing | 0.805679 | 0.000000 | 0.868191 | 22 | 12 |
| fan | 0.899431 | 0.000000 | 0.898569 | 19 | 10 |
| gearbox | 0.411351 | 0.000000 | 0.958547 | 9 | 3 |
| slider | 0.341406 | 0.666667 | 0.952721 | 3 | 11 |
| toycar | 0.207830 | 0.333333 | 0.953333 | 9 | 5 |
| toytrain | 0.382547 | 0.000000 | 0.956498 | 9 | 4 |
| valve | 0.580686 | 0.000000 | 0.851076 | 10 | 33 |

## 10. Final Verdict

`V63 frozen route PASS`

## 11. Confirmation

- no training occurred
- no original V37/V61 score file was modified
- no fusion weight tuning was performed
- no per-machine selection was used
- no oracle selection was used
- no V64 opened
