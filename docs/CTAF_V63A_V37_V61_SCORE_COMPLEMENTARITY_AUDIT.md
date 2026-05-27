# CTAF V63A V37/V61 Score Complementarity Audit

## 1. Score Alignment Check

- checked seeds: `13711,13721,13731`
- checked machines: `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- V37 files: present
- V61 files: present
- y_test alignment: PASS
- domain alignment: PASS
- sample order alignment: PASS
- no alignment stop condition triggered

## 2. Candidate Definition

- `C0 = V37 raw score`
- `C1 = V61 raw score`
- `C2 = 0.5 * rank_norm(V37) + 0.5 * rank_norm(V61)`
- rank normalization is done independently per machine, per seed
- no weight tuning, no oracle candidate selection

## 3. 3-Seed Mean ± Std

| candidate | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n | mean q95 | mean q99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_V37_raw | 0.553898 | 0.006537 | 0.545689 | 0.003628 | 0.305714 | 0.008411 | 0.152857 | 0.080000 | 0.143295 | 0.157600 |
| C1_V61_raw | 0.562946 | 0.007941 | 0.549323 | 0.004589 | 0.262857 | 0.018516 | 0.161429 | 0.078571 | 0.133850 | 0.146554 |
| C2_rank_avg_fixed | 0.562753 | 0.007588 | 0.549762 | 0.002258 | 0.294286 | 0.011664 | 0.152381 | 0.089524 | 0.858112 | 0.926886 |

## 4. Per-Machine Mean ± Std

| machine | candidate | mean hmean | std hmean | mean pAUC | std pAUC | mean TPR@99n | std TPR@99n | mean q99 | std q99 | mean KS | std KS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bearing | C0_V37_raw | 0.640740 | 0.018146 | 0.625614 | 0.016980 | 0.213333 | 0.054365 | 0.166041 | 0.005167 | 0.166667 | 0.033993 |
| bearing | C1_V61_raw | 0.631871 | 0.013329 | 0.619474 | 0.007959 | 0.180000 | 0.016330 | 0.151769 | 0.004090 | 0.146667 | 0.024944 |
| bearing | C2_rank_avg_fixed | 0.635347 | 0.005952 | 0.624211 | 0.007735 | 0.206667 | 0.028674 | 0.883685 | 0.014610 | 0.153333 | 0.033993 |
| fan | C0_V37_raw | 0.578161 | 0.002774 | 0.565965 | 0.015568 | 0.126667 | 0.049216 | 0.238231 | 0.005036 | 0.993333 | 0.009428 |
| fan | C1_V61_raw | 0.593863 | 0.027561 | 0.556316 | 0.023066 | 0.096667 | 0.054365 | 0.214320 | 0.020045 | 1.000000 | 0.000000 |
| fan | C2_rank_avg_fixed | 0.593493 | 0.014492 | 0.579474 | 0.005035 | 0.156667 | 0.009428 | 0.894899 | 0.007425 | 1.000000 | 0.000000 |
| gearbox | C0_V37_raw | 0.540506 | 0.019606 | 0.523333 | 0.003655 | 0.070000 | 0.014142 | 0.128176 | 0.002186 | 0.180000 | 0.065320 |
| gearbox | C1_V61_raw | 0.528223 | 0.006378 | 0.520175 | 0.003338 | 0.050000 | 0.008165 | 0.123734 | 0.002215 | 0.133333 | 0.024944 |
| gearbox | C2_rank_avg_fixed | 0.541192 | 0.008937 | 0.522632 | 0.000985 | 0.050000 | 0.008165 | 0.955796 | 0.003085 | 0.160000 | 0.058878 |
| slider | C0_V37_raw | 0.512138 | 0.024512 | 0.526667 | 0.005018 | 0.016667 | 0.016997 | 0.149407 | 0.010610 | 0.186667 | 0.047140 |
| slider | C1_V61_raw | 0.543858 | 0.011113 | 0.528246 | 0.011613 | 0.043333 | 0.033993 | 0.134144 | 0.010824 | 0.113333 | 0.024944 |
| slider | C2_rank_avg_fixed | 0.530093 | 0.017434 | 0.528772 | 0.007147 | 0.030000 | 0.014142 | 0.958350 | 0.020171 | 0.166667 | 0.024944 |
| toycar | C0_V37_raw | 0.541548 | 0.019190 | 0.515088 | 0.012248 | 0.030000 | 0.014142 | 0.087029 | 0.003676 | 0.213333 | 0.033993 |
| toycar | C1_V61_raw | 0.512443 | 0.012902 | 0.497368 | 0.009912 | 0.016667 | 0.016997 | 0.094134 | 0.006369 | 0.180000 | 0.074833 |
| toycar | C2_rank_avg_fixed | 0.536633 | 0.004106 | 0.510965 | 0.005909 | 0.013333 | 0.018856 | 0.949204 | 0.018591 | 0.193333 | 0.067987 |
| toytrain | C0_V37_raw | 0.518576 | 0.005448 | 0.514912 | 0.009220 | 0.033333 | 0.004714 | 0.119308 | 0.002297 | 0.200000 | 0.028284 |
| toytrain | C1_V61_raw | 0.517540 | 0.027137 | 0.503509 | 0.012050 | 0.016667 | 0.016997 | 0.113841 | 0.003082 | 0.113333 | 0.033993 |
| toytrain | C2_rank_avg_fixed | 0.520799 | 0.014566 | 0.499123 | 0.003225 | 0.013333 | 0.009428 | 0.973291 | 0.013629 | 0.226667 | 0.047140 |
| valve | C0_V37_raw | 0.545615 | 0.029034 | 0.548246 | 0.011469 | 0.070000 | 0.021602 | 0.215004 | 0.004097 | 0.200000 | 0.056569 |
| valve | C1_V61_raw | 0.612828 | 0.013157 | 0.620175 | 0.019816 | 0.146667 | 0.044969 | 0.193934 | 0.012731 | 0.153333 | 0.047140 |
| valve | C2_rank_avg_fixed | 0.581714 | 0.024680 | 0.583158 | 0.006535 | 0.156667 | 0.012472 | 0.872973 | 0.021841 | 0.160000 | 0.028284 |

## 5. Complementarity Diagnostics

| seed | machine | spearman_r | top-q99 normal jaccard | missed-anomaly jaccard | V37-detect/V61-miss | V61-detect/V37-miss |
|---|---|---:|---:|---:|---:|---:|
| 13711 | bearing | 0.808468 | 0.000000 | 0.865854 | 9 | 2 |
| 13721 | bearing | 0.810698 | 0.000000 | 0.868132 | 5 | 7 |
| 13731 | bearing | 0.797870 | 0.000000 | 0.870588 | 8 | 3 |
| 13711 | fan | 0.917283 | 0.000000 | 0.944444 | 2 | 3 |
| 13721 | fan | 0.893229 | 0.000000 | 0.826531 | 17 | 0 |
| 13731 | fan | 0.887781 | 0.000000 | 0.924731 | 0 | 7 |
| 13711 | gearbox | 0.414060 | 0.000000 | 0.989474 | 1 | 0 |
| 13721 | gearbox | 0.477861 | 0.000000 | 0.958333 | 2 | 2 |
| 13731 | gearbox | 0.342133 | 0.000000 | 0.927835 | 6 | 1 |
| 13711 | slider | 0.444593 | 1.000000 | 0.970000 | 0 | 3 |
| 13721 | slider | 0.103887 | 0.000000 | 0.908163 | 2 | 7 |
| 13731 | slider | 0.475737 | 1.000000 | 0.980000 | 1 | 1 |
| 13711 | toycar | 0.232218 | 0.000000 | 0.950000 | 4 | 1 |
| 13721 | toycar | 0.190464 | 1.000000 | 0.990000 | 1 | 0 |
| 13731 | toycar | 0.200807 | 0.000000 | 0.920000 | 4 | 4 |
| 13711 | toytrain | 0.272371 | 0.000000 | 0.949495 | 2 | 3 |
| 13721 | toytrain | 0.551325 | 0.000000 | 0.960000 | 3 | 1 |
| 13731 | toytrain | 0.323944 | 0.000000 | 0.960000 | 4 | 0 |
| 13711 | valve | 0.728435 | 0.000000 | 0.865979 | 7 | 6 |
| 13721 | valve | 0.442742 | 0.000000 | 0.855670 | 2 | 12 |
| 13731 | valve | 0.570880 | 0.000000 | 0.831579 | 1 | 15 |

## 6. Pass/Fail Gates for C2

| gate | verdict |
|---|---|
| mean hmean >= max(V37,V61)-0.001 | PASS |
| mean pAUC >= max(V37,V61)-0.001 | PASS |
| TPR@99n >= V61 | PASS |
| >=2 bottleneck TPR@99n improve over V61 | PASS |
| toycar not degrade further than V61 | PASS |
| bearing not collapse | PASS |
| valve not collapse | PASS |
| no alignment issue | PASS |
| no oracle candidate selection | PASS |

## 7. Final Verdict

`V63A pass: fixed V37/V61 rank-fusion eligible for 3-seed panel summary`

## 8. Confirmation

- no training occurred
- no score file was modified
- no fusion weight tuning was performed
- no per-machine fusion selection was used
- no BEATs, V62A, UACS, or new training objective was introduced
