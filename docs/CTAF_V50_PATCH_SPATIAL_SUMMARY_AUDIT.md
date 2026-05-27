# CTAF V50 Patch Spatial Summary Audit

## 1. Stage 5 / V49 Summary

- mechanism 3-view frontends cannot replace full_5view
- hard-normal consistency compresses q99 but fails as deployable method
- current mainline remains full_5view + latent_summary_knn

## 2. Hypothesis

- global pooling may erase spatial local evidence
- patch 2x2 summary preserves coarse spatial structure

## 3. Overall Comparison

| score_mode | mean_hmean | mean_pAUC | mean_KS | std_hmean | std_pAUC | std_KS |
|---|---:|---:|---:|---:|---:|---:|
| global_latent_summary_knn | 0.553898 | 0.545689 | 0.305714 | 0.045965 | 0.039661 | 0.291352 |
| patch_2x2_summary_knn | 0.513617 | 0.515664 | 0.295238 | 0.038487 | 0.020766 | 0.157595 |

## 4. Seed-Panel Table

| seed | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| 13711 | global_latent_summary_knn | 0.648035 | 0.644211 | 0.200000 |
| 13711 | patch_2x2_summary_knn | 0.527098 | 0.503684 | 0.200000 |
| 13721 | global_latent_summary_knn | 0.615785 | 0.603158 | 0.120000 |
| 13721 | patch_2x2_summary_knn | 0.525139 | 0.509474 | 0.140000 |
| 13731 | global_latent_summary_knn | 0.658401 | 0.629474 | 0.180000 |
| 13731 | patch_2x2_summary_knn | 0.545174 | 0.501579 | 0.160000 |

## 5. Per-Machine hmean / pAUC / KS

| machine | score_mode | mean hmean | mean pAUC | mean KS |
|---|---|---:|---:|---:|
| bearing | global_latent_summary_knn | 0.640740 | 0.625614 | 0.166667 |
| bearing | patch_2x2_summary_knn | 0.532470 | 0.504912 | 0.166667 |
| fan | global_latent_summary_knn | 0.578161 | 0.565965 | 0.993333 |
| fan | patch_2x2_summary_knn | 0.574986 | 0.530175 | 0.540000 |
| gearbox | global_latent_summary_knn | 0.540506 | 0.523333 | 0.180000 |
| gearbox | patch_2x2_summary_knn | 0.526482 | 0.527544 | 0.466667 |
| slider | global_latent_summary_knn | 0.512138 | 0.526667 | 0.186667 |
| slider | patch_2x2_summary_knn | 0.494569 | 0.527895 | 0.240000 |
| toycar | global_latent_summary_knn | 0.541548 | 0.515088 | 0.213333 |
| toycar | patch_2x2_summary_knn | 0.473921 | 0.512982 | 0.166667 |
| toytrain | global_latent_summary_knn | 0.518576 | 0.514912 | 0.200000 |
| toytrain | patch_2x2_summary_knn | 0.531380 | 0.531228 | 0.340000 |
| valve | global_latent_summary_knn | 0.545615 | 0.548246 | 0.200000 |
| valve | patch_2x2_summary_knn | 0.461512 | 0.474912 | 0.146667 |

## 6. Tail Table

| machine | score_mode | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | global_latent_summary_knn | 0.1168 | 0.1282 | 0.0929 | 0.1030 | 0.1067 | 0.0700 |
| gearbox | patch_2x2_summary_knn | 1.2617 | 1.2839 | 1.1489 | 1.2210 | 0.1200 | 0.0433 |
| slider | global_latent_summary_knn | 0.1267 | 0.1494 | 0.1085 | 0.1187 | 0.1267 | 0.0167 |
| slider | patch_2x2_summary_knn | 1.0386 | 1.1099 | 0.8866 | 0.9573 | 0.1233 | 0.0333 |
| toycar | global_latent_summary_knn | 0.0827 | 0.0870 | 0.0744 | 0.0781 | 0.0933 | 0.0300 |
| toycar | patch_2x2_summary_knn | 1.2633 | 1.2772 | 1.1933 | 1.2305 | 0.0800 | 0.0367 |
| toytrain | global_latent_summary_knn | 0.0928 | 0.1193 | 0.0687 | 0.0757 | 0.0933 | 0.0333 |
| toytrain | patch_2x2_summary_knn | 1.1210 | 1.1385 | 1.0260 | 1.0886 | 0.1233 | 0.0700 |
| fan | global_latent_summary_knn | 0.2295 | 0.2382 | 0.1711 | 0.2209 | 0.1900 | 0.1267 |
| fan | patch_2x2_summary_knn | 1.1339 | 1.1509 | 0.9863 | 1.0770 | 0.0967 | 0.0133 |
| bearing | global_latent_summary_knn | 0.1570 | 0.1660 | 0.1305 | 0.1624 | 0.2900 | 0.2133 |
| bearing | patch_2x2_summary_knn | 1.1207 | 1.1458 | 0.9648 | 1.0333 | 0.0533 | 0.0400 |
| valve | global_latent_summary_knn | 0.1976 | 0.2150 | 0.1611 | 0.1828 | 0.1700 | 0.0700 |
| valve | patch_2x2_summary_knn | 1.0898 | 1.1100 | 0.9291 | 0.9903 | 0.0033 | 0.0000 |

## 7. q99 Inflation Check

- q99 inflation is checked by relative tail-width inflation:
  `(q99 - q95) / q95` under patch_2x2 versus global summary.
- machines with >10% relative tail-width inflation under patch_2x2: `0`

## 8. Final Verdict

`patch_2x2 negative`

Success criteria check:

- mean hmean >= global - 0.003: `FAIL`
- mean pAUC >= global: `FAIL`
- >= 2 bottleneck machines improve TPR@99n: `3/5`
- normal q99 does not inflate clearly: `PASS`
- bearing/valve do not collapse: `FAIL`
- toycar does not clearly degrade: `PASS`

Interpretation:

- patch summaries do help some bottleneck `TPR@99n` values
- but they lose too much on aggregate `hmean / pAUC`
- and `bearing / valve` collapse makes the method non-viable under the fixed rule

This is a no-training audit. No checkpoint was modified. No grid search, no score fusion, and no per-machine selection were used.
