# CTAF V38 Density Scoring Audit

## 1. V16 Seed-Panel Mean Baseline

```text
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. V37 latent_summary_knn Baseline

```text
hmean = 0.553898
pAUC  = 0.545689
KS    = 0.305733
bottleneck TPR@99n improvement = 0/5
```

## 3. Seed-Panel Table

| seed | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| 13711 | latent_summary_knn | 0.563084 | 0.550451 | 0.3143 |
| 13711 | density_gaussian_shrinkage | 0.538825 | 0.531128 | 0.2771 |
| 13711 | density_gmm_diag_k4 | 0.532953 | 0.518722 | 0.3743 |
| 13721 | latent_summary_knn | 0.550205 | 0.544962 | 0.2943 |
| 13721 | density_gaussian_shrinkage | 0.537818 | 0.529925 | 0.3257 |
| 13721 | density_gmm_diag_k4 | 0.532560 | 0.520075 | 0.3429 |
| 13731 | latent_summary_knn | 0.548404 | 0.541654 | 0.3086 |
| 13731 | density_gaussian_shrinkage | 0.534898 | 0.536165 | 0.3229 |
| 13731 | density_gmm_diag_k4 | 0.528192 | 0.524737 | 0.3429 |

## 4. Aggregate Mean/Std

| score_mode | mean_hmean | mean_pAUC | mean_KS | std_hmean | std_pAUC | std_KS |
|---|---:|---:|---:|---:|---:|---:|
| latent_summary_knn | 0.553898 | 0.545689 | 0.305733 | 0.008006 | 0.004443 | 0.010304 |
| density_gaussian_shrinkage | 0.537180 | 0.532406 | 0.308567 | 0.002040 | 0.003310 | 0.027287 |
| density_gmm_diag_k4 | 0.531235 | 0.521178 | 0.353367 | 0.002643 | 0.003156 | 0.018129 |

## 5. Per-Machine hmean / pAUC / KS

| machine | score_mode | mean hmean | mean pAUC | mean KS |
|---|---|---:|---:|---:|
| bearing | latent_summary_knn | 0.640740 | 0.625614 | 0.166667 |
| bearing | density_gaussian_shrinkage | 0.597129 | 0.612281 | 0.153333 |
| bearing | density_gmm_diag_k4 | 0.587686 | 0.605789 | 0.100000 |
| fan | latent_summary_knn | 0.578161 | 0.565965 | 0.993333 |
| fan | density_gaussian_shrinkage | 0.606815 | 0.589298 | 1.000000 |
| fan | density_gmm_diag_k4 | 0.525459 | 0.530351 | 1.000000 |
| gearbox | latent_summary_knn | 0.540506 | 0.523333 | 0.180000 |
| gearbox | density_gaussian_shrinkage | 0.537457 | 0.515263 | 0.200000 |
| gearbox | density_gmm_diag_k4 | 0.529802 | 0.516491 | 0.260000 |
| slider | latent_summary_knn | 0.512138 | 0.526667 | 0.186667 |
| slider | density_gaussian_shrinkage | 0.553246 | 0.509474 | 0.240000 |
| slider | density_gmm_diag_k4 | 0.542030 | 0.514035 | 0.140000 |
| toycar | latent_summary_knn | 0.541548 | 0.515088 | 0.213333 |
| toycar | density_gaussian_shrinkage | 0.567873 | 0.527018 | 0.120000 |
| toycar | density_gmm_diag_k4 | 0.537496 | 0.491579 | 0.406667 |
| toytrain | latent_summary_knn | 0.518576 | 0.514912 | 0.200000 |
| toytrain | density_gaussian_shrinkage | 0.509565 | 0.488772 | 0.240000 |
| toytrain | density_gmm_diag_k4 | 0.517078 | 0.501930 | 0.286667 |
| valve | latent_summary_knn | 0.545615 | 0.548246 | 0.200000 |
| valve | density_gaussian_shrinkage | 0.388179 | 0.484737 | 0.206667 |
| valve | density_gmm_diag_k4 | 0.479094 | 0.488070 | 0.280000 |

## 6. Bottleneck-Machine Tail Table

| machine | score_mode | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | latent_summary_knn | 0.1168 | 0.1282 | 0.0929 | 0.1030 | 0.1067 | 0.0700 |
| gearbox | density_gaussian_shrinkage | -351.7227 | -327.5618 | -387.3324 | -377.8499 | 0.0733 | 0.0500 |
| gearbox | density_gmm_diag_k4 | -249.0001 | -173.0007 | -389.0206 | -343.7242 | 0.0933 | 0.0267 |
| slider | latent_summary_knn | 0.1267 | 0.1494 | 0.1085 | 0.1187 | 0.1267 | 0.0167 |
| slider | density_gaussian_shrinkage | -309.2050 | -276.4673 | -337.6087 | -327.4053 | 0.0667 | 0.0133 |
| slider | density_gmm_diag_k4 | -204.3734 | -152.8239 | -268.5380 | -246.4706 | 0.0667 | 0.0167 |
| toycar | latent_summary_knn | 0.0827 | 0.0870 | 0.0744 | 0.0781 | 0.0933 | 0.0300 |
| toycar | density_gaussian_shrinkage | -485.9382 | -477.9394 | -499.5649 | -493.1096 | 0.1133 | 0.0167 |
| toycar | density_gmm_diag_k4 | -414.5370 | -389.6063 | -497.8906 | -472.6926 | 0.0233 | 0.0000 |
| toytrain | latent_summary_knn | 0.0928 | 0.1193 | 0.0687 | 0.0757 | 0.0933 | 0.0333 |
| toytrain | density_gaussian_shrinkage | -419.1702 | -355.9904 | -458.8076 | -451.8958 | 0.0333 | 0.0000 |
| toytrain | density_gmm_diag_k4 | -255.4372 | 110.9686 | -436.0291 | -393.1248 | 0.0667 | 0.0000 |
| fan | latent_summary_knn | 0.2295 | 0.2382 | 0.1711 | 0.2209 | 0.1900 | 0.1267 |
| fan | density_gaussian_shrinkage | 481.7031 | 557.7305 | -135.0980 | 461.9696 | 0.2400 | 0.1533 |
| fan | density_gmm_diag_k4 | 1378.5325 | 1500.8720 | 164.6521 | 1110.2913 | 0.1200 | 0.0633 |

## 7. Final Verdict

Success criteria check:

- `density_gaussian_shrinkage` matched-baseline gate: `1/3` seeds
- `density_gaussian_shrinkage` seed-panel mean: hmean=0.537180, pAUC=0.532406, KS=0.308567
- `density_gaussian_shrinkage` bottleneck TPR@99n improvements: `1/5`
- `density_gmm_diag_k4` matched-baseline gate: `0/3` seeds
- `density_gmm_diag_k4` seed-panel mean: hmean=0.531235, pAUC=0.521178, KS=0.353367
- `density_gmm_diag_k4` bottleneck TPR@99n improvements: `0/5`

Density scoring does not solve the low-FPR bottleneck if it fails to improve bottleneck-machine `TPR@99n`, even when overall hmean/pAUC remain competitive.

```text
If both density scores fail the low-FPR gate, record V38 as a scoring-rule audit failure.
```
