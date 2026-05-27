# CTAF V51 Learned One-Class Boundary Audit

## 1. V51 prereg summary

- V51 is a new Stage-8 hypothesis
- it is not a continuation of V39-V50 parameter search
- encoder / frontend / pooling remain frozen
- only the low-FPR boundary scoring is replaced

## 2. OC-SVM fixed protocol

- model: One-Class SVM
- kernel: `RBF`
- `nu = 0.05`
- `gamma = "scale"`
- feature: frozen V37 full_5view latent_summary (192-dim)
- preprocessing: train-normal feature-wise z-score
- score = `-decision_function(x)`
- one model per `seed / machine`
- no sweep
- no anomaly label

## 3. Overall comparison

| score_mode | mean hmean | mean pAUC | mean KS | std hmean | std pAUC | std KS |
|---|---:|---:|---:|---:|---:|---:|
| V37 latent_summary_knn | 0.553898 | 0.545689 | 0.305733 | 0.008006 | 0.004443 | 0.010304 |
| V51 ocsvm_rbf | 0.533110 | 0.526341 | 0.340000 | 0.009475 | 0.002436 | 0.010302 |

## 4. Seed-level table

| seed | hmean | pAUC | KS |
|---|---:|---:|---:|
| 13711 | 0.527467 | 0.523534 | 0.331429 |
| 13721 | 0.544048 | 0.527594 | 0.351429 |
| 13731 | 0.527813 | 0.527895 | 0.337143 |

## 5. Per-machine table

| machine | mean hmean | mean pAUC | mean KS |
|---|---:|---:|---:|
| bearing | 0.612690 | 0.617018 | 0.140000 |
| fan | 0.527498 | 0.569123 | 1.000000 |
| gearbox | 0.449731 | 0.511053 | 0.280000 |
| slider | 0.524175 | 0.502281 | 0.120000 |
| toycar | 0.547714 | 0.492632 | 0.386667 |
| toytrain | 0.515013 | 0.497018 | 0.293333 |
| valve | 0.554945 | 0.495263 | 0.160000 |

## 6. Tail table

| machine | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.4234 | 0.7450 | -0.1741 | 0.5633 | 0.2800 | 0.2100 |
| fan | 4.1942 | 4.1953 | 3.4121 | 4.1932 | 0.2100 | 0.1500 |
| gearbox | -0.0217 | 0.8691 | -0.5222 | -0.3467 | 0.0800 | 0.0400 |
| slider | 0.1742 | 0.5966 | -0.6186 | -0.3110 | 0.0800 | 0.0200 |
| toycar | 1.2142 | 1.4334 | -0.3169 | 0.1700 | 0.0100 | 0.0100 |
| toytrain | 1.1654 | 3.3865 | -0.5878 | -0.0684 | 0.0400 | 0.0000 |
| valve | 0.4612 | 0.8263 | -0.4283 | -0.0406 | 0.0200 | 0.0100 |

## 7. q99 inflation check

- machines with >5% q99 inflation relative to V37 baseline: `bearing, fan, gearbox, slider, toycar, toytrain, valve`

## 8. OC-SVM Boundary vs V48 Hard-Normal Tail

| machine | n_train_outliers | outlier_rate | hard_normal_count | overlap_count | outlier_hard_normal_overlap | hard_normal_recall |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 62 | 0.0626 | 99 | 32 | 0.5161 | 0.3232 |
| fan | 64 | 0.0646 | 99 | 19 | 0.2969 | 0.1919 |
| gearbox | 65 | 0.0657 | 99 | 42 | 0.6462 | 0.4242 |
| slider | 72 | 0.0727 | 99 | 19 | 0.2639 | 0.1919 |
| toycar | 66 | 0.0667 | 99 | 29 | 0.4394 | 0.2929 |
| toytrain | 76 | 0.0768 | 99 | 29 | 0.3816 | 0.2929 |
| valve | 62 | 0.0626 | 99 | 5 | 0.0806 | 0.0505 |
| bearing | 65 | 0.0657 | 99 | 30 | 0.4615 | 0.3030 |
| fan | 62 | 0.0626 | 99 | 18 | 0.2903 | 0.1818 |
| gearbox | 63 | 0.0636 | 99 | 33 | 0.5238 | 0.3333 |
| slider | 58 | 0.0586 | 99 | 6 | 0.1034 | 0.0606 |
| toycar | 65 | 0.0657 | 99 | 32 | 0.4923 | 0.3232 |
| toytrain | 59 | 0.0596 | 99 | 20 | 0.3390 | 0.2020 |
| valve | 59 | 0.0596 | 99 | 5 | 0.0847 | 0.0505 |
| bearing | 72 | 0.0727 | 99 | 34 | 0.4722 | 0.3434 |
| fan | 62 | 0.0626 | 99 | 25 | 0.4032 | 0.2525 |
| gearbox | 63 | 0.0636 | 99 | 31 | 0.4921 | 0.3131 |
| slider | 65 | 0.0657 | 99 | 12 | 0.1846 | 0.1212 |
| toycar | 70 | 0.0707 | 99 | 26 | 0.3714 | 0.2626 |
| toytrain | 65 | 0.0657 | 99 | 32 | 0.4923 | 0.3232 |
| valve | 58 | 0.0586 | 99 | 4 | 0.0690 | 0.0404 |

Diagnostic interpretation rule:

- high overlap suggests OC-SVM and V48 hard-normal tail are aligned
- low overlap suggests OC-SVM is learning a different boundary structure
- if train outlier rate deviates strongly from `nu=0.05`, the learned boundary behavior is not matching the nominal expectation exactly

## 9. Final verdict

`V51 negative`

Success gates:

- mean hmean >= V37 - 0.003: `FAIL`
- mean pAUC >= V37: `FAIL`
- >=2 bottleneck machines improve TPR@99n: `2/5`
- bearing / valve do not collapse: `PASS`
- toycar does not clearly degrade: `FAIL`
- normal q99 does not clearly inflate: `FAIL`
- matched baseline majority gate: `1/3` seeds

Missing path policy:

- no missing required score files
- no missing required hard-normal indices

This audit trains only fixed OC-SVM boundaries on frozen latent summaries. No encoder/checkpoint training occurred. No protocol changes were made.
