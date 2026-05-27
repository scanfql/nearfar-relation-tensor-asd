# CTAF V37 Teacher-Student Pretrain Seed Panel

## Objective

Evaluate teacher-student local distillation as representation pretraining while
keeping `latent_summary_knn` as the only main score.

## Fixed Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## Why V37 Exists

`V36` established two facts:

- `teacher_student_top5` failed as the main score
- `latent_summary_knn` was positive as a diagnostic score

So `V37` asks a narrower question:

```text
Does teacher-student local distillation improve the encoder representation
enough that latent_summary_knn becomes a better main score?
```

## Fixed Configuration

```text
model_type = viewwise_unshared_late_fusion
ssl_task = teacher_student_local_distillation
experiments = logmel_only, full_5view
latent_channels = 64
stem_channels = 8
per_view_channels = 16
dropout_p = 0.0
score_modes = latent_summary_knn
latent_knn_k = 5
training = source normal only
memory bank = source normal only
```

Seeds:

- `13711`
- `13721`
- `13731`

No corruption-rule tuning, no `teacher_student_top5` fusion, no k tuning, no
dropout tuning, no target data, and no automatic `V38`.

## Stage 1 Command

```bash
python -m src.run_experiment \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --experiments logmel_only,full_5view \
  --epochs 50 \
  --batch_size 32 \
  --latent_channels 64 \
  --stem_channels 8 \
  --per_view_channels 16 \
  --dropout_p 0.0 \
  --ssl_task teacher_student_local_distillation \
  --model_type viewwise_unshared_late_fusion \
  --seed <SEED> \
  --device cuda:0 \
  --output_dir results/v37_teacher_student_pretrain_seedpanel/seed_<SEED> \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Pending Audit

The completed report will include:

1. `V16` seed-panel mean baseline
2. `V36` failure summary
3. seed-panel table:
   `seed | logmel hmean | full_5view hmean | delta | logmel pAUC | full_5view pAUC | delta | full_5view KS`
4. aggregate mean/std:
   `hmean`, `pAUC`, `KS`
5. per-machine `hmean / pAUC / KS`
6. bottleneck-machine tail table:
   `normal q95, normal q99, anomaly median, anomaly q75, TPR@95n, TPR@99n`
7. final verdict

## 1. V16 Seed-Panel Mean Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. V36 Failure Summary

`V36` established:

- `teacher_student_top5` failed as the main score
- `latent_summary_knn` was positive as a diagnostic score

So `V37` keeps only:

```text
teacher-student local distillation as representation pretraining
```

while restoring:

```text
latent_summary_knn
```

as the only main score.

## 3. Seed-Panel Table

| seed | logmel hmean | full_5view hmean | delta | logmel pAUC | full_5view pAUC | delta | full_5view KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13711 | 0.534091 | 0.563084 | +0.028993 | 0.520150 | 0.550451 | +0.030301 | 0.3143 |
| 13721 | 0.532118 | 0.550205 | +0.018087 | 0.518571 | 0.544962 | +0.026391 | 0.2943 |
| 13731 | 0.536815 | 0.548404 | +0.011589 | 0.526992 | 0.541654 | +0.014662 | 0.3086 |

Matched-baseline gate:

```text
3/3 seeds beat matched logmel_only on both hmean and pAUC.
```

## 4. Aggregate Mean / Std

`full_5view / latent_summary_knn`:

```text
mean_hmean = 0.553898
std_hmean  = 0.008006

mean_pAUC  = 0.545689
std_pAUC   = 0.004443

mean_KS    = 0.305733
std_KS     = 0.010304
```

Compared with `V16 seed-panel mean`:

```text
hmean delta = +0.004351
pAUC  delta = +0.003358
KS    delta = +0.010853
```

Interpretation:

- `hmean` and `pAUC` both clear the `V16 seed-panel mean`
- `KS` is clearly worse than the active baseline

## 5. Per-Machine hmean / pAUC / KS

`full_5view / latent_summary_knn`, averaged across `13711/13721/13731`:

| machine | mean hmean | std | mean pAUC | std | mean KS | std |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.640740 | 0.022225 | 0.625614 | 0.020797 | 0.166667 | 0.041633 |
| fan | 0.578161 | 0.003398 | 0.565965 | 0.019066 | 0.993333 | 0.011547 |
| gearbox | 0.540506 | 0.024012 | 0.523333 | 0.004476 | 0.180000 | 0.080000 |
| slider | 0.512138 | 0.030021 | 0.526667 | 0.006145 | 0.186667 | 0.057735 |
| toycar | 0.541548 | 0.023503 | 0.515088 | 0.015001 | 0.213333 | 0.041633 |
| toytrain | 0.518576 | 0.006673 | 0.514912 | 0.011292 | 0.200000 | 0.034641 |
| valve | 0.545615 | 0.035559 | 0.548246 | 0.014047 | 0.200000 | 0.069282 |

Pattern:

- `bearing` is a strong positive
- `fan` has good `hmean/pAUC`, but its `KS` remains extremely bad
- `slider`, `toycar`, and `toytrain` remain low-FPR bottlenecks

## 6. Bottleneck-Machine Tail Table

Tail comparison uses:

- `V16` reference
- `V37 seed 13711` as the concrete local-tail diagnostic snapshot

| machine | model | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | V16 | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| gearbox | V37 | 0.1176 | 0.1296 | 0.0962 | 0.1038 | 0.10 | 0.06 |
| slider | V16 | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| slider | V37 | 0.1227 | 0.1596 | 0.1073 | 0.1161 | 0.11 | 0.00 |
| toycar | V16 | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| toycar | V37 | 0.0779 | 0.0820 | 0.0713 | 0.0746 | 0.15 | 0.04 |
| toytrain | V16 | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| toytrain | V37 | 0.0951 | 0.1220 | 0.0707 | 0.0750 | 0.07 | 0.03 |
| fan | V16 | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |
| fan | V37 | 0.2291 | 0.2385 | 0.1710 | 0.2235 | 0.21 | 0.12 |

Low-FPR gate:

```text
At least 2 bottleneck machines must improve TPR@99n.
Observed: 0/5 improve.
```

## 7. Final Verdict

Evaluation against the pre-registered rules:

- matched baseline gate: `PASS`
- seed-panel mean hmean gate: `PASS`
- seed-panel mean pAUC gate: `PASS`
- KS gate: `FAIL`, `0.305733 > 0.294880`
- low-FPR gate: `FAIL`, `0/5` bottleneck machines improve `TPR@99n`

Final verdict:

```text
V37 improves representation enough to lift seed-panel hmean and pAUC above the
V16 baseline, but it does not improve the low-FPR bottleneck and its KS is
clearly worse.
```

Therefore:

```text
V37 is a representation diagnostic, not a replacement for V16.
```

Operational conclusion:

- do not replace `V16`
- do not treat `V37` as a new mainline
- if teacher-student pretraining is explored again, it should be under a new
  pre-registered rule that explicitly addresses the low-FPR tail bottleneck,
  because current evidence says representation quality alone is not enough
