# CTAF V36 Teacher-Student Local Distillation Audit

## Objective

Learn normal local predictability directly, then use teacher-student local
discrepancy tail as the anomaly signal.

## Fixed Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## Failure Context

- `V32`: local-tail score on existing latent tokens raised the normal tail
- `V33`: tail compactness compressed both normal and anomaly tails
- `V34`: cross-view SSL improved geometry, not bottleneck `TPR@99n`
- `V35`: temporal predictive SSL improved geometry, not bottleneck `TPR@99n`

So `V36` does not reuse ordinary AE reconstruction or score-side tail tricks.

## Two-Stage Training

### Stage 1: Teacher

- architecture: `viewwise_unshared_late_fusion`
- input: clean `full_5view`
- target: clean `full_5view`
- loss: standard all-view reconstruction
- train on source normal only
- freeze teacher after training

### Stage 2: Student

- architecture: same as teacher
- teacher input: clean `full_5view`
- student input: fixed-rule TF block corrupted `full_5view`
- target: teacher final local latent map
- loss: local normalized feature discrepancy

Main score:

```text
teacher_student_top5
```

Rule:

- teacher sees clean tensor
- student sees fixed corrupted tensor
- local discrepancy map is computed from normalized local latent vectors
- file score = mean(top `5%` local discrepancies)
- larger discrepancy = more anomalous

Diagnostic score:

- `latent_summary_knn`

No score fusion is allowed.

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
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v36_teacher_student_local_distill_seed13711 \
  --score_modes latent_summary_knn,teacher_student_top5 \
  --latent_knn_k 5 \
  --no-progress
```

## Pending Audit

The completed audit will report:

1. V16 seed-panel mean baseline
2. V32/V33/V34/V35 failure summary
3. matched `logmel_only` vs `full_5view`
4. score comparison:
   - `latent_summary_knn` diagnostic
   - `teacher_student_top5` main score
5. overall `hmean / pAUC / KS`
6. per-machine `hmean / pAUC / KS`
7. bottleneck-machine tail table:
   - normal `q95`
   - normal `q99`
   - anomaly median
   - anomaly `q75`
   - `TPR@95n`
   - `TPR@99n`
8. whether bottleneck-machine `TPR@99n` improves

## 1. V16 Seed-Panel Mean Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. V36 Overall Comparison

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.534091 | 0.520150 | 0.4057 |
| full_5view | latent_summary_knn | 0.563084 | 0.550451 | 0.3143 |
| logmel_only | teacher_student_top5 | 0.534008 | 0.503985 | 0.3171 |
| full_5view | teacher_student_top5 | 0.507969 | 0.515564 | 0.2743 |

Key contrast:

- `teacher_student_top5` is the main score and fails
- `latent_summary_knn` is only diagnostic, but it is clearly positive

## 3. Main Decision

```text
teacher_student_top5 failed the single-seed gate.
```

Reason:

- matched `logmel_only / teacher_student_top5`:
  `hmean=0.534008`, `pAUC=0.503985`
- `full_5view / teacher_student_top5`:
  `hmean=0.507969`, `pAUC=0.515564`

So:

- `pAUC` improves over matched `logmel_only`
- but `hmean` drops below matched `logmel_only`
- and the score remains below the `V16` seed-panel baseline on both `hmean`
  and `pAUC`

## 4. Diagnostic Finding

```text
latent_summary_knn is positive, suggesting representation benefit but not
discrepancy-score success.
```

The diagnostic score:

```text
full_5view / latent_summary_knn:
hmean = 0.563084
pAUC  = 0.550451
KS    = 0.3143
```

This exceeds the active `V16` seed-panel mean on `hmean` and `pAUC`, but its
`KS` is worse than `0.294880`. Since `latent_summary_knn` is only diagnostic in
`V36`, this cannot promote the method. It only indicates that teacher-student
training may have helped the representation learned by the encoder.

## 5. Per-Machine Table

`full_5view` only:

| machine | latent_summary hmean | latent_summary pAUC | latent_summary KS | teacher_student hmean | teacher_student pAUC | teacher_student KS |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.648035 | 0.644211 | 0.20 | 0.477544 | 0.503158 | 0.22 |
| fan | 0.580905 | 0.568421 | 0.98 | 0.573930 | 0.551579 | 0.78 |
| gearbox | 0.522195 | 0.520000 | 0.10 | 0.533661 | 0.523684 | 0.18 |
| slider | 0.530757 | 0.520000 | 0.22 | 0.479939 | 0.505263 | 0.14 |
| toycar | 0.551497 | 0.530000 | 0.20 | 0.506414 | 0.530526 | 0.38 |
| toytrain | 0.522922 | 0.507368 | 0.22 | 0.498060 | 0.512632 | 0.14 |
| valve | 0.585275 | 0.563158 | 0.28 | 0.486235 | 0.482105 | 0.08 |

Pattern:

- `teacher_student_top5` helps `gearbox` and `toytrain` a bit on `pAUC`
- but it collapses `bearing`, `slider`, and `valve`
- `latent_summary_knn` remains much stronger overall

## 6. Bottleneck-Machine Tail Table

Below, `V16_latent_summary` is the original reference tail, `V36_teacher_student`
is the main score under evaluation, and `V36_latent_summary` is the diagnostic
score.

### gearbox

| score | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| V16_latent_summary | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| V36_teacher_student | 0.3827 | 0.3884 | 0.3615 | 0.3710 | 0.10 | 0.06 |
| V36_latent_summary | 0.1176 | 0.1296 | 0.0962 | 0.1038 | 0.10 | 0.06 |

### slider

| score | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| V16_latent_summary | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| V36_teacher_student | 0.2761 | 0.2926 | 0.2358 | 0.2515 | 0.07 | 0.01 |
| V36_latent_summary | 0.1227 | 0.1596 | 0.1073 | 0.1161 | 0.11 | 0.00 |

### toycar

| score | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| V16_latent_summary | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| V36_teacher_student | 0.1703 | 0.1722 | 0.1592 | 0.1649 | 0.11 | 0.09 |
| V36_latent_summary | 0.0779 | 0.0820 | 0.0713 | 0.0746 | 0.15 | 0.04 |

### toytrain

| score | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| V16_latent_summary | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| V36_teacher_student | 0.2096 | 0.2143 | 0.1941 | 0.2007 | 0.08 | 0.02 |
| V36_latent_summary | 0.0951 | 0.1220 | 0.0707 | 0.0750 | 0.07 | 0.03 |

### fan

| score | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| V16_latent_summary | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |
| V36_teacher_student | 0.5374 | 0.5705 | 0.3349 | 0.5145 | 0.16 | 0.08 |
| V36_latent_summary | 0.2291 | 0.2385 | 0.1710 | 0.2235 | 0.21 | 0.12 |

## 7. Failure Analysis

### Does `teacher_student_top5` improve any bottleneck-machine `TPR@99n`?

No.

Compared with `V16_latent_summary`:

- `gearbox`: `0.06 -> 0.06`
- `slider`: `0.09 -> 0.01`
- `toycar`: `0.10 -> 0.09`
- `toytrain`: `0.05 -> 0.02`
- `fan`: `0.13 -> 0.08`

So `0/5` bottleneck machines improve.

### Does `latent_summary_knn` improve pAUC but worsen KS?

Yes, at the overall level.

Compared with the active `V16` seed-panel mean:

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V36 diagnostic latent_summary_knn:
hmean = 0.563084
pAUC  = 0.550451
KS    = 0.3143
```

Interpretation:

- the representation learned under teacher-student training may be better for
  the old latent-summary score
- but the geometry is worse than the active `V16` seed-panel baseline
- this is evidence of representation benefit, not evidence that the new main
  score works

### Is the local discrepancy map mainly capturing normal variability?

Yes, that is the strongest reading of the result.

Evidence:

- `teacher_student_top5` raises normal `q95/q99` massively on bottleneck
  machines
- anomaly scores also rise, but not enough to move far beyond that inflated
  normal tail
- `KS` often improves or stays acceptable, so the problem is not simple domain
  mismatch
- the decisive failure is weak or worsening `TPR@99n`

So the discrepancy map is responding strongly to difficult normal local
variation, not selectively to anomaly-only local unpredictability.

## 8. Final Verdict

```text
V36 teacher_student_top5 should not replace V16 and should not enter seed panel.
```

However:

```text
teacher-student training as representation pretraining may be considered as a
separate future diagnostic, but only under a new pre-registered rule.
```
