# CTAF V35 Temporal Predictive SSL Audit

## Objective

Switch from reconstruction-style self-supervision to temporal prediction, so
the model must learn normal time evolution rather than just reconstructing the
observed tensor.

## Fixed Structure

`V35` keeps the `V16` architecture and scoring setup:

```text
model_type = viewwise_unshared_late_fusion
experiments = logmel_only,full_5view
latent_channels = 64
stem_channels = 8
per_view_channels = 16
dropout_p = 0.0
score_modes = latent_summary_knn
latent_knn_k = 5
training = source normal only
memory bank = source normal only
seed = 13711
```

## Self-Supervised Task

```text
ssl_task = temporal_predictive_ssl
loss_view_mode = future_only
```

Fixed rule:

- input is the full tensor with the later half of the time axis zeroed
- target is the original tensor
- loss is computed only on the later half of the time axis
- no anomaly labels are used in training, selection, or scoring

This is a fixed half-context, half-future prediction task. No horizon search is
allowed.

## Baselines

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

Failure context carried forward:

- `V30/V31`: temporal pyramid helps hmean, not low-FPR tail separation
- `V32`: local-tail score raises the normal extreme tail
- `V33`: tail compactness compresses both normal and anomaly tails
- `V34`: masked cross-view prediction improves geometry, but not bottleneck
  `TPR@99n`

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
  --ssl_task temporal_predictive_ssl \
  --loss_view_mode future_only \
  --model_type viewwise_unshared_late_fusion \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v35_temporal_predictive_ssl_seed13711 \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Pending Audit

The completed audit will report:

1. `V16` seed-panel mean baseline
2. matched `logmel_only` vs `full_5view`
3. `hmean / pAUC / KS`
4. per-machine `hmean / pAUC`
5. bottleneck-machine tail table:
   - normal `q95`
   - normal `q99`
   - anomaly median
   - anomaly `q75`
   - `TPR@95n`
   - `TPR@99n`
6. whether at least `2` bottleneck machines improve `TPR@99n`
7. whether `V16` seed-panel mean is exceeded

## Stage 1 Result: Seed 13711

## 1. V16 seed-panel mean baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. matched logmel_only vs full_5view

Overall result:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.535139 | 0.524887 | 0.3886 |
| full_5view | latent_summary_knn | 0.528608 | 0.522180 | 0.2743 |

Matched-baseline comparison:

```text
full_5view vs matched logmel_only:
hmean delta = -0.006531
pAUC  delta = -0.002707
KS    delta = -0.1143
```

So `V35` does improve geometry, but it does not beat the matched baseline.

## 3. Gate Check

Target result:

```text
V35 full_5view / latent_summary_knn:
hmean = 0.528608
pAUC  = 0.522180
KS    = 0.2743
```

Single-seed gate evaluation:

- full_5view must beat matched logmel_only: `FAIL`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least `2` bottleneck machines improve `TPR@99n`: `FAIL`, `0/5`

This is a clear single-seed failure.

## 4. Per-Machine hmean / pAUC

`full_5view / latent_summary_knn`:

| machine | hmean | pAUC |
|---|---:|---:|
| bearing | 0.635728 | 0.614737 |
| fan | 0.559551 | 0.516316 |
| gearbox | 0.554074 | 0.533158 |
| slider | 0.518983 | 0.527895 |
| toycar | 0.475531 | 0.490526 |
| toytrain | 0.464749 | 0.487368 |
| valve | 0.491638 | 0.485263 |

Pattern:

- `bearing` is strong
- `gearbox` becomes decent on aggregate metrics
- `toycar`, `toytrain`, and `valve` remain clear failures
- `fan` and `slider` do not solve the low-FPR problem

## 5. Bottleneck-Machine Tail Table

Compared against `V16`:

| machine | model | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | V16 | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| gearbox | V35 | 0.1762 | 0.1895 | 0.1303 | 0.1547 | 0.12 | 0.06 |
| slider | V16 | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| slider | V35 | 0.1674 | 0.1804 | 0.1444 | 0.1575 | 0.12 | 0.05 |
| toycar | V16 | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| toycar | V35 | 0.1102 | 0.1171 | 0.0918 | 0.0997 | 0.04 | 0.01 |
| toytrain | V16 | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| toytrain | V35 | 0.1361 | 0.1533 | 0.1035 | 0.1122 | 0.02 | 0.00 |
| fan | V16 | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |
| fan | V35 | 0.2746 | 0.2997 | 0.2072 | 0.2495 | 0.09 | 0.02 |

## 6. Did bottleneck-machine TPR@99n improve?

No.

`TPR@99n` is:

- unchanged on `gearbox`: `0.06 -> 0.06`
- worse on `slider`: `0.09 -> 0.05`
- worse on `toycar`: `0.10 -> 0.01`
- worse on `toytrain`: `0.05 -> 0.00`
- worse on `fan`: `0.13 -> 0.02`

So `0/5` bottleneck machines improve.

## 7. Did V35 exceed V16 seed-panel mean?

No.

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V35 seed 13711:
hmean = 0.528608
pAUC  = 0.522180
KS    = 0.2743
```

Interpretation:

- temporal predictive SSL learns a cleaner score geometry
- but it raises both normal and anomaly tails together
- anomaly tail does not move far enough beyond the normal extreme tail
- so low-FPR separation still does not improve

This makes `V35` another instance of:

```text
better geometry, insufficient tail separation
```

## Verdict

```text
V35 does not pass the single-seed gate.
```

Therefore:

```text
Do not continue to a V35 seed panel.
Do not open V36 from this result.
Record this as a temporal-predictive SSL failure analysis.
```
