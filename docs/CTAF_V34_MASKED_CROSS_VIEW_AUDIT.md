# CTAF V34 Masked Cross-View Audit

## Objective

Switch from standard all-view autoencoder reconstruction to masked cross-view
prediction, so the model must learn stable cross-view relations of normal
machine sound instead of only identity-style reconstruction.

## Fixed Structure

`V34` keeps the `V16` architecture and scoring setup:

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
ssl_task = masked_cross_view_prediction
loss_view_mode = masked_view_only
```

Training rule:

1. input is still `full_5view`
2. one view is masked uniformly at random from the 5 views
3. model predicts all views
4. loss is computed only on the masked view
5. no anomaly labels are used in training, mask choice, or scoring

## Baselines

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

Failure context carried forward:

- `V30/V31`: temporal pyramid gives hmean signal, but not enough low-FPR tail
  separation
- `V32`: local-tail kNN raises the normal extreme tail
- `V33`: tail compactness compresses both normal and anomaly tails

So `V34` asks whether cross-view prediction can create a better anomaly tail
without explicit tail postprocessing or extra tail loss.

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
  --ssl_task masked_cross_view_prediction \
  --loss_view_mode masked_view_only \
  --model_type viewwise_unshared_late_fusion \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v34_masked_cross_view_prediction_seed13711 \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Pending Audit

The completed audit will report:

1. V16 seed-panel mean baseline
2. V30/V31/V32/V33 failure summary
3. matched `logmel_only` vs `full_5view`
4. `hmean / pAUC / KS`
5. per-machine `hmean / pAUC`
6. bottleneck-machine tail table:
   - normal `q95`
   - normal `q99`
   - anomaly median
   - anomaly `q75`
   - `TPR@95n`
   - `TPR@99n`
7. whether bottleneck-machine `TPR@99n` improves
8. whether `V16 seed-panel mean` is exceeded

## Stage 1 Result: Seed 13711

## 1. V16 Seed-Panel Mean Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. V30/V31/V32/V33 Failure Summary

- `V30/V31`: temporal pyramid gives hmean signal, but does not solve low-FPR
  tail separation
- `V32`: local-tail score raises the normal extreme tail
- `V33`: tail compactness compresses both normal and anomaly tails

So `V34` asked whether cross-view prediction can learn a more useful anomaly
tail without explicit tail engineering.

## 3. matched logmel_only vs full_5view

Overall result:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.524520 | 0.518496 | 0.3829 |
| full_5view | latent_summary_knn | 0.539228 | 0.531955 | 0.2457 |

Matched-baseline comparison:

```text
full_5view vs matched logmel_only:
hmean delta = +0.014708
pAUC  delta = +0.013459
KS    delta = -0.1372
```

This is a real positive signal versus the matched baseline.

## 4. Overall hmean / pAUC / KS

Target result:

```text
V34 full_5view / latent_summary_knn:
hmean = 0.539228
pAUC  = 0.531955
KS    = 0.2457
```

Single-seed gate evaluation:

- full_5view beats matched logmel_only: `PASS`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`, clearly better
- at least 2 bottleneck machines improve `TPR@99n`: `FAIL`, `0/5`

So `V34` does not pass the single-seed gate.

## 5. Per-Machine hmean / pAUC

`full_5view / latent_summary_knn`:

| machine | hmean | pAUC |
|---|---:|---:|
| bearing | 0.635533 | 0.591579 |
| fan | 0.494381 | 0.490000 |
| gearbox | 0.541327 | 0.514211 |
| slider | 0.514453 | 0.507895 |
| toycar | 0.510203 | 0.498421 |
| toytrain | 0.505082 | 0.496316 |
| valve | 0.573620 | 0.625263 |

Pattern:

- `bearing` and `valve` are strong positives
- `fan`, `slider`, `toycar`, and `toytrain` remain low-FPR failures
- `gearbox` is only marginal

## 6. Bottleneck-Machine Tail Table

Compared against `V16`, bottleneck machines:

| machine | model | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | V16 | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| gearbox | V34 | 0.1495 | 0.1716 | 0.1251 | 0.1322 | 0.10 | 0.03 |
| slider | V16 | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| slider | V34 | 0.1887 | 0.1995 | 0.1484 | 0.1623 | 0.07 | 0.04 |
| toycar | V16 | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| toycar | V34 | 0.1342 | 0.1472 | 0.1146 | 0.1209 | 0.06 | 0.01 |
| toytrain | V16 | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| toytrain | V34 | 0.1405 | 0.1724 | 0.1049 | 0.1165 | 0.06 | 0.02 |
| fan | V16 | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |
| fan | V34 | 0.2438 | 0.2587 | 0.1724 | 0.2071 | 0.00 | 0.00 |

## 7. Did bottleneck-machine TPR@99n improve?

No. It becomes worse on every audited bottleneck machine:

- `gearbox`: `0.06 -> 0.03`
- `slider`: `0.09 -> 0.04`
- `toycar`: `0.10 -> 0.01`
- `toytrain`: `0.05 -> 0.02`
- `fan`: `0.13 -> 0.00`

So the cross-view SSL objective does not improve low-FPR tail occupancy.

## 8. Did V34 exceed V16 seed-panel mean?

No.

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V34 seed 13711:
hmean = 0.539228
pAUC  = 0.531955
KS    = 0.2457
```

Interpretation:

- `V34` improves matched-baseline discrimination and improves score geometry
- but it raises normal q95/q99 together with anomaly scores
- anomaly tails do not move farther right than the normal extreme tail
- so low-FPR separation still fails

## Verdict

```text
V34 does not pass the single-seed gate.
```

Therefore:

```text
Do not continue to a V34 seed panel.
Do not change the mask strategy yet.
Record this as a masked cross-view failure analysis.
```
