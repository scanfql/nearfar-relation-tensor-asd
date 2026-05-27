# CTAF V33 Tail Compact Audit

## Objective

Test a training-side tail-sensitive regularizer that explicitly compresses the
right tail of train-normal latent summary distances, instead of continuing to
do score-side tail postprocessing.

## Fixed Structure

`V33` keeps the frozen `V16` architecture:

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
loss_view_mode = all_views
seed = 13711
```

## Tail-Compact Training Loss

Training objective:

```text
loss = reconstruction_loss + lambda_tail * normal_tail_compactness_loss
```

Fixed rule:

1. extract latent summary `z` from each normal batch sample
2. compute batch center
3. compute each sample's L2 distance to the center
4. take the top `10%` distances
5. tail compactness loss = mean of that top tail
6. `lambda_tail = 0.01`

No sweep, no per-machine setting, no anomaly labels.

## Baselines

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

`V32` summary:

- local-tail score-side postprocessing failed
- it inflated the normal extreme tail
- it did not improve bottleneck-machine `TPR@99n`

So `V33` tests whether tail control should happen during representation
learning rather than after scoring.

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
  --loss_view_mode all_views \
  --model_type viewwise_unshared_late_fusion \
  --tail_compact_loss \
  --tail_compact_lambda 0.01 \
  --tail_compact_top_frac 0.10 \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v33_tail_compact_seed13711 \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Pending Audit

The completed audit will report:

1. V16 seed-panel mean baseline
2. V32 local-tail failure summary
3. V33 matched logmel vs full_5view
4. overall `hmean / pAUC / KS`
5. per-machine `pAUC`
6. normal `q95 / q99`
7. anomaly median / anomaly `q75`
8. `TPR@95n / TPR@99n`
9. whether normal `q99` is reduced
10. whether at least two bottleneck machines improve `TPR@99n`

## Stage 1 Result: Seed 13711

## 1. V16 Seed-Panel Mean Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. V32 Local-Tail Failure Summary

`V32` failed because score-side local-tail postprocessing:

- did not recover low-FPR `pAUC`
- inflated the normal extreme tail
- failed to improve bottleneck-machine `TPR@99n`

So `V33` moved the tail-sensitive intervention into training.

## 3. V33 Matched logmel vs full_5view

Overall result:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.530561 | 0.522632 | 0.3714 |
| full_5view | latent_summary_knn | 0.531218 | 0.518195 | 0.2800 |

Matched-baseline comparison:

```text
full_5view vs matched logmel_only:
hmean delta = +0.000657
pAUC  delta = -0.004437
KS    delta = -0.0914
```

So `V33` barely improves hmean over matched `logmel_only`, but loses pAUC.

## 4. Gate Check

Target result:

```text
V33 full_5view / latent_summary_knn:
hmean = 0.531218
pAUC  = 0.518195
KS    = 0.2800
```

Single-seed gate evaluation:

- full_5view must beat matched logmel_only: `FAIL`
  - pAUC is lower than matched `logmel_only`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least 2 bottleneck machines improve `TPR@99n`: `FAIL`, `0/5`
- normal `q99` cannot rise overall: `PASS`

This is a clear single-seed failure.

## 5. Per-Machine pAUC

`full_5view / latent_summary_knn`:

| machine | V33 pAUC | V33 hmean |
|---|---:|---:|
| bearing | 0.556316 | 0.584732 |
| fan | 0.556842 | 0.560764 |
| gearbox | 0.509474 | 0.513346 |
| slider | 0.510526 | 0.577687 |
| toycar | 0.486316 | 0.473515 |
| toytrain | 0.486316 | 0.517350 |
| valve | 0.521579 | 0.491128 |

Main drags are:

- `toycar`
- `toytrain`
- `gearbox`
- `valve`
- `slider` also remains too weak on pAUC despite decent hmean

## 6. Bottleneck Tail Table

Compared with `V16`, for bottleneck machines:

| machine | model | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | V16 | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| gearbox | V33 | 0.0899 | 0.1068 | 0.0800 | 0.0844 | 0.08 | 0.01 |
| slider | V16 | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| slider | V33 | 0.0938 | 0.0985 | 0.0829 | 0.0876 | 0.05 | 0.01 |
| toycar | V16 | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| toycar | V33 | 0.0470 | 0.0491 | 0.0413 | 0.0434 | 0.03 | 0.01 |
| toytrain | V16 | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| toytrain | V33 | 0.0713 | 0.0763 | 0.0580 | 0.0607 | 0.00 | 0.00 |
| fan | V16 | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |
| fan | V33 | 0.1289 | 0.1324 | 0.0964 | 0.1233 | 0.19 | 0.10 |

## 7. Did V33 Lower Normal q99?

Yes.

On every listed bottleneck machine, `V33` reduces normal `q99`:

- `gearbox`: `0.1245 -> 0.1068`
- `slider`: `0.1319 -> 0.0985`
- `toycar`: `0.0757 -> 0.0491`
- `toytrain`: `0.1122 -> 0.0763`
- `fan`: `0.2367 -> 0.1324`

So the training-side compactness loss does what it is designed to do on normal
tail compression.

## 8. Did V33 Improve Bottleneck TPR@99n?

No. It makes it worse on every audited bottleneck machine:

- `gearbox`: `0.06 -> 0.01`
- `slider`: `0.09 -> 0.01`
- `toycar`: `0.10 -> 0.01`
- `toytrain`: `0.05 -> 0.00`
- `fan`: `0.13 -> 0.10`

This is the decisive failure.

## 9. Failure Analysis

`V33` shows that:

```text
compressing the train-normal latent-summary tail is not enough.
```

More precisely:

- the compactness loss successfully shrinks the normal right tail
- but it also shrinks anomaly scores toward the center
- anomaly median and anomaly q75 both move downward with the normal tail
- because anomaly tail contraction is as strong as, or stronger than, normal
  tail contraction, low-FPR separation does not improve

This means the current tail-compactness regularizer is:

```text
tail-compressing, but not tail-separating
```

That is why `KS` improves and normal `q99` falls, yet `pAUC` still collapses.

## Verdict

```text
V33 does not pass the single-seed gate.
```

Therefore:

```text
Do not continue to V34.
Do not sweep lambda.
Record this as a training-side tail-compactness failure analysis.
```
