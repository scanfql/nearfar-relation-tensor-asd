# CTAF V32 Local Tail kNN Audit

## Objective

Test whether file-level latent summaries are smoothing away local anomaly tail
evidence that is still present in the encoder latent map.

## Low-FPR Bottleneck Context

Current confirmed bottleneck:

- `V30` and `V31` show real `hmean` signal
- but `pAUC` does not reliably clear the active `V16` seed-panel baseline
- `V31` improves `KS`, so normal score geometry is more aligned
- the remaining problem is that anomaly scores are not pushed far enough beyond
  the normal `95th–99th` percentile tail

Therefore `V32` does not change the encoder architecture. It only changes the
fixed score rule to probe local latent tail evidence.

## Fixed Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## Score Modes

- `latent_summary_knn`
- `local_tail_knn_top5`

`local_tail_knn_top5`:

1. extract encoder latent map `Z ∈ R^{C×H×W}`
2. tokenize each spatial position into a local latent token
3. build a train-normal local memory bank
4. compute token-wise kNN distance to the memory bank
5. file score = mean(top `5%` token distances)
6. larger score = more anomalous

This is a fixed global rule:

- no per-machine percentile
- no top-1/top-10 search
- no fusion with `latent_summary_knn`
- no reverse score

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
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v32_local_tail_knn_seed13711 \
  --score_modes latent_summary_knn,local_tail_knn_top5 \
  --latent_knn_k 5 \
  --no-progress
```

## Pending Audit

The completed audit will report:

1. V16 seed-panel mean baseline
2. V31 low-FPR bottleneck summary
3. `latent_summary_knn` vs `local_tail_knn_top5` comparison
4. overall `hmean / pAUC / KS`
5. per-machine `hmean / pAUC / KS`
6. bottleneck-machine tail table:
   - normal `q95`
   - normal `q99`
   - anomaly median
   - anomaly `q75`
   - `TPR@95n`
   - `TPR@99n`

Single-seed success gates:

- beats matched `logmel_only`
- `hmean > 0.549547`
- `pAUC >= 0.542331`
- `KS` not clearly worse than `0.294880`
- at least two bottleneck machines show clearly improved `TPR@99n`

## Stage 1 Result: Seed 13711

## 1. V16 Seed-Panel Mean Baseline

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## 2. V31 Low-FPR Bottleneck Summary

The confirmed bottleneck entering `V32` was:

- `V30/V31` have real `hmean` signal
- `V31` improves `KS`, so normal score geometry is better aligned
- but anomaly scores are still not pushed far enough beyond the normal
  `95th–99th` percentile tail

So `V32` was designed to test whether local latent tokens contain anomaly-tail
evidence that is being averaged away by file-level latent summaries.

## 3. latent_summary_knn vs local_tail_knn_top5

Overall comparison:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.539762 | 0.524737 | 0.4057 |
| logmel_only | local_tail_knn_top5 | 0.543065 | 0.523534 | 0.2971 |
| full_5view | latent_summary_knn | 0.561823 | 0.548647 | 0.2686 |
| full_5view | local_tail_knn_top5 | 0.505107 | 0.503383 | 0.2771 |

Interpretation:

- on `logmel_only`, `local_tail_knn_top5` slightly improves `hmean` and greatly
  improves `KS`, but does not improve `pAUC`
- on `full_5view`, `local_tail_knn_top5` collapses both `hmean` and `pAUC`
  relative to `latent_summary_knn`

## 4. Overall Gate Check

Target score mode:

```text
full_5view / local_tail_knn_top5
hmean = 0.505107
pAUC  = 0.503383
KS    = 0.2771
```

Single-seed gate evaluation:

- must beat matched `logmel_only`: `FAIL`
  - matched `logmel_only / local_tail_knn_top5`: `hmean=0.543065`,
    `pAUC=0.523534`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least 2 bottleneck machines with improved `TPR@99n`: `FAIL`, `0/5`

So `V32` fails clearly on the first-stage gate and must not enter a seed panel.

## 5. Per-Machine hmean / pAUC / KS

`full_5view` only:

| machine | latent_summary hmean | local_tail hmean | latent_summary pAUC | local_tail pAUC | latent_summary KS | local_tail KS |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.668910 | 0.478472 | 0.625789 | 0.495263 | 0.14 | 0.22 |
| fan | 0.591640 | 0.485762 | 0.546842 | 0.497895 | 0.98 | 0.74 |
| gearbox | 0.477025 | 0.517644 | 0.515789 | 0.517368 | 0.10 | 0.12 |
| slider | 0.530142 | 0.507682 | 0.534211 | 0.498421 | 0.12 | 0.12 |
| toycar | 0.568182 | 0.483464 | 0.527368 | 0.492105 | 0.16 | 0.36 |
| toytrain | 0.504216 | 0.555399 | 0.505263 | 0.513684 | 0.28 | 0.18 |
| valve | 0.592646 | 0.507328 | 0.585263 | 0.508947 | 0.10 | 0.20 |

Pattern:

- only `gearbox` and `toytrain` get meaningful `hmean/pAUC` relief
- `bearing`, `fan`, `slider`, `toycar`, and `valve` all lose substantial
  low-FPR quality
- the largest collapses are `bearing`, `toycar`, and `valve`

## 6. Bottleneck-Machine Tail Table

The current bottleneck machines are:

- `gearbox`
- `slider`
- `toycar`
- `toytrain`
- `fan`

Tail table for `full_5view`:

| machine | score_mode | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | latent_summary_knn | 0.1269 | 0.1351 | 0.0938 | 0.1033 | 0.08 | 0.06 |
| gearbox | local_tail_knn_top5 | 0.4215 | 0.4316 | 0.4142 | 0.4171 | 0.10 | 0.01 |
| slider | latent_summary_knn | 0.1275 | 0.1503 | 0.1099 | 0.1190 | 0.14 | 0.03 |
| slider | local_tail_knn_top5 | 0.4547 | 0.4759 | 0.4416 | 0.4459 | 0.04 | 0.01 |
| toycar | latent_summary_knn | 0.0734 | 0.0796 | 0.0662 | 0.0697 | 0.12 | 0.04 |
| toycar | local_tail_knn_top5 | 0.3922 | 0.3946 | 0.3852 | 0.3886 | 0.05 | 0.01 |
| toytrain | latent_summary_knn | 0.0909 | 0.1139 | 0.0693 | 0.0742 | 0.07 | 0.04 |
| toytrain | local_tail_knn_top5 | 0.4043 | 0.4111 | 0.3954 | 0.3992 | 0.11 | 0.01 |
| fan | latent_summary_knn | 0.2036 | 0.2198 | 0.1535 | 0.1967 | 0.19 | 0.03 |
| fan | local_tail_knn_top5 | 0.4564 | 0.4591 | 0.4346 | 0.4441 | 0.05 | 0.01 |

Critical observation:

- `local_tail_knn_top5` does **not** push anomaly tails away from the normal
  extreme tail
- instead, it inflates the normal `q95/q99` thresholds dramatically
- anomaly medians rise too, but not enough
- as a result, `TPR@99n` gets worse on every bottleneck machine in this audit

This is the opposite of the intended effect.

## 7. Failure Analysis

`V32` answers the original question clearly:

```text
The current encoder latent map does not contain a directly exploitable local
tail signal under this fixed top-5% local-token kNN rule.
```

What the local-tail score is actually doing:

- it is highly sensitive to local normal variability
- it elevates the right tail of normal scores
- it does not elevate anomaly local tokens enough beyond that raised normal tail

Why this hurts low-FPR:

- `pAUC` depends on what happens near the extreme right end of the normal score
  distribution
- `local_tail_knn_top5` raises the normal extreme tail much faster than it
  improves anomaly occupancy beyond that tail
- so low-FPR discrimination collapses even when some average-rank signal
  remains

The two machines with some benefit:

- `gearbox`
- `toytrain`

are not enough to offset large regressions on:

- `bearing`
- `fan`
- `slider`
- `toycar`
- `valve`

## Verdict

```text
V32 does not pass the single-seed gate.
```

Therefore:

```text
Do not continue to a V32 seed panel.
Do not tune the top percentile.
Write this down as a fixed-rule failure analysis.
```
