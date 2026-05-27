# CTAF V30 Temporal Pyramid Audit

## Objective

Test whether internal multi-scale temporal modeling inside the frozen V16 family
can beat the current V16 seed-panel evidence baseline under the clean
label-free protocol.

## Structure

`CTAF-v2A = viewwise_unshared_temporal_pyramid`

Design:

- keep the V16 per-view unshared stem
- insert a temporal pyramid block after each view stem
- use three fixed lightweight temporal branches:
  - short branch
  - medium branch
  - long branch
- fuse branch outputs back to the original per-view channel width
- keep late fusion, latent summary vectors, and train-normal kNN scoring
  unchanged

## Fixed Evaluation Baselines

Matched baseline gate:

```text
new structure must beat matched logmel_only
```

V16 seed-panel mean gate:

```text
hmean > 0.549547
pAUC  >= 0.542331
KS not clearly worse than 0.294880
```

Original frozen V16 point:

```text
hmean = 0.564294
pAUC  = 0.553459
KS    = 0.2886
```

If V30 exceeds the original frozen V16 point on this single seed, it can only
be labeled `candidate replacement`. It cannot become the new frozen mainline
without a seed-panel recheck.

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
  --model_type viewwise_unshared_temporal_pyramid \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v30_temporal_pyramid_seed13711 \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Stage 1 Result: Seed 13711

Overall comparison:

| experiment | hmean | pAUC | KS |
|---|---:|---:|---:|
| logmel_only | 0.549725 | 0.539248 | 0.4143 |
| full_5view | 0.562034 | 0.548045 | 0.2943 |

Stage-1 gate check:

- matched baseline gate: `PASS`
- V16 seed-panel mean gate: `PASS`
- original frozen V16 point: `FAIL`

Interpretation:

```text
V30 seed 13711 passes the seed-panel mean gate, but is not a candidate
replacement for frozen V16.
```

This justified the pre-registered `3-seed` follow-up panel:

- `13711`
- `13721`
- `13731`

## Stage 2 Command

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
  --model_type viewwise_unshared_temporal_pyramid \
  --seed <13711|13721|13731> \
  --device cuda:0 \
  --output_dir results/v30_temporal_pyramid_seedpanel/seed_<SEED> \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Stage 2 Per-Seed Results

| seed | logmel hmean | full_5view hmean | delta | logmel pAUC | full_5view pAUC | delta | full_5view KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13711 | 0.549725 | 0.562034 | +0.012309 | 0.539248 | 0.548045 | +0.008797 | 0.2943 |
| 13721 | 0.540117 | 0.545995 | +0.005878 | 0.534286 | 0.538947 | +0.004661 | 0.2886 |
| 13731 | 0.546489 | 0.557611 | +0.011122 | 0.529173 | 0.536917 | +0.007744 | 0.3029 |

## Stage 2 Seed-Panel Mean

`V30 temporal pyramid / full_5view / latent_summary_knn`:

```text
mean_hmean = 0.555213
mean_pAUC  = 0.541303
mean_KS    = 0.295267
```

Compared with the active V16 seed-panel baseline:

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V30 3-seed mean:
hmean = 0.555213   delta = +0.005666
pAUC  = 0.541303   delta = -0.001028
KS    = 0.295267   delta = +0.000387
```

## Verdict

Evaluation against the updated freeze rule:

- matched baseline majority gate: `PASS`, `3/3` seeds beat matched
  `logmel_only` on both hmean and pAUC
- seed-panel mean hmean gate: `PASS`
- seed-panel mean pAUC gate: `FAIL`, short by `0.001028`
- KS stability gate: `NEAR-TIE`, but slightly worse than V16 mean

Final verdict:

```text
V30 temporal pyramid is a real positive structure signal, but it does not
cleanly beat the V16 seed-panel baseline.
```

More specifically:

```text
V30 should not replace V16.
```

Failure analysis:

- the temporal pyramid consistently improves over matched `logmel_only`
- the improvement is reproducible across all `3/3` tested seeds
- the structure helps balanced discrimination enough to raise panel-mean
  `hmean`
- but the gain does not preserve enough low-FPR quality, so panel-mean `pAUC`
  remains below the V16 seed-panel baseline
- `KS` is effectively tied, so the main failure is not domain-shift collapse
- under the seed-panel evidence rule, this is a promising branch but not yet a
  mainline replacement
