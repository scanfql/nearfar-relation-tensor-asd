# CTAF V31 Residual Temporal Pyramid Audit

## Objective

Test whether a residual temporal-pyramid correction inside the V16 family can
keep the `V30` hmean gain while recovering the `V16`-level `pAUC`.

## Structure

`CTAF-v2B = viewwise_unshared_residual_temporal_pyramid`

Design:

- keep the original `V16` unshared per-view stem as the main path
- add a temporal pyramid residual branch after each per-view stem output
- use a global learnable scalar `alpha`, initialized to `0.1`
- form the corrected per-view feature as:

```text
h = original_view_stem_output
p = temporal_pyramid(h)
h_out = h + alpha * p
```

- keep late fusion, latent summary vectors, and train-normal latent kNN score
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

V30 seed `13711` reference:

```text
hmean = 0.562034
pAUC  = 0.548045
KS    = 0.2943
```

If V31 exceeds both `hmean` and `pAUC` of `V30 seed 13711`, it is recorded as
`strong positive`, but it still cannot become a new mainline on a single seed.

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
  --model_type viewwise_unshared_residual_temporal_pyramid \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/v31_residual_temporal_pyramid_seed13711 \
  --score_modes latent_summary_knn \
  --latent_knn_k 5 \
  --no-progress
```

## Stage 1 Result: Seed 13711

Overall comparison:

| experiment | hmean | pAUC | KS |
|---|---:|---:|---:|
| logmel_only | 0.542030 | 0.532632 | 0.3829 |
| full_5view | 0.555806 | 0.541654 | 0.2714 |

Comparison against active baselines:

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V30 seed 13711:
hmean = 0.562034
pAUC  = 0.548045
KS    = 0.2943

V31 seed 13711:
hmean = 0.555806
pAUC  = 0.541654
KS    = 0.2714
```

Delta against matched `logmel_only`:

```text
hmean delta = +0.013776
pAUC  delta = +0.009022
KS    delta = -0.1115
```

Delta against `V16` seed-panel mean:

```text
hmean delta = +0.006259
pAUC  delta = -0.000677
KS    delta = -0.023480
```

Delta against `V30 seed 13711`:

```text
hmean delta = -0.006228
pAUC  delta = -0.006391
KS    delta = -0.0229
```

## Single-Seed Gate Check

- must beat matched `logmel_only`: `PASS`
- `hmean > 0.549547`: `PASS`
- `pAUC >= 0.542331`: `FAIL`, short by `0.000677`
- `KS not clearly worse than 0.294880`: `PASS`, clearly better
- strong positive versus `V30 seed 13711`: `FAIL`

## Verdict

```text
V31 does not pass the single-seed gate.
```

Therefore:

```text
Do not continue to the 3-seed panel.
Do not continue to V32.
```

Failure analysis:

- the residual temporal-pyramid design succeeds at the intended structural
  correction goal better than plain `V30` on score geometry: `KS = 0.2714`,
  which is better than both `V16` seed-panel mean and `V30 seed 13711`
- it also keeps a real discrimination gain over matched `logmel_only`
- however, the residual correction does not recover enough low-FPR separation:
  `pAUC = 0.541654`, which is still below the active V16 seed-panel threshold
  `0.542331`
- the failure margin is very small, so this is not a collapse result
- but under the updated seed-panel evidence rule, near-miss is still fail

Per-machine signal summary:

- strongest gains are concentrated in `bearing`, `toycar`, and improved score
  geometry for `gearbox`, `slider`, and `fan`
- `gearbox` remains a major weak point on `pAUC/hmean`
- `valve` and `toytrain` do not preserve the `V30` high-end discrimination
- the net effect is: better stability geometry, insufficient `pAUC`
