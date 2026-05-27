# CTAF V29 Multi-Resolution Audit

## 1. V16 Frozen Baseline

```text
CTAF V16 =
full_5view tensor frontend
+ viewwise_unshared_late_fusion Conv2D AE
+ latent summary vectors(mean/std/max)
+ train-normal latent kNN memory score
+ k=5
+ source-only normal training
```

Frozen metrics:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

## 2. V29 Command

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view,multires_7view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v29_multires_7view_unshared_late_fusion --score_modes latent_summary_knn --latent_knn_k 5
```

## 3. Overall Comparison

| experiment | hmean | pAUC | KS |
|---|---:|---:|---:|
| logmel_only | 0.543587 | 0.524586 | 0.4029 |
| full_5view | 0.553511 | 0.547820 | 0.3286 |
| multires_7view | 0.556256 | 0.553233 | 0.3286 |

## 4. Per-Machine hmean / pAUC / KS

| machine | logmel_only hmean | full_5view hmean | multires_7view hmean | logmel_only pAUC | full_5view pAUC | multires_7view pAUC | logmel_only KS | full_5view KS | multires_7view KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bearing | 0.534217 | 0.638861 | 0.639499 | 0.524737 | 0.594737 | 0.632632 | 0.2800 | 0.2800 | 0.1600 |
| fan | 0.602736 | 0.578160 | 0.552164 | 0.581053 | 0.580526 | 0.554737 | 1.0000 | 1.0000 | 1.0000 |
| gearbox | 0.556221 | 0.552767 | 0.536318 | 0.514737 | 0.545789 | 0.532105 | 0.4200 | 0.2000 | 0.2200 |
| slider | 0.520811 | 0.545818 | 0.549576 | 0.500526 | 0.534211 | 0.530000 | 0.2200 | 0.2200 | 0.3400 |
| toycar | 0.499815 | 0.471622 | 0.516215 | 0.484211 | 0.504737 | 0.498421 | 0.4200 | 0.2200 | 0.2200 |
| toytrain | 0.522382 | 0.527997 | 0.514497 | 0.503158 | 0.511053 | 0.504211 | 0.3600 | 0.1800 | 0.1800 |
| valve | 0.568928 | 0.559355 | 0.585521 | 0.563684 | 0.563684 | 0.620526 | 0.1200 | 0.2000 | 0.1800 |

## 5. Matched Logmel Comparison

| experiment | logmel_only hmean | experiment hmean | delta | logmel_only pAUC | experiment pAUC | delta |
|---|---:|---:|---:|---:|---:|---:|
| full_5view | 0.543587 | 0.553511 | +0.009924 | 0.524586 | 0.547820 | +0.023234 |
| multires_7view | 0.543587 | 0.556256 | +0.012669 | 0.524586 | 0.553233 | +0.028647 |

## 6. Does V29 Beat V16?

No.

`multires_7view / latent_summary_knn`:

```text
mean_hmean = 0.556256 < 0.564294
mean_pAUC  = 0.553233 < 0.553459
```

## 7. Does V29 Keep pAUC?

Almost, but not enough to pass the frozen criterion.

```text
V16 mean_pAUC  = 0.553459
V29 mean_pAUC  = 0.553233
delta          = -0.000226
```

## 8. Does V29 Harm KS?

Yes, relative to the frozen V16 mainline.

```text
V16 mean_KS = 0.2886
V29 mean_KS = 0.3286
delta       = +0.0400
```

## 9. Are At Least 4/7 Machines Not Below V16?

No. Only `3/7` machines are not below V16 on hmean.

Machines meeting `multires_7view >= V16`:

- `bearing`
- `slider`
- `valve`

## 10. Final Verdict

```text
V29 does not cleanly beat V16.
```

Detailed verdict:

- `multires_7view` does beat the matched `logmel_only` baseline
- it nearly preserves `pAUC`
- but it fails the frozen success gate because:
  `hmean` does not exceed `0.564294`
  `pAUC` does not reach `0.553459`
  `KS` is clearly worse than `0.2886`

Final sentence:

```text
V16 remains the frozen CTAF-v1 mainline.
```

Failure analysis:

- multi-resolution logmel does add useful anomaly signal on some machines
- the benefit is not uniform enough to overcome the increased source-target drift
- the main weakness is not raw pAUC anymore, but balanced cross-machine hmean
  under stable score geometry
