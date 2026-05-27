# CTAF V47 Cepstral Frontend Audit

## 1. V46 Stage-0 Summary

This experiment is unlocked by `V46`, which repaired the cepstral frontend
implementation and re-ran the fixed Stage-0 prescreen.

Key Stage-0 facts:

- NaN-safe upsampling fix removed the implementation-invalid status
- `machines passing Gate A = 5/5`
- `machines passing all gates = 4/5`
- `max_redundancy_r = 0.608859 <= 0.85`
- `mean_robust_d_missed = 2.930138`
- `toycar robust_d_missed = 5.904637`
- `stage0_clearance = True`

Therefore `cepstral_3view` enters `V47` as a valid frontend candidate.

## 2. Engineering Registration

Required engineering registration for `V47`:

- `src/features/tensor_builder.py` supports `cepstral_3view`
- `src/run_experiment.py` recognizes `cepstral_3view`
- checkpoint saving is enabled under:
  `results/v47_cepstral_frontend_seed13711/models/<machine>_<experiment>_seed13711.pt`

## 3. Fixed Training Configuration

```text
experiments = logmel_only, full_5view, cepstral_3view
model_type = viewwise_unshared_late_fusion
score_modes = latent_summary_knn
latent_knn_k = 5
latent_channels = 64
stem_channels = 8
per_view_channels = 16
dropout_p = 0.0
training = source normal only
memory bank = source normal only
loss_view_mode = all_views
seed = 13711
epochs = 50
batch_size = 32
```

## Pending Audit

The completed report will include:

1. V46 Stage-0 summary
2. engineering registration and checkpoint path
3. overall comparison:
   - logmel_only / latent_summary_knn
   - full_5view / latent_summary_knn
   - cepstral_3view / latent_summary_knn
4. per-machine hmean / pAUC / KS
5. bottleneck-machine tail table
6. toycar-specific observation
7. final single-seed verdict

## 4. Overall Comparison

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.543587 | 0.524586 | 0.4029 |
| full_5view | latent_summary_knn | 0.553511 | 0.547820 | 0.3286 |
| cepstral_3view | latent_summary_knn | 0.507016 | 0.513684 | 0.2629 |

Direct comparison:

```text
cepstral_3view vs matched logmel_only:
hmean delta = -0.036571
pAUC  delta = -0.010902
KS    delta = -0.1400

cepstral_3view vs full_5view:
hmean delta = -0.046495
pAUC  delta = -0.034136
KS    delta = -0.0657
```

So `cepstral_3view` improves geometry, but not discrimination.

## 5. Per-Machine hmean / pAUC / KS

| machine | experiment | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| bearing | logmel_only | 0.534217 | 0.524737 | 0.28 |
| bearing | full_5view | 0.638861 | 0.594737 | 0.28 |
| bearing | cepstral_3view | 0.514263 | 0.533684 | 0.10 |
| fan | logmel_only | 0.602736 | 0.581053 | 1.00 |
| fan | full_5view | 0.578160 | 0.580526 | 1.00 |
| fan | cepstral_3view | 0.572065 | 0.554211 | 0.94 |
| gearbox | logmel_only | 0.556221 | 0.514737 | 0.42 |
| gearbox | full_5view | 0.552767 | 0.545789 | 0.20 |
| gearbox | cepstral_3view | 0.518570 | 0.498421 | 0.10 |
| slider | logmel_only | 0.520811 | 0.500526 | 0.22 |
| slider | full_5view | 0.545818 | 0.534211 | 0.22 |
| slider | cepstral_3view | 0.490260 | 0.496842 | 0.08 |
| toycar | logmel_only | 0.499815 | 0.484211 | 0.42 |
| toycar | full_5view | 0.471622 | 0.504737 | 0.22 |
| toycar | cepstral_3view | 0.495435 | 0.507895 | 0.28 |
| toytrain | logmel_only | 0.522382 | 0.503158 | 0.36 |
| toytrain | full_5view | 0.527997 | 0.511053 | 0.18 |
| toytrain | cepstral_3view | 0.526426 | 0.517895 | 0.14 |
| valve | logmel_only | 0.568928 | 0.563684 | 0.12 |
| valve | full_5view | 0.559355 | 0.563684 | 0.20 |
| valve | cepstral_3view | 0.432093 | 0.486842 | 0.20 |

Pattern:

- `bearing` and `valve` collapse relative to `full_5view`
- `toycar` and `toytrain` get slightly better `pAUC` than both `logmel_only`
  and `full_5view`
- but these local gains do not scale to a good overall score

## 6. Bottleneck-Machine Tail Table

Compared against `V16 full_5view / latent_summary_knn`:

| machine | model | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| gearbox | V16_full5 | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| gearbox | V47_cepstral | 0.1999 | 0.2139 | 0.1129 | 0.1328 | 0.06 | 0.02 |
| slider | V16_full5 | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| slider | V47_cepstral | 0.1665 | 0.2199 | 0.1242 | 0.1333 | 0.05 | 0.00 |
| toycar | V16_full5 | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| toycar | V47_cepstral | 0.1005 | 0.1239 | 0.0908 | 0.0958 | 0.09 | 0.00 |
| toytrain | V16_full5 | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| toytrain | V47_cepstral | 0.1066 | 0.1300 | 0.0943 | 0.0994 | 0.12 | 0.00 |
| fan | V16_full5 | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |
| fan | V47_cepstral | 0.2022 | 0.2087 | 0.1771 | 0.1973 | 0.14 | 0.09 |

Low-FPR gate:

```text
At least 2 bottleneck machines must improve TPR@99n.
Observed: 0/5 improve.
```

## 7. Toycar-Specific Observation

`V46` made `toycar` the strongest input-layer cepstral signal:

```text
toycar robust_d_missed = 5.904637
```

But after training:

- `toycar pAUC` is only mildly positive:
  - `logmel_only = 0.484211`
  - `full_5view = 0.504737`
  - `cepstral_3view = 0.507895`
- `toycar TPR@99n` collapses:
  - `V16_full5 = 0.10`
  - `V47_cepstral = 0.00`

Interpretation:

```text
The strong Stage-0 input-level cepstral signal on toycar does not survive
training as useful low-FPR tail separation.
```

So the cepstral frontend does not justify any toycar-specific conclusion or
per-machine treatment.

## 8. Checkpoint Paths

Checkpoint saving is active and clear:

```text
results/v47_cepstral_frontend_seed13711/models/<machine>_<experiment>_seed13711.pt
```

Examples:

- `results/v47_cepstral_frontend_seed13711/models/bearing_cepstral_3view_seed13711.pt`
- `results/v47_cepstral_frontend_seed13711/models/gearbox_cepstral_3view_seed13711.pt`
- `results/v47_cepstral_frontend_seed13711/models/toycar_cepstral_3view_seed13711.pt`

## 9. Final Single-Seed Verdict

Success criteria check:

- `cepstral_3view hmean > matched logmel_only`: `FAIL`
- `cepstral_3view hmean > full_5view`: `FAIL`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least `2` bottleneck machines improve `TPR@99n`: `FAIL`

Final verdict:

```text
V47 single-seed fails.
```

Interpretation:

- `V46` correctly showed that cepstral_3view is a valid input-level candidate
- but `V47` shows that the Stage-0 cepstral signal does not convert into
  trained low-FPR tail separation under the fixed V16 training pipeline
- this is now a scientific negative, not an implementation failure

Therefore:

```text
Do not continue to a V47 seed panel.
Do not tune cepstral parameters.
Do not try cepstral_6view or fusion from this result.
Record this as a scientific failure analysis.
```
