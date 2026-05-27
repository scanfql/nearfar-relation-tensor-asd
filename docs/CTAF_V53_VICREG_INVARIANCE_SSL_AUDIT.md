# CTAF V53 VICReg Invariance SSL Audit

## 1. V52 Prereg Summary

V53 is the first executed Stage-9 pilot under the approved V52 preregistration.

Key frozen intent:

- do not change frontend
- do not change latent_summary pooling
- do not search scoring rules
- do not use negative-pair contrastive learning
- train only a non-contrastive invariance SSL objective on `full_5view`

## 2. V53 Fixed Protocol

```text
SSL objective: VICReg only
Encoder: viewwise_unshared_late_fusion
latent_channels = 64
stem_channels = 8
per_view_channels = 16
dropout_p = 0.0

Representation for scoring:
encoder latent_summary = channel-wise mean + std + max
dimension = 192

Projector:
192 -> 256 -> 256
ReLU
dropout = 0.0

VICReg weights:
lambda_inv = 25
lambda_var = 25
lambda_cov = 1

Augmentations:
small time mask
small frequency mask
mild gain jitter
time_mask_frac = 0.05
freq_mask_frac = 0.05
gain_jitter_range = [0.95, 1.05]
n_augmentations_per_view = 2
no time shift

Training:
full_5view only
source train-normal only
seed = 13711 only
no anomaly labels

Scoring:
latent_summary_knn
k = 5
```

## Pending Audit

The completed audit will report:

1. V52 prereg summary
2. V53 fixed protocol
3. overall comparison:
   - V37 full_5view latent_summary_knn baseline
   - V53 VICReg full_5view latent_summary_knn
4. per-machine hmean / pAUC / KS
5. tail table:
   - normal q95
   - normal q99
   - anomaly median
   - anomaly q75
   - TPR@95n
   - TPR@99n
6. representation-collapse diagnostics:
   - latent_summary feature std
   - projector output std
   - VICReg variance/covariance loss
7. final verdict:
   - V53 single-seed pass
   - V53 single-seed fail
   - unclear

## 3. Overall Comparison

Reference baseline:

```text
V37 full_5view / latent_summary_knn (seed 13711):
hmean = 0.563084
pAUC  = 0.550451
KS    = 0.3143
```

V53 result:

```text
V53 VICReg full_5view / latent_summary_knn:
hmean = 0.498014
pAUC  = 0.500000
KS    = 0.3771
```

Delta vs V37:

```text
hmean delta = -0.065070
pAUC  delta = -0.050451
KS    delta = +0.0628
```

Interpretation:

- the V53 single-seed pilot is materially below the frozen V37 baseline
- it does not preserve low-FPR behavior
- geometry also degrades

## 4. Per-Machine hmean / pAUC / KS

| machine | hmean | pAUC | KS |
|---|---:|---:|---:|
| bearing | 0.531869 | 0.505263 | 0.20 |
| fan | 0.528791 | 0.508421 | 1.00 |
| gearbox | 0.541414 | 0.508421 | 0.32 |
| slider | 0.508086 | 0.513158 | 0.38 |
| toycar | 0.499724 | 0.502632 | 0.32 |
| toytrain | 0.468608 | 0.488421 | 0.18 |
| valve | 0.407604 | 0.473684 | 0.24 |

Immediate pattern:

- `bearing` and especially `valve` collapse relative to V37
- `toycar` degrades
- bottleneck machines do not show the expected low-FPR gain

## 5. Tail Table

Comparison against `V37` baseline:

| machine | model | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---:|---:|---:|---:|---:|---:|
| bearing | V37 | 0.1533 | 0.1605 | 0.1286 | 0.1635 | 0.34 | 0.27 |
| bearing | V53 | 0.2253 | 0.2411 | 0.1820 | 0.2039 | 0.07 | 0.02 |
| fan | V37 | 0.2291 | 0.2385 | 0.1710 | 0.2235 | 0.21 | 0.12 |
| fan | V53 | 0.3272 | 0.3379 | 0.1742 | 0.2835 | 0.09 | 0.04 |
| gearbox | V37 | 0.1176 | 0.1296 | 0.0962 | 0.1038 | 0.10 | 0.06 |
| gearbox | V53 | 0.2034 | 0.2156 | 0.1711 | 0.1877 | 0.08 | 0.04 |
| slider | V37 | 0.1227 | 0.1596 | 0.1073 | 0.1161 | 0.11 | 0.00 |
| slider | V53 | 0.1741 | 0.2163 | 0.1298 | 0.1492 | 0.08 | 0.02 |
| toycar | V37 | 0.0779 | 0.0820 | 0.0713 | 0.0746 | 0.15 | 0.04 |
| toycar | V53 | 0.1476 | 0.1603 | 0.1271 | 0.1352 | 0.07 | 0.00 |
| toytrain | V37 | 0.0951 | 0.1220 | 0.0707 | 0.0750 | 0.07 | 0.03 |
| toytrain | V53 | 0.1473 | 0.1570 | 0.0909 | 0.0959 | 0.00 | 0.00 |
| valve | V37 | 0.1933 | 0.2130 | 0.1633 | 0.1840 | 0.19 | 0.10 |
| valve | V53 | 0.2426 | 0.2642 | 0.1731 | 0.1911 | 0.00 | 0.00 |

Tail interpretation:

- `normal q95/q99` inflate on every machine
- anomaly scores also rise, but not enough to clear the new normal extreme tail
- `TPR@99n` improves on no machine where improvement is actually needed under the Stage-9 gate

## 6. Representation-Collapse Diagnostics

The model does not collapse to a constant representation, but the diagnostics
still show unhealthy variance behavior.

Representative diagnostics from the saved `per_machine.csv`:

- `vicreg_lambda_inv = 25`
- `vicreg_lambda_var = 25`
- `vicreg_lambda_cov = 1`

Across machines, train-time diagnostics are in this rough range:

- latent_summary feature std:
  about `0.20` to `0.48`
- projector output std:
  about `1.03` to `1.16`
- VICReg variance loss:
  about `0.0035` to `0.0114`
- VICReg covariance loss:
  about `0.000004` to `1.596489`

Interpretation:

- there is no trivial collapse to zero-variance features
- but non-collapse alone is not sufficient
- the learned invariances appear to smooth normal and anomaly structure together
  rather than improving low-FPR separation

## 7. Final Verdict

Success gate check:

- `hmean >= V37 baseline - 0.003`: `FAIL`
- `pAUC >= V37 baseline`: `FAIL`
- at least 2 bottleneck machines improve `TPR@99n`: `FAIL`
- `bearing / valve` do not collapse: `FAIL`
- `toycar` does not clearly degrade: `FAIL`
- `normal q99` does not clearly inflate: `FAIL`
- gains are not carried by only one machine: `FAIL`

Final verdict:

```text
V53 single-seed fail
```

Interpretation:

- VICReg invariance SSL under the fixed Stage-9 protocol does not improve the
  current full_5view latent_summary scoring regime
- it produces broader normal tails and worse low-FPR ranking
- this is a scientific negative under the preregistered protocol
