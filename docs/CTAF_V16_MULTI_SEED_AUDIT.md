# CTAF V16 Multi-Seed Stability Audit

## Frozen Baseline

- baseline dir: `/Users/apple/Project/codex/LAES_2025/new_projects/clean_tensor_asd/results/v16_viewwise_unshared_late_fusion`
- focus: `full_5view / latent_summary_knn`
- matched baseline: `logmel_only / latent_summary_knn`
- frozen hmean: `0.564294`
- frozen pAUC: `0.553459`
- frozen KS: `0.2886`

## Seed Panel

| seed | logmel hmean | V16 hmean | delta | logmel pAUC | V16 pAUC | delta | V16 KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13711 | 0.543587 | 0.553511 | +0.009924 | 0.524586 | 0.547820 | +0.023234 | 0.3286 |
| 13721 | 0.532118 | 0.550205 | +0.018087 | 0.518571 | 0.544962 | +0.026391 | 0.2943 |
| 13731 | 0.536815 | 0.548404 | +0.011589 | 0.526992 | 0.541654 | +0.014662 | 0.3086 |
| 13741 | 0.543400 | 0.553533 | +0.010133 | 0.539173 | 0.543008 | +0.003835 | 0.2686 |
| 13751 | 0.540003 | 0.542080 | +0.002077 | 0.536767 | 0.534211 | -0.002556 | 0.2743 |

## Aggregate Stability

| metric | mean | std | min | max | baseline | delta(mean-baseline) |
|---|---:|---:|---:|---:|---:|---:|
| hmean | 0.549547 | 0.004720 | 0.542080 | 0.553533 | 0.564294 | -0.014747 |
| pAUC | 0.542331 | 0.005097 | 0.534211 | 0.547820 | 0.553459 | -0.011128 |
| KS | 0.294880 | 0.024697 | 0.268600 | 0.328600 | 0.2886 | +0.006280 |

## Per-Machine Stability

| machine | baseline hmean | seed-panel hmean mean | std | baseline pAUC | seed-panel pAUC mean | std | observation |
|---|---:|---:|---:|---:|---:|---:|---|
| bearing | 0.619938 | 0.636343 | 0.015200 | 0.600526 | 0.603368 | 0.016951 | hmean delta +0.016405, pAUC delta +0.002842 |
| fan | 0.599299 | 0.572991 | 0.013326 | 0.576842 | 0.562947 | 0.026383 | hmean delta -0.026308, pAUC delta -0.013895 |
| gearbox | 0.543746 | 0.542791 | 0.026038 | 0.523158 | 0.524632 | 0.014758 | hmean delta -0.000956, pAUC delta +0.001474 |
| slider | 0.529575 | 0.522276 | 0.026034 | 0.545263 | 0.528316 | 0.005462 | hmean delta -0.007300, pAUC delta -0.016947 |
| toycar | 0.562915 | 0.501151 | 0.036409 | 0.559474 | 0.503263 | 0.009020 | hmean delta -0.061764, pAUC delta -0.056211 |
| toytrain | 0.529068 | 0.517280 | 0.012142 | 0.516842 | 0.511895 | 0.009286 | hmean delta -0.011788, pAUC delta -0.004947 |
| valve | 0.565519 | 0.553996 | 0.033082 | 0.552105 | 0.561895 | 0.028593 | hmean delta -0.011524, pAUC delta +0.009789 |

## Verdict

- seeds beating matched logmel_only on both hmean and pAUC: `4/5`
- seeds beating frozen V16 point estimate on both hmean and pAUC: `0/5`
- hmean std across seeds: `0.004720`; pAUC std: `0.005097`; KS std: `0.024697`
- V16 is directionally stable: `full_5view` beats the matched `logmel_only` baseline on hmean in all `5/5` seeds, and beats it on both hmean and pAUC in `4/5` seeds.
- The original frozen V16 single-point result is not stable enough to remain the only hard baseline. The seed-panel mean is lower by `0.014747` hmean and `0.011128` pAUC.
- The direct rerun of seed `13711` does not reproduce the original frozen V16 numbers:
  - frozen V16: `hmean=0.564294`, `pAUC=0.553459`, `KS=0.2886`
  - rerun seed 13711: `hmean=0.553511`, `pAUC=0.547820`, `KS=0.3286`
- The biggest instability is concentrated in `toycar`, with secondary weakness in `fan` and `slider`. `bearing` is the most stable positive contributor.

## Updated Baseline Rule

From this point onward, the effective V16 baseline is the fixed seed-panel mean,
not the original frozen single-seed point.

Effective baseline:

```text
mean_hmean = 0.549547
mean_pAUC  = 0.542331
mean_KS    = 0.294880
```

The original frozen V16 point:

```text
hmean = 0.564294
pAUC  = 0.553459
KS    = 0.2886
```

is still useful, but only as a stronger candidate threshold, not as the only
hard replacement gate.

## Evaluation Gates

### A. Matched Baseline Gate

Any new structure must first beat the matched `logmel_only` baseline.

### B. Seed-Panel Mean Gate

Any new structure evaluated on a single seed must at least exceed the V16
seed-panel mean:

```text
hmean > 0.549547
pAUC  >= 0.542331
KS not clearly worse than 0.294880
```

### C. Candidate Gate

If a new structure on a single seed exceeds the original frozen V16 point:

```text
hmean > 0.564294
pAUC  >= 0.553459
```

it can only be labeled:

```text
candidate replacement
```

and must pass a multi-seed recheck before any freeze decision.

### D. Freeze Gate

A new structure can replace V16 only if:

- its `3-5` seed mean exceeds the V16 seed-panel mean
- the majority of its seeds beat the matched `logmel_only` baseline
- `KS` is not clearly degraded

## Practical Conclusion

```text
Do not use a single-seed result to declare a new best method.
All future structure innovation must be judged against the V16 seed-panel rule.
```
