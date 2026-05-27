# CODEX HANDOFF STAGE22

## 1. Stage22D frozen result

Stage22B/C geometry-objective adapted EAT produced a weak hmean-positive full7 route:

- route: `adapted_ldn_diff_K1_scorek5`
- route description: `Stage22 geometry-objective adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5`
- mean_hmean: `0.6166917954459288`
- mean_pAUC: `0.5581954887218045`
- mean_TPR@99n: `0.10142857142857142`
- delta_hmean_vs_clean_best: `0.0007450666033770315`
- delta_pAUC_vs_clean_best: `-0.002105263157894832`
- delta_TPR99_vs_clean_best: `-0.0042857142857142955`
- improved_machine_count_vs_adapted_baseline: `5`
- strong_machine_severe_collapse: `False`
- freeze_status: `weak_hmean_positive_freeze`

Conclusion: this is a new hmean best, but not a balanced metric improvement.

## 2. Machine-level pattern

- positive machines vs adapted baseline: `bearing, gearbox, slider, toycar, valve`
- negative machines vs adapted baseline: `fan, toytrain`
- strongest gain: `valve (+0.062674276665)`
- largest drop: `fan (-0.026327961558)`

The gain is mostly carried by valve, with additional gains on gearbox, slider, bearing, and toycar. Fan drops substantially; toytrain drops slightly.

## 3. Current route hierarchy

- hmean-best route: `Stage22D geometry-objective adapted EAT + adapted_ldn_diff_K1_scorek5`
- balanced reference route: `EAT min_clamp__mean + ldn_diff_K1_scorek5`

Use Stage22D only when optimizing mean hmean. Use the clean frozen route as the balanced reference because Stage22D lowers mean pAUC and mean TPR@99n.

## 4. Forbidden next moves

- Do not tune `LDN` `K / r / score_k`.
- Do not add per-machine selector or fusion to hide fan degradation.
- Do not claim pAUC or TPR@99n improvement.
- Do not use test labels for checkpoint selection.
- Do not continue fan-only adaptation as a main route.

## 5. Recommended next step

Stop Stage22B as a route-search branch and freeze the result as `weak_hmean_positive_freeze`. If continuing, design a new objective specifically to recover fan without sacrificing the valve gain, but it must be validated without selector, LDN tuning, or test-label checkpoint selection.
