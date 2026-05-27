# CODEX HANDOFF STAGE23

## 1. Stage22 status after Stage23 diagnosis

Stage22D remains frozen as a weak hmean-positive route:

- freeze_status: `weak_hmean_positive_freeze`
- route: `Stage22 geometry-objective adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5`
- mean_hmean: `0.6166917954459288`
- delta_hmean_vs_clean_best: `0.0007450666033770315`
- delta_pAUC_vs_clean_best: `-0.002105263157894832`
- delta_TPR99_vs_clean_best: `-0.0042857142857142955`

Conclusion: Stage22D is a new hmean best only. Keep clean EAT + frozen LDN as the balanced reference.

## 2. Stage23A component probe

Single-component ablations did not identify a sufficient fan-harm culprit.

- no_ldn_proxy fan delta vs Stage22B LDN: `0.0024853641700368634`
- no_aug fan delta vs Stage22B LDN: `-0.005473662055527906`
- no_dispersion fan delta vs Stage22B LDN: `-0.0012132532388539063`
- candidate_for_full7_followup_count: `0`

Interpretation: LDN proxy has a small fan-harm signal, augmentation protects fan, and dispersion is not the main driver.

## 3. Stage23B fan density failure

The fan failure is backend-density mismatch under adapted geometry:

- all routes LDN worse than adapted baseline: `True`
- best fan adapted baseline route: `stronger_neighbor`
- best fan adapted baseline hmean: `0.5859456234041`
- worst LDN gap route: `stronger_neighbor`
- worst LDN gap hmean: `-0.03662288973639016`
- stronger_neighbor baseline hmean: `0.5859456234041`
- stronger_neighbor LDN hmean: `0.5493227336677098`
- stronger_neighbor LDN-baseline hmean: `-0.03662288973639016`

Key conclusion: stronger_neighbor improves fan baseline geometry, but frozen LDN subtraction amplifies the wrong fan tail geometry and destroys the score.

## 4. Current diagnosis

`fan harm is a frozen-density recalibration mismatch under adapted geometry, not a single loss-component failure.`

## 5. Forbidden next moves

- Do not tune `LDN` `K / r / score_k`.
- Do not add per-machine selector or fusion to hide fan degradation.
- Do not continue one-component ablations as if they identified a culprit.
- Do not increase neighbor/anchor blindly; stronger_neighbor raised fan baseline but worsened LDN.
- Do not claim Stage22 improves pAUC or TPR@99n.
- Do not use test labels for checkpoint selection.

## 6. Recommended next question

If continuing, ask: can a future objective preserve fan density ordering while keeping valve gain, without tuning LDN parameters or adding selectors?
