# CTAF Stage20A Robust LDN Reference-Density Full7 Implementation

## Goal

Audit fixed full-7 LDN candidates by changing only the reference-density estimator `rho_i`.

## Fixed Inputs

- existing Stage16D embeddings only
- machines:
  - `bearing`
  - `fan`
  - `gearbox`
  - `slider`
  - `toycar`
  - `toytrain`
  - `valve`

## Fixed Candidates

- `baseline_knn_k5`
- `ldn_diff_K1_scorek5`
- `ldn_diff_K5_medianrho_scorek5`
- `ldn_diff_K10_trim20rho_scorek5`
- `ldn_diff_K10_winsor20rho_scorek5`
- `ldn_diff_core80_K5_medianrho_scorek5`

## Rules

- no model load
- no forward
- no training
- no embedding extraction
- no pooling changes
- no min-clamp changes
- no selector
- no fusion
- no oracle
- no candidate tuning after seeing results

## Scoring

- L2 normalize train/test embeddings
- use chord distance
- `score_k = 5`
- anomaly score = mean of 5 smallest normalized distances

## Promotion Logic

- compare mean hmean to Stage18F best `0.6159467288425517`
- prefer `>= 0.620`
- reject any candidate with strong-machine severe collapse
- require improvement across more than one machine
- reject candidates that are worse than the reference on more than 3 machines

## Outputs

- `results/stage20a_robust_ldn_reference_density_full7/`
- `docs/CTAF_STAGE20A_ROBUST_LDN_REFERENCE_DENSITY_FULL7_AUDIT.md`
