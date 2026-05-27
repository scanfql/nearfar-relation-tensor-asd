# CTAF Stage20A Robust LDN Reference-Density Full7 Audit

## Summary

Stage20A is a frozen full-7 audit over a fixed candidate set. It keeps the Stage16D `EAT min_clamp__mean` embeddings unchanged and varies only the LDN reference-density estimator.

## Fixed Candidate Set

- `baseline_knn_k5`
- `ldn_diff_K1_scorek5`
- `ldn_diff_K5_medianrho_scorek5`
- `ldn_diff_K10_trim20rho_scorek5`
- `ldn_diff_K10_winsor20rho_scorek5`
- `ldn_diff_core80_K5_medianrho_scorek5`

## Guardrails

- no model load
- no forward
- no training
- no selector
- no fusion
- no oracle
- no per-machine choice after results

## Decision Rule

- best mean hmean must exceed Stage18F best `0.6159467288425517`
- `>= 0.620` is preferred
- strong-machine collapse is a hard stop
- improvement must not be driven by only one machine

## Promotion

The audit should state whether promotion is recommended and, if not, whether the branch should stop or move to a lighter pooling audit.
