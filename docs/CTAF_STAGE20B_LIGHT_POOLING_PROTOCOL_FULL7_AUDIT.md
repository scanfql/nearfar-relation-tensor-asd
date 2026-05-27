# CTAF Stage20B Light Pooling Protocol Full7 Audit

## Mode

- fixed light pooling audit
- full7 audit
- no selector
- no fusion
- no training
- no test-label candidate selection

## Fixed Candidate Set

- `min_clamp__mean__baseline_knn_k5`
- `min_clamp__mean__ldn_diff_K1_scorek5`
- `min_clamp__mean_std__baseline_knn_k5`
- `min_clamp__mean_std__ldn_diff_K1_scorek5`
- `min_clamp__q90__baseline_knn_k5`
- `min_clamp__q90__ldn_diff_K1_scorek5`
- `min_clamp__mean_q90__baseline_knn_k5`
- `min_clamp__mean_q90__ldn_diff_K1_scorek5`
- `min_clamp__top10norm_mean__baseline_knn_k5`
- `min_clamp__top10norm_mean__ldn_diff_K1_scorek5`

## Decision Criteria

- current Stage18F best hmean: `0.6159467288425517`
- promotion only if best candidate exceeds Stage18F best and clears guardrails
- strong-machine guardrail applies to `bearing`, `toytrain`, `valve`
- bottleneck focus: `gearbox`, `slider`, `toycar`

## Reproduction Check

- `min_clamp__mean` baseline is compared against Stage16D if the reference file is available.
