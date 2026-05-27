# CTAF Stage20B Light Pooling Protocol Full7 Implementation

## Goal

Audit a fixed light-pooling family on frozen clean EAT embeddings only.
No training, no adaptation, no selector, no fusion.

## Fixed Pooling Candidates

- `min_clamp__mean`
- `min_clamp__mean_std`
- `min_clamp__q90`
- `min_clamp__mean_q90`
- `min_clamp__top10norm_mean`

## Fixed Backends

- `baseline_knn_k5`
- `ldn_diff_K1_scorek5`

## Protocol

- clean EAT checkpoint fixed
- 16 kHz
- 128-bin Kaldi fbank
- frame shift 10 ms
- target length 1024
- norm mean `-4.268`
- norm std `4.569`
- input shape `[B, 1, T, F]`
- min-clamp before pooling
- score_k fixed at 5

## Script Modes

- `check`: path, import, and inventory audit only
- `extract_light_pooling`: frozen forward and pooling extraction only
- `score_light_pooling`: score existing embeddings only

## Outputs

- `check_status.json`
- `machine_check.csv`
- `embedding_summary.csv`
- `pooling_diagnostics.csv`
- `embeddings/<pooling>/<machine>_embeddings.npz`
- `scores/<candidate>/<machine>_scores.npz`
- `summary.csv`
- `per_machine.csv`
- `candidate_rank.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `comparison_vs_stage18f.csv`
- `collapse_check.csv`
- `stage20b_light_pooling_verdict.json`
- `docs/CTAF_STAGE20B_LIGHT_POOLING_PROTOCOL_FULL7_AUDIT.md`

