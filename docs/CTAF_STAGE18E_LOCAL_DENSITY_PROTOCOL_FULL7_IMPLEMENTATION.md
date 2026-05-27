# CTAF Stage18E Local Density Protocol Full7 Implementation

## 1. Modified File List

- `scripts/audit_stage18e_local_density_protocol_full7.py`
- `docs/CTAF_STAGE18E_LOCAL_DENSITY_PROTOCOL_FULL7_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the full7 frozen local-density protocol audit on top of existing `EAT min_clamp__mean` embeddings.
- It is strictly score-level only:
  - no model load
  - no forward
  - no embedding extraction
  - no training
  - no LoRA
  - no adapter
  - no Conformer
  - no fusion with external routes
  - no selector
  - no oracle
  - no alpha sweep
- It supports:
  - `--mode check`
  - `--mode score_ldn`
- It reads only:
  - `results/stage16d_minclamp_mean_full7/embeddings/<machine>_minclamp_mean_embeddings.npz`

## 3. Fixed Candidate Set

- `baseline_knn_k5`
- `ldn_ratio_K1_scorek5`
- `ldn_diff_K1_scorek5`
- `ldn_ratio_K5_scorek5`
- `ldn_diff_K5_scorek5`
- `gwrp_ratio_r09_scorek5`
- `gwrp_diff_r09_scorek5`
- `rasdsr_ratio_avg_K5_r09_scorek5`

Fixed parameters:

- `eps = 1e-12`
- local density `K ∈ {1, 5}`
- `gwrp r = 0.9`
- score aggregation `score_k = 5`

## 4. Scoring Protocol

- input embeddings are L2-normalized
- distance is chord/cosine distance:
  - `sqrt(max(||x||^2 + ||y||^2 - 2 x@y, 0))`
- source train-normal only reference memory
- no test labels in scoring
- metrics only after scoring:
  - `hmean`
  - `pAUC`
  - `KS`
  - `TPR@95n`
  - `TPR@99n`
  - `q95`
  - `q99`
  - `anomaly_median`
  - `anomaly_q75`

## 5. Expected Outputs

Base output root:

- `results/stage18e_local_density_protocol_full7/`

Expected files:

- `check_status.json`
- `embedding_check.csv`
- `summary.csv`
- `per_machine.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `comparison_vs_baseline.csv`
- `candidate_rank.csv`
- `collapse_check.csv`
- `stage18e_ldn_verdict.json`
- `scores/<candidate>/<machine>_scores.npz`
- `docs/CTAF_STAGE18E_LOCAL_DENSITY_PROTOCOL_FULL7_AUDIT.md`

## 6. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18e_local_density_protocol_full7.py --mode check --embedding_root results/stage16d_minclamp_mean_full7/embeddings --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage18e_local_density_protocol_full7 --doc_path docs/CTAF_STAGE18E_LOCAL_DENSITY_PROTOCOL_FULL7_AUDIT.md --latent_knn_k 5
```

## 7. Score_LDN Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18e_local_density_protocol_full7.py --mode score_ldn --embedding_root results/stage16d_minclamp_mean_full7/embeddings --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage18e_local_density_protocol_full7 --doc_path docs/CTAF_STAGE18E_LOCAL_DENSITY_PROTOCOL_FULL7_AUDIT.md --latent_knn_k 5
```

## 8. Confirmation

- no experiment was run in Codex
- no scoring was run in Codex
