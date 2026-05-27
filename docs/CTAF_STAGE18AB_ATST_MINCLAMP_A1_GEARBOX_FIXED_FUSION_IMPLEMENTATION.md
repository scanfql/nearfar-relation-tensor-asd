# CTAF Stage18A-B ATST MinClamp A1 Gearbox Fixed Fusion Implementation

## 1. Modified File List

- `scripts/audit_stage18ab_atst_minclamp_a1_gearbox_fixed_fusion.py`
- `docs/CTAF_STAGE18AB_ATST_MINCLAMP_A1_GEARBOX_FIXED_FUSION_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the gearbox-only pure score-level fixed fusion audit for:
  - `ATST_knn`
  - `min_clamp__mean`
  - `A1`
- It is strictly score-level only:
  - no model load
  - no forward
  - no embedding extraction
  - no training
  - no LoRA
  - no adapter
  - no alpha sweep
  - no per-machine selector
  - no oracle
- It supports:
  - `--mode check`
  - `--mode fixed_fusion`

## 3. Fixed Inputs

- `ATST_knn`:
  - `results/stage18a_atst_one_machine_knn/scores/gearbox_atst_knn_scores.npz`
- `min_clamp__mean`:
  - `results/stage16d_minclamp_mean_full7/scores/gearbox_minclamp_mean_scores.npz`
- `A1`:
  - `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz`

## 4. Fixed Fusion Candidates

Rank normalization:

- `rank_norm(score) = (rank(score) - 1) / (N - 1)`

Fixed candidates:

- `F_ATST_minclamp_050 = 0.5 * rank_norm(ATST_knn) + 0.5 * rank_norm(min_clamp__mean)`
- `F_ATST_A1_050 = 0.5 * rank_norm(ATST_knn) + 0.5 * rank_norm(A1)`
- `F_minclamp_A1_050 = 0.5 * rank_norm(min_clamp__mean) + 0.5 * rank_norm(A1)`
- `F_ATST_minclamp_A1_TRI = (rank_norm(ATST_knn) + rank_norm(min_clamp__mean) + rank_norm(A1)) / 3`

No additional candidates are added.

## 5. Expected Outputs

Base root:

- `results/stage18ab_atst_minclamp_a1_gearbox_fixed_fusion/`

Files:

- `check_status.json`
- `alignment_summary.csv`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `reference_comparison.csv`
- `fusion_verdict.json`
- `scores/<candidate>/gearbox_scores.npz`
- `docs/CTAF_STAGE18AB_ATST_MINCLAMP_A1_GEARBOX_FIXED_FUSION_AUDIT.md`

## 6. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18ab_atst_minclamp_a1_gearbox_fixed_fusion.py --mode check --machine gearbox --seed 13711 --output_root results/stage18ab_atst_minclamp_a1_gearbox_fixed_fusion --doc_path docs/CTAF_STAGE18AB_ATST_MINCLAMP_A1_GEARBOX_FIXED_FUSION_AUDIT.md --atst_score_path results/stage18a_atst_one_machine_knn/scores/gearbox_atst_knn_scores.npz --minclamp_score_path results/stage16d_minclamp_mean_full7/scores/gearbox_minclamp_mean_scores.npz --a1_score_path results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz
```

## 7. Fixed Fusion Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18ab_atst_minclamp_a1_gearbox_fixed_fusion.py --mode fixed_fusion --machine gearbox --seed 13711 --output_root results/stage18ab_atst_minclamp_a1_gearbox_fixed_fusion --doc_path docs/CTAF_STAGE18AB_ATST_MINCLAMP_A1_GEARBOX_FIXED_FUSION_AUDIT.md --atst_score_path results/stage18a_atst_one_machine_knn/scores/gearbox_atst_knn_scores.npz --minclamp_score_path results/stage16d_minclamp_mean_full7/scores/gearbox_minclamp_mean_scores.npz --a1_score_path results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz
```

## 8. Confirmation

- no experiment was run in Codex
- no scoring was run in Codex
