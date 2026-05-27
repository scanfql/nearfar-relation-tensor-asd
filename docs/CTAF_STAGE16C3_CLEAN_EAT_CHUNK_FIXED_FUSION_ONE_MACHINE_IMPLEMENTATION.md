# CTAF Stage16C3 Clean EAT Chunk Fixed Fusion One-Machine Implementation

## 1. Modified File List

- `scripts/audit_stage16c3_clean_eat_chunk_fixed_fusion_one_machine.py`
- `docs/CTAF_STAGE16C3_CLEAN_EAT_CHUNK_FIXED_FUSION_ONE_MACHINE_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements Stage16C3 one-machine fixed score-level fusion on `gearbox`.
- It is strictly score-level only:
  - no model load
  - no forward
  - no embedding extraction
  - no training
  - no LoRA
  - no adapter
  - no alpha sweep
  - no per-machine selection
  - no oracle
  - no 7-machine run
- The script has two modes:
  - `--mode check`
  - `--mode fixed_fusion`
- `fixed_fusion` loads existing score bundles, performs strict alignment checks, computes only preregistered fixed fusion candidates, writes per-candidate score NPZs, summary tables, and verdict.

## 3. Fixed Input Score Paths

- P1:
  - `results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz`
- A1:
  - `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz`
- chunk candidate `C16S08_top1`:
  - `results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C16S08_top1/gearbox_scores.npz`
- chunk candidate `C08S04_top1`:
  - `results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C08S04_top1/gearbox_scores.npz`
- optional raw-frame best:
  - `results/stage16c_clean_eat_segment_one_machine_gearbox/scores/F_P1_S3_050/gearbox_scores.npz`

If the optional raw-frame score file is missing:

- do not fail
- skip raw-frame TRI fusion candidates
- still include registered raw-frame reference numbers in the comparison outputs

## 4. Fixed Fusion Formulas

Rank normalization:

- `rank_norm(score) = (rank(score) - 1) / (N - 1)`

Required candidates:

1. `F_P1_C16S08_top1_050`
   - `0.5 * rank_norm(P1) + 0.5 * rank_norm(C16S08_top1)`
2. `F_P1_C08S04_top1_050`
   - `0.5 * rank_norm(P1) + 0.5 * rank_norm(C08S04_top1)`
3. `F_A1_C16S08_top1_050`
   - `0.5 * rank_norm(A1) + 0.5 * rank_norm(C16S08_top1)`
4. `F_A1_C08S04_top1_050`
   - `0.5 * rank_norm(A1) + 0.5 * rank_norm(C08S04_top1)`
5. `F_P1_A1_C16S08_top1_TRI`
   - `(rank_norm(P1) + rank_norm(A1) + rank_norm(C16S08_top1)) / 3`
6. `F_P1_A1_C08S04_top1_TRI`
   - `(rank_norm(P1) + rank_norm(A1) + rank_norm(C08S04_top1)) / 3`

Optional candidates only if raw-frame best exists:

7. `F_P1_Fraw_C16S08_top1_TRI`
   - `(rank_norm(P1) + rank_norm(F_P1_S3_050) + rank_norm(C16S08_top1)) / 3`
8. `F_P1_Fraw_C08S04_top1_TRI`
   - `(rank_norm(P1) + rank_norm(F_P1_S3_050) + rank_norm(C08S04_top1)) / 3`

No additional candidates are added.

## 5. Alignment Rules

For every loaded score bundle, the script checks:

- `y_test` match
- `d_test / domains` match
- `test_paths` match if available

If any alignment fails:

- `alignment_summary.csv` is written
- the script raises an error
- it does not silently continue

## 6. Expected Outputs

Base output root:

- `results/stage16c3_clean_eat_chunk_fixed_fusion_one_machine_gearbox/`

Expected files:

- `check_status.json`
- `alignment_summary.csv`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `reference_comparison.csv`
- `fixed_fusion_verdict.json`
- `scores/<candidate>/gearbox_scores.npz`
- `docs/CTAF_STAGE16C3_CLEAN_EAT_CHUNK_FIXED_FUSION_ONE_MACHINE_GEARBOX_AUDIT.md`

## 7. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16c3_clean_eat_chunk_fixed_fusion_one_machine.py --mode check --machine gearbox --seed 13711 --output_root results/stage16c3_clean_eat_chunk_fixed_fusion_one_machine_gearbox --doc_path docs/CTAF_STAGE16C3_CLEAN_EAT_CHUNK_FIXED_FUSION_ONE_MACHINE_GEARBOX_AUDIT.md --p1_score_path results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz --a1_score_path results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz --c16_score_path results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C16S08_top1/gearbox_scores.npz --c08_score_path results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C08S04_top1/gearbox_scores.npz --fraw_score_path results/stage16c_clean_eat_segment_one_machine_gearbox/scores/F_P1_S3_050/gearbox_scores.npz
```

## 8. Fixed Fusion Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16c3_clean_eat_chunk_fixed_fusion_one_machine.py --mode fixed_fusion --machine gearbox --seed 13711 --output_root results/stage16c3_clean_eat_chunk_fixed_fusion_one_machine_gearbox --doc_path docs/CTAF_STAGE16C3_CLEAN_EAT_CHUNK_FIXED_FUSION_ONE_MACHINE_GEARBOX_AUDIT.md --p1_score_path results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz --a1_score_path results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz --c16_score_path results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C16S08_top1/gearbox_scores.npz --c08_score_path results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C08S04_top1/gearbox_scores.npz --fraw_score_path results/stage16c_clean_eat_segment_one_machine_gearbox/scores/F_P1_S3_050/gearbox_scores.npz
```

## 9. Confirmation

- no experiment was run in Codex
- no scoring was run in Codex
