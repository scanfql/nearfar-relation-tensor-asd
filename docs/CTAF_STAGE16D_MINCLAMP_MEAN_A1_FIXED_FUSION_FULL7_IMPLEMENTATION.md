# CTAF Stage16D MinClamp Mean A1 Fixed Fusion Full7 Implementation

## 1. Modified File List

- `scripts/audit_stage16d_minclamp_mean_a1_fixed_fusion_full7.py`
- `docs/CTAF_STAGE16D_MINCLAMP_MEAN_A1_FIXED_FUSION_FULL7_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements a pure score-level full7 fixed fusion audit for `min_clamp__mean + A1`.
- It is strictly score-level only:
  - no model load
  - no model forward
  - no embedding extraction
  - no training
  - no LoRA
  - no adapter
  - no Conformer
  - no alpha sweep
  - no per-machine selection
  - no oracle
- The script supports:
  - `--mode check`
  - `--mode fixed_fusion`
- Fixed candidates only:
  - `min_clamp__mean`
  - `A1`
  - `F_minclampmean_A1_050`
  - `F_minclampmean_A1_065`
- No additional candidates are added.

## 3. Fixed Inputs

- min_clamp__mean score root:
  - `results/stage16d_minclamp_mean_full7`
- A1 score root:
  - `results/stage14a_frozen_beats_v63_fusion`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seeds:
  - `13711,13721,13731`

## 4. Fixed Fusion Formulas

Rank normalization:

- `rank_norm(score) = (rank(score) - 1) / (N - 1)`

Fixed fusion candidates:

- `F_minclampmean_A1_050 = 0.5 * rank_norm(min_clamp__mean) + 0.5 * rank_norm(A1)`
- `F_minclampmean_A1_065 = 0.65 * rank_norm(min_clamp__mean) + 0.35 * rank_norm(A1)`

## 5. Alignment Rules

For every machine and seed:

- `y_test` must match
- `domains / d_test` must match
- `test_paths` must match if available

If alignment fails:

- `alignment_summary.csv` is written
- the script raises an error
- it does not silently continue

## 6. Expected Outputs

Base root:

- `results/stage16d_minclamp_mean_a1_fixed_fusion_full7/`

Files:

- `check_status.json`
- `check_files.csv`
- `alignment_summary.csv`
- `per_machine_seed_metrics.csv`
- `seed_summary.csv`
- `summary.csv`
- `per_machine.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `collapse_check.csv`
- `reference_comparison.csv`
- `fusion_verdict.json`
- `scores/<candidate>/seed_<seed>/<machine>_scores.npz`
- `docs/CTAF_STAGE16D_MINCLAMP_MEAN_A1_FIXED_FUSION_FULL7_AUDIT.md`

## 7. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16d_minclamp_mean_a1_fixed_fusion_full7.py --mode check --data_root /data1/test/LAES_2025/data/dcase --minclamp_root results/stage16d_minclamp_mean_full7 --stage14a_root results/stage14a_frozen_beats_v63_fusion --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seeds 13711,13721,13731 --output_root results/stage16d_minclamp_mean_a1_fixed_fusion_full7 --doc_path docs/CTAF_STAGE16D_MINCLAMP_MEAN_A1_FIXED_FUSION_FULL7_AUDIT.md
```

## 8. Fixed Fusion Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16d_minclamp_mean_a1_fixed_fusion_full7.py --mode fixed_fusion --data_root /data1/test/LAES_2025/data/dcase --minclamp_root results/stage16d_minclamp_mean_full7 --stage14a_root results/stage14a_frozen_beats_v63_fusion --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seeds 13711,13721,13731 --output_root results/stage16d_minclamp_mean_a1_fixed_fusion_full7 --doc_path docs/CTAF_STAGE16D_MINCLAMP_MEAN_A1_FIXED_FUSION_FULL7_AUDIT.md
```

## 9. Confirmation

- no experiment was run in Codex
- no scoring was run in Codex
