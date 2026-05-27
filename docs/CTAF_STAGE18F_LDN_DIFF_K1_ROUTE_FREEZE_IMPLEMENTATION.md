# CTAF Stage18F LDN-Diff-K1 Route Freeze Implementation

## 1. Modified File List

- `scripts/audit_stage18f_ldn_diff_k1_route_freeze.py`
- `docs/CTAF_STAGE18F_LDN_DIFF_K1_ROUTE_FREEZE_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the paper-ready freeze / confirmation audit for the Stage18E best route:
  - `ldn_diff_K1_scorek5`
- It is strictly read-only on Stage18E result artifacts:
  - no model load
  - no forward
  - no embedding extraction
  - no training
  - no score recomputation
  - no parameter tuning
  - no extra candidates
- It supports:
  - `--mode check`
  - `--mode freeze_audit`
- `check` verifies required files, required columns, score file presence, and frozen-candidate presence.
- `freeze_audit` reads existing Stage18E outputs and produces:
  - route summary
  - candidate ranking for paper tables
  - per-machine delta table
  - tail delta table
  - strong-machine guardrail table
  - score-file alignment table
  - main and ablation paper-ready tables
  - final freeze verdict JSON

## 3. Frozen Route

- frozen route:
  - `ldn_diff_K1_scorek5`
- baseline route:
  - `baseline_knn_k5`
- comparison routes included:
  - `ldn_diff_K5_scorek5`
  - `gwrp_diff_r09_scorek5`
  - `rasdsr_ratio_avg_K5_r09_scorek5`
- registered old best:
  - `F_P1_mean_std_A1_065 = 0.607175`
- registered EAT min_clamp baseline:
  - `0.604713`

## 4. Expected Outputs

Base output root:

- `results/stage18f_ldn_diff_k1_route_freeze/`

Expected files:

- `check_status.json`
- `freeze_audit_status.json`
- `route_summary.csv`
- `candidate_ranking_paper.csv`
- `per_machine_delta_paper.csv`
- `tail_delta_paper.csv`
- `strong_machine_guardrail.csv`
- `score_file_alignment_check.csv`
- `paper_ready_table_main.csv`
- `paper_ready_table_ablation.csv`
- `stage18f_route_freeze_verdict.json`
- `docs/CTAF_STAGE18F_LDN_DIFF_K1_ROUTE_FREEZE_AUDIT.md`

## 5. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18f_ldn_diff_k1_route_freeze.py --mode check --stage18e_root results/stage18e_local_density_protocol_full7 --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage18f_ldn_diff_k1_route_freeze --doc_path docs/CTAF_STAGE18F_LDN_DIFF_K1_ROUTE_FREEZE_AUDIT.md
```

## 6. Freeze Audit Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18f_ldn_diff_k1_route_freeze.py --mode freeze_audit --stage18e_root results/stage18e_local_density_protocol_full7 --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage18f_ldn_diff_k1_route_freeze --doc_path docs/CTAF_STAGE18F_LDN_DIFF_K1_ROUTE_FREEZE_AUDIT.md
```

## 7. Confirmation

- no experiment was run in Codex
- no scoring was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
