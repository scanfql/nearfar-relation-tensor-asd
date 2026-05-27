# CTAF Stage18G LDN-Diff-K1 Explainability Implementation

## 1. Modified File List

- `scripts/audit_stage18g_ldn_diff_k1_explainability.py`
- `docs/CTAF_STAGE18G_LDN_DIFF_K1_EXPLAINABILITY_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the explanation/stability audit for the frozen route:
  - `ldn_diff_K1_scorek5`
- It is strictly explanation-only:
  - no model load
  - no forward
  - no embedding extraction
  - no training
  - no score recomputation for new candidates
  - no selector
  - no oracle
  - no fusion
  - no hyperparameter tuning
- It supports:
  - `--mode check`
  - `--mode explain`
- `check` verifies required Stage18E/F artifacts, embedding npz files, score npz files, required columns, and `freeze_accepted == true`.
- `explain` reads existing results and produces:
  - machine-level deltas
  - score distribution shifts
  - normal-tail shifts
  - anomaly-separation shifts
  - domain-shift geometry summaries
  - local-density reference statistics
  - local-density nearest-neighbor diagnostics
  - explicit fan failure analysis
  - strong-machine guardrail explanation
  - paper-ready figure data tables
  - final explainability verdict JSON

## 3. Expected Outputs

Base output root:

- `results/stage18g_ldn_diff_k1_explainability/`

Expected files:

- `check_status.json`
- `explain_status.json`
- `machine_delta_summary.csv`
- `score_distribution_shift.csv`
- `normal_tail_shift.csv`
- `anomaly_separation_shift.csv`
- `domain_shift_geometry.csv`
- `local_density_reference_stats.csv`
- `local_density_nearest_neighbor_diagnostics.csv`
- `fan_failure_analysis.csv`
- `machine_contribution_to_gain.csv`
- `strong_machine_guardrail_explain.csv`
- `paper_figure_score_distribution_data.csv`
- `paper_figure_density_bias_data.csv`
- `stage18g_explainability_verdict.json`
- `docs/CTAF_STAGE18G_LDN_DIFF_K1_EXPLAINABILITY_AUDIT.md`

## 4. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18g_ldn_diff_k1_explainability.py --mode check --stage18e_root results/stage18e_local_density_protocol_full7 --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --embedding_root results/stage16d_minclamp_mean_full7/embeddings --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage18g_ldn_diff_k1_explainability --doc_path docs/CTAF_STAGE18G_LDN_DIFF_K1_EXPLAINABILITY_AUDIT.md
```

## 5. Explain Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18g_ldn_diff_k1_explainability.py --mode explain --stage18e_root results/stage18e_local_density_protocol_full7 --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --embedding_root results/stage16d_minclamp_mean_full7/embeddings --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage18g_ldn_diff_k1_explainability --doc_path docs/CTAF_STAGE18G_LDN_DIFF_K1_EXPLAINABILITY_AUDIT.md
```

## 6. Confirmation

- no experiment was run in Codex
- no score recomputation for new candidates was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
