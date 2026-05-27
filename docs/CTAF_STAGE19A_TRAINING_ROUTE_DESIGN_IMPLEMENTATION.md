# CTAF Stage19A Training Route Design Implementation

## 1. Modified File List

- `scripts/audit_stage19a_training_route_design.py`
- `docs/CTAF_STAGE19A_TRAINING_ROUTE_DESIGN_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the Stage19A protocol-design and data-audit stage.
- It is strictly planning and inventory only:
  - no training
  - no model load
  - no forward
  - no embedding extraction
  - no pseudo-label generation
  - no selectors
  - no oracles
  - no fusion
- It supports:
  - `--mode check`
  - `--mode inventory`
- `check` verifies path existence and required Stage18F/G artifacts.
- `inventory` scans data roots, infers machine / split / normal-anomaly / source-target / section / root id, and outputs:
  - full file inventory
  - machine summary
  - root summary
  - usable data table for domain-adaptive pretraining
  - excluded/high-risk data table
  - Stage18 anchor summary
  - protocol freeze JSON
  - training route plan markdown

## 3. Expected Outputs

Base output root:

- `results/stage19a_training_route_design/`

Expected files:

- `check_status.json`
- `data_inventory_filelist.csv`
- `data_inventory_summary_by_machine.csv`
- `data_inventory_summary_by_root.csv`
- `usable_data_for_dap.csv`
- `excluded_data_risk_table.csv`
- `stage18_anchor_summary.csv`
- `stage19_training_route_plan.md`
- `stage19_protocol_freeze.json`
- `docs/CTAF_STAGE19A_TRAINING_ROUTE_DESIGN_AUDIT.md`

## 4. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage19a_training_route_design.py --mode check --data_roots /data1/test/LAES_2025/data/dcase --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --stage18g_root results/stage18g_ldn_diff_k1_explainability --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage19a_training_route_design --doc_path docs/CTAF_STAGE19A_TRAINING_ROUTE_DESIGN_AUDIT.md
```

## 5. Inventory Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage19a_training_route_design.py --mode inventory --data_roots /data1/test/LAES_2025/data/dcase --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --stage18g_root results/stage18g_ldn_diff_k1_explainability --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --output_root results/stage19a_training_route_design --doc_path docs/CTAF_STAGE19A_TRAINING_ROUTE_DESIGN_AUDIT.md
```

## 6. Confirmation

- no experiment was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
