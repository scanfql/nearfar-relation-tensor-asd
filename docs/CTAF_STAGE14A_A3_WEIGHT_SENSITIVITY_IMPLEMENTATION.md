# CTAF Stage14A A3 Weight Sensitivity Implementation

## 1. Modified File List

- `scripts/audit_stage14a_a3_v63_beats_weight_sensitivity.py`
- `docs/CTAF_STAGE14A_A3_WEIGHT_SENSITIVITY_IMPLEMENTATION.md`

## 2. Script Modes

The script supports:

- `--mode check`
  - verifies required roots exist
  - verifies V63 per-seed score files exist
  - verifies A0 raw BEATs score files exist
  - verifies V37/V61 roots exist
  - no metrics
  - no fusion

- `--mode a3_weight_sensitivity`
  - loads V63 raw rank-fusion score
  - loads A0 raw BEATs score
  - alignment check
  - computes fixed alpha sensitivity for:
    - `alpha ∈ {0.2, 0.35, 0.5, 0.65, 0.8}`
  - saves fused NPZs and summary tables

## 3. Expected Inputs

- V63 score root:
  - `results/v63_fixed_v37_v61_rankfusion`
- A0 raw BEATs score root:
  - `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats`
- V37 root:
  - `results/v37_teacher_student_pretrain_seedpanel`
- V61 root:
  - `results/v61_cross_year_original_encoder_seedpanel`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seeds:
  - `13711,13721,13731`

## 4. Expected Outputs

Output root:

- `results/stage14a_a3_v63_beats_weight_sensitivity/`

Files:

- `check_status.json`
- `summary.csv`
- `per_machine.csv`
- `tail_table.csv`
- `alpha_sensitivity.csv`
- `complementarity_summary.csv`
- `a3_weight_sensitivity/alignment_summary.csv`
- `a3_weight_sensitivity/seed_metrics.csv`
- `a3_weight_sensitivity/complementarity_diagnostics.csv`
- `a3_weight_sensitivity/alpha_<alpha>/seed_<seed>/scores/*.npz`
- `docs/CTAF_STAGE14A_A3_WEIGHT_SENSITIVITY_AUDIT.md`

## 5. Server Command for mode=check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a3_v63_beats_weight_sensitivity.py \
  --mode check \
  --data_root /data1/test/LAES_2025/data/dcase \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a3_v63_beats_weight_sensitivity \
  --doc_path docs/CTAF_STAGE14A_A3_WEIGHT_SENSITIVITY_AUDIT.md
```

## 6. Server Command for mode=a3_weight_sensitivity

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a3_v63_beats_weight_sensitivity.py \
  --mode a3_weight_sensitivity \
  --data_root /data1/test/LAES_2025/data/dcase \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a3_v63_beats_weight_sensitivity \
  --doc_path docs/CTAF_STAGE14A_A3_WEIGHT_SENSITIVITY_AUDIT.md
```

## 7. Confirmation No Experiment Was Run in Codex

- no `check` mode was executed in Codex
- no `a3_weight_sensitivity` mode was executed in Codex

## 8. Confirmation No Training/Adaptation Code Was Added

- no training loop
- no BEATs fine-tuning
- no projection-head training
- no adaptation code
- no LoRA / adapter
- no per-machine alpha selection
