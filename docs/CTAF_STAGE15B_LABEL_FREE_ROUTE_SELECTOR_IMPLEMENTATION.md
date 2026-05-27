# CTAF Stage15B Label-Free Route Selector Implementation

## 1. Modified File List

- `scripts/audit_stage15b_label_free_route_selector.py`
- `docs/CTAF_STAGE15B_LABEL_FREE_ROUTE_SELECTOR_IMPLEMENTATION.md`

## 2. Implementation Summary

The script implements a no-training feasibility audit for label-free
machine-level route selection.

Supported modes:

- `--mode feature_audit`
  - computes unlabeled route features
  - does not run selectors

- `--mode selector_audit`
  - evaluates fixed label-free selector rules:
    - `S0`
    - `S1`
    - `S2`
    - `S3`
    - `S4`
    - `S5`

No training, no new model, no supervised selector, and no anomaly/test-label
route choice are introduced.

## 3. feature_audit command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage15b_label_free_route_selector.py \
  --mode feature_audit \
  --data_root /data1/test/LAES_2025/data/dcase \
  --stage15a_root results/stage15a_existing_route_hmean_ceiling \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --a3_root results/stage14a_a3_v63_beats_weight_sensitivity \
  --a4_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage15b_label_free_route_selector \
  --doc_path docs/CTAF_STAGE15B_LABEL_FREE_ROUTE_SELECTOR_AUDIT.md
```

## 4. selector_audit command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage15b_label_free_route_selector.py \
  --mode selector_audit \
  --data_root /data1/test/LAES_2025/data/dcase \
  --stage15a_root results/stage15a_existing_route_hmean_ceiling \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --a3_root results/stage14a_a3_v63_beats_weight_sensitivity \
  --a4_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage15b_label_free_route_selector \
  --doc_path docs/CTAF_STAGE15B_LABEL_FREE_ROUTE_SELECTOR_AUDIT.md
```

## 5. Expected Output Files

- `results/stage15b_label_free_route_selector/feature_table.csv`
- `results/stage15b_label_free_route_selector/selector_results.csv`
- `results/stage15b_label_free_route_selector/selector_per_machine.csv`
- `results/stage15b_label_free_route_selector/selector_vs_oracle_gap.csv`
- `results/stage15b_label_free_route_selector/selected_routes_by_machine.csv`
- `results/stage15b_label_free_route_selector/selector_seed_stability.csv`
- `results/stage15b_label_free_route_selector/missing_candidates.csv`
- `results/stage15b_label_free_route_selector/alignment_summary.csv`
- `docs/CTAF_STAGE15B_LABEL_FREE_ROUTE_SELECTOR_AUDIT.md`

## 6. Confirmation No Training Occurred

- no training loop
- no new model
- no BEATs fine-tuning
- no supervised selector
- no selector/router training
