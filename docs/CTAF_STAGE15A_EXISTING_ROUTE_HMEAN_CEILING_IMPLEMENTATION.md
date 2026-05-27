# CTAF Stage15A Existing Route Hmean Ceiling Implementation

## 1. Modified File List

- `scripts/audit_stage15a_existing_route_hmean_ceiling.py`
- `docs/CTAF_STAGE15A_EXISTING_ROUTE_HMEAN_CEILING_IMPLEMENTATION.md`

## 2. Script Modes

- `--mode check`
  - checks candidate roots
  - checks per-candidate score files can be found
  - checks seed/machine availability
  - does not compute oracle
  - does not compute metrics

- `--mode ceiling_audit`
  - collects existing candidate scores
  - performs alignment checks
  - computes candidate metrics
  - computes oracle ceiling analyses
  - writes summary CSVs and markdown interpretation

## 3. Candidate List

The script is designed to collect these candidates when available:

- `V37`
- `V61`
- `V63`
- `A0_raw_BEATs`
- `A1_alpha_0.50`
- `A3_alpha_0.20`
- `A3_alpha_0.35`
- `A3_alpha_0.65`
- `A3_alpha_0.80`
- `A4_B2s`
- `A4_B2f`
- `A4_F2s`
- `A4_F2f`

If any candidate is missing, it is recorded in `missing_candidates.csv`.

## 4. Expected Inputs

- V37 root:
  - `results/v37_teacher_student_pretrain_seedpanel`
- V61 root:
  - `results/v61_cross_year_original_encoder_seedpanel`
- V63 root:
  - `results/v63_fixed_v37_v61_rankfusion`
- Stage14A root:
  - `results/stage14a_frozen_beats_v63_fusion`
- A3 root:
  - `results/stage14a_a3_v63_beats_weight_sensitivity`
- A4 root:
  - `results/stage14a_a4_cross_year_frozen_beats_memory`
- current data root:
  - `/data1/test/LAES_2025/data/dcase`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seeds:
  - `13711,13721,13731`

## 5. Expected Outputs

Output root:

- `results/stage15a_existing_route_hmean_ceiling/`

Expected files:

- `check_status.json`
- `candidate_summary.csv`
- `per_machine_candidate_metrics.csv`
- `oracle_global_best.csv`
- `oracle_per_machine_best.csv`
- `oracle_constrained.csv`
- `oracle_no_diagnostic.csv`
- `seed_stability.csv`
- `missing_candidates.csv`
- `alignment_summary.csv`
- `candidate_metrics_raw.csv`
- `docs/CTAF_STAGE15A_EXISTING_ROUTE_HMEAN_CEILING_AUDIT.md`

## 6. Server Command for check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage15a_existing_route_hmean_ceiling.py \
  --mode check \
  --data_root /data1/test/LAES_2025/data/dcase \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --a3_root results/stage14a_a3_v63_beats_weight_sensitivity \
  --a4_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage15a_existing_route_hmean_ceiling \
  --doc_path docs/CTAF_STAGE15A_EXISTING_ROUTE_HMEAN_CEILING_AUDIT.md
```

## 7. Server Command for ceiling_audit

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage15a_existing_route_hmean_ceiling.py \
  --mode ceiling_audit \
  --data_root /data1/test/LAES_2025/data/dcase \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --a3_root results/stage14a_a3_v63_beats_weight_sensitivity \
  --a4_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage15a_existing_route_hmean_ceiling \
  --doc_path docs/CTAF_STAGE15A_EXISTING_ROUTE_HMEAN_CEILING_AUDIT.md
```

## 8. Confirmation No Experiment Was Run in Codex

- no `check` mode was executed in Codex
- no `ceiling_audit` mode was executed in Codex

## 9. Confirmation No Training/Adaptation Code Was Added

- no training loop
- no new model
- no BEATs fine-tuning
- no adaptation head
- no selector/router training
