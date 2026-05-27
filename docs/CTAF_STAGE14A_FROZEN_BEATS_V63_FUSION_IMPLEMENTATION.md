# CTAF Stage14A Frozen BEATs + V63 Fusion Implementation

## 1. Modified File List

- `scripts/audit_stage14a_frozen_beats_v63_fusion.py`
- `docs/CTAF_STAGE14A_FROZEN_BEATS_V63_FUSION_IMPLEMENTATION.md`

## 2. Script Modes

The script supports three modes:

- `--mode check`
  - lightweight path / existence checks only
  - no model load
  - no embedding extraction
  - no metric computation

- `--mode a0_frozen_beats`
  - frozen BEATs utterance embedding extraction
  - source train-normal memory only
  - same current test set
  - no fine-tuning
  - no projection-head training

- `--mode a1_v63_beats_fusion`
  - fixed V63 + frozen BEATs fusion audit
  - fixed `0.5 * rank_norm(V63) + 0.5 * rank_norm(BEATs)`
  - no weight tuning
  - no per-machine selection
  - no oracle selection

## 3. Expected Inputs

- BEATs checkpoint:
  - `/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt`
- current data root:
  - `/data1/test/LAES_2025/data/dcase`
- V63 root:
  - `results/v63_fixed_v37_v61_rankfusion`
- V37 root:
  - `results/v37_teacher_student_pretrain_seedpanel`
- V61 root:
  - `results/v61_cross_year_original_encoder_seedpanel`
- seeds:
  - `13711,13721,13731`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`

## 4. Expected Outputs

Base output root:

- `results/stage14a_frozen_beats_v63_fusion/`

Mode-specific outputs:

### check

- `results/stage14a_frozen_beats_v63_fusion/check_status.json`

### a0_frozen_beats

- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/per_machine.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/tail_table.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/embedding_summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/scores/*_frozen_beats_knn_scores.npz`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/<machine>/<machine>_beats_embeddings.npz`

### a1_v63_beats_fusion

- `results/stage14a_frozen_beats_v63_fusion/summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/per_machine.csv`
- `results/stage14a_frozen_beats_v63_fusion/tail_table.csv`
- `results/stage14a_frozen_beats_v63_fusion/complementarity_summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/alignment_summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/complementarity_diagnostics.csv`
- `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_<seed>/scores/*_v63_beats_rankfusion_scores.npz`

Audit markdown:

- `docs/CTAF_STAGE14A_FROZEN_BEATS_V63_FUSION_AUDIT.md`

## 5. Server Command for mode=check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_frozen_beats_v63_fusion.py \
  --mode check \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_frozen_beats_v63_fusion \
  --doc_path docs/CTAF_STAGE14A_FROZEN_BEATS_V63_FUSION_AUDIT.md \
  --device cpu
```

## 6. Server Command for mode=a0_frozen_beats

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_frozen_beats_v63_fusion.py \
  --mode a0_frozen_beats \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_frozen_beats_v63_fusion \
  --doc_path docs/CTAF_STAGE14A_FROZEN_BEATS_V63_FUSION_AUDIT.md \
  --device cuda:0 \
  --latent_knn_k 5
```

## 7. Server Command for mode=a1_v63_beats_fusion

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_frozen_beats_v63_fusion.py \
  --mode a1_v63_beats_fusion \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_frozen_beats_v63_fusion \
  --doc_path docs/CTAF_STAGE14A_FROZEN_BEATS_V63_FUSION_AUDIT.md \
  --device cpu \
  --latent_knn_k 5
```

## 8. Expected Output Files

- `results/stage14a_frozen_beats_v63_fusion/check_status.json`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/per_machine.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/tail_table.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/embedding_summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats/scores/*_frozen_beats_knn_scores.npz`
- `results/stage14a_frozen_beats_v63_fusion/summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/per_machine.csv`
- `results/stage14a_frozen_beats_v63_fusion/tail_table.csv`
- `results/stage14a_frozen_beats_v63_fusion/complementarity_summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/alignment_summary.csv`
- `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/complementarity_diagnostics.csv`
- `results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_<seed>/scores/*_v63_beats_rankfusion_scores.npz`
- `docs/CTAF_STAGE14A_FROZEN_BEATS_V63_FUSION_AUDIT.md`

## 9. Confirmation That No Experiment Was Run in Codex

- no `check` mode was executed in Codex
- no `a0_frozen_beats` mode was executed in Codex
- no `a1_v63_beats_fusion` mode was executed in Codex

## 10. Confirmation That No Training/Adaptation Code Was Added

- no BEATs fine-tuning code
- no projection-head training code
- no LoRA / adapter code
- no adaptation training loop
- no model-training logic beyond frozen no-training audit flow
