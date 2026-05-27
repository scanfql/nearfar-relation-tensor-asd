# CTAF Stage14A A4 Cross-Year Frozen BEATs Memory Implementation

## 1. Modified File List

- `docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_PREREG.md`
- `scripts/audit_stage14a_a4_cross_year_frozen_beats_memory.py`
- `docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_IMPLEMENTATION.md`

## 2. Script Modes

The script supports four modes:

- `--mode check`
  - check BEATs checkpoint exists
  - check V61-clean manifest exists
  - check V63 root exists
  - check A0 root exists
  - check `data_root` exists
  - report source-only count and full-manifest count per machine/year
  - no model load
  - no extraction
  - no scoring

- `--mode extract_cross_year_beats`
  - load frozen BEATs checkpoint
  - `eval()` + `torch.no_grad()`
  - extract embeddings for:
    - source-only memory rows
    - full-manifest memory rows
  - reuse existing A0 current-test embeddings if present
  - otherwise extract current-test embeddings
  - save separate embedding caches

- `--mode score_cross_year_beats`
  - compute `B2s` using source-only memory
  - compute `B2f` using full-manifest memory
  - `k = 5`
  - save separate score NPZs and metrics

- `--mode fuse_v63_cross_year_beats`
  - load `V63`
  - load `B2s`
  - load `B2f`
  - alignment check
  - compute:
    - `F2s = 0.5 * rank_norm(V63) + 0.5 * rank_norm(B2s)`
    - `F2f = 0.5 * rank_norm(V63) + 0.5 * rank_norm(B2f)`
  - `F2s` is primary
  - `F2f` is diagnostic only

## 3. Expected Inputs

- BEATs checkpoint:
  - `/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt`
- current data root:
  - `/data1/test/LAES_2025/data/dcase`
- V61-clean manifest:
  - `results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv`
- V63 root:
  - `results/v63_fixed_v37_v61_rankfusion`
- A0 root:
  - `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats`
- V37 root:
  - `results/v37_teacher_student_pretrain_seedpanel`
- V61 root:
  - `results/v61_cross_year_original_encoder_seedpanel`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seeds:
  - `13711,13721,13731`

## 4. Expected Outputs

Output root:

- `results/stage14a_a4_cross_year_frozen_beats_memory/`

Expected files:

- `check_status.json`
- `check_manifest_counts.csv`
- `embedding_summary.csv`
- `extract_cross_year_beats/<machine>/<machine>_b2_source_memory_embeddings.npz`
- `extract_cross_year_beats/<machine>/<machine>_b2_full_memory_embeddings.npz`
- `extract_cross_year_beats/<machine>/<machine>_current_test_embeddings.npz`
- `b2_cross_year_beats/summary.csv`
- `b2_cross_year_beats/per_machine.csv`
- `b2_cross_year_beats/tail_table.csv`
- `b2_cross_year_beats/embedding_summary.csv`
- `b2_cross_year_beats/scores/*_b2_source_beats_scores.npz`
- `b2_cross_year_beats/scores/*_b2_full_beats_scores.npz`
- `f2_v63_cross_year_beats_fusion/summary.csv`
- `f2_v63_cross_year_beats_fusion/per_machine.csv`
- `f2_v63_cross_year_beats_fusion/tail_table.csv`
- `f2_v63_cross_year_beats_fusion/alignment_summary.csv`
- `f2_v63_cross_year_beats_fusion/complementarity_summary.csv`
- `f2_v63_cross_year_beats_fusion/complementarity_diagnostics.csv`
- `f2_v63_cross_year_beats_fusion/seed_<seed>/scores/*.npz`
- `docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_AUDIT.md`

## 5. Server Command for check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a4_cross_year_frozen_beats_memory.py \
  --mode check \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --doc_path docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_AUDIT.md \
  --device cpu \
  --latent_knn_k 5
```

## 6. Server Command for extract_cross_year_beats

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a4_cross_year_frozen_beats_memory.py \
  --mode extract_cross_year_beats \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --doc_path docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_AUDIT.md \
  --device cuda:0 \
  --latent_knn_k 5
```

## 7. Server Command for score_cross_year_beats

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a4_cross_year_frozen_beats_memory.py \
  --mode score_cross_year_beats \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --doc_path docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_AUDIT.md \
  --device cpu \
  --latent_knn_k 5
```

## 8. Server Command for fuse_v63_cross_year_beats

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a4_cross_year_frozen_beats_memory.py \
  --mode fuse_v63_cross_year_beats \
  --data_root /data1/test/LAES_2025/data/dcase \
  --beats_checkpoint /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt \
  --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a4_cross_year_frozen_beats_memory \
  --doc_path docs/CTAF_STAGE14A_A4_CROSS_YEAR_FROZEN_BEATS_MEMORY_AUDIT.md \
  --device cpu \
  --latent_knn_k 5
```

## 9. Confirmation No Experiment Was Run in Codex

- no `check` mode was executed in Codex
- no `extract_cross_year_beats` mode was executed in Codex
- no `score_cross_year_beats` mode was executed in Codex
- no `fuse_v63_cross_year_beats` mode was executed in Codex

## 10. Confirmation No Training/Adaptation Code Was Added

- no training loop
- no BEATs fine-tuning
- no PCA / whitening
- no projection head
- no LoRA / adapter
- no fusion weight tuning
- only frozen-memory expansion logic was added
