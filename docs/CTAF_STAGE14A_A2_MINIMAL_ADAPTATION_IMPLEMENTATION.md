# CTAF Stage14A A2 Minimal Adaptation Implementation

## 1. Modified File List

- `docs/CTAF_STAGE14A_A2_MINIMAL_ADAPTATION_PREREG.md`
- `scripts/audit_stage14a_a2_beats_minimal_adaptation.py`
- `docs/CTAF_STAGE14A_A2_MINIMAL_ADAPTATION_IMPLEMENTATION.md`

## 2. Script Modes

The script supports three modes:

- `--mode check`
  - verify required paths and A0 embedding files exist
  - no model load
  - no PCA fitting
  - no metric computation

- `--mode a2_pca_beats`
  - load existing A0 frozen BEATs embeddings
  - fit PCA/whitening on source train-normal embeddings only
  - fixed dimension = `256`
  - compute B1 score files and metrics

- `--mode a2_v63_pca_beats_fusion`
  - load V63 score files
  - load B1 PCA-BEATs score files
  - alignment check
  - fixed rank fusion:
    - `F1 = 0.5 * rank_norm(V63) + 0.5 * rank_norm(B1)`
  - output fused route metrics and diagnostics

## 3. Expected Inputs

- data root:
  - `/data1/test/LAES_2025/data/dcase`
- A0 root:
  - `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats`
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
- fixed PCA dimension:
  - `256`

## 4. Expected Outputs

Base output root:

- `results/stage14a_a2_beats_minimal_adaptation/`

Mode-specific outputs:

### check

- `results/stage14a_a2_beats_minimal_adaptation/check_status.json`

### a2_pca_beats

- `results/stage14a_a2_beats_minimal_adaptation/a2_pca_beats/summary.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_pca_beats/per_machine.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_pca_beats/tail_table.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_pca_beats/pca_summary.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_pca_beats/scores/*_pca_beats_knn_scores.npz`

### a2_v63_pca_beats_fusion

- `results/stage14a_a2_beats_minimal_adaptation/summary.csv`
- `results/stage14a_a2_beats_minimal_adaptation/per_machine.csv`
- `results/stage14a_a2_beats_minimal_adaptation/complementarity_summary.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_v63_pca_beats_fusion/alignment_summary.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_v63_pca_beats_fusion/seed_metrics.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_v63_pca_beats_fusion/complementarity_diagnostics.csv`
- `results/stage14a_a2_beats_minimal_adaptation/a2_v63_pca_beats_fusion/seed_<seed>/scores/*_v63_pca_beats_rankfusion_scores.npz`

Audit markdown:

- `docs/CTAF_STAGE14A_A2_MINIMAL_ADAPTATION_AUDIT.md`

## 5. Server Command for mode=check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a2_beats_minimal_adaptation.py \
  --mode check \
  --data_root /data1/test/LAES_2025/data/dcase \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a2_beats_minimal_adaptation \
  --doc_path docs/CTAF_STAGE14A_A2_MINIMAL_ADAPTATION_AUDIT.md \
  --latent_knn_k 5 \
  --pca_dim 256
```

## 6. Server Command for mode=a2_pca_beats

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a2_beats_minimal_adaptation.py \
  --mode a2_pca_beats \
  --data_root /data1/test/LAES_2025/data/dcase \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a2_beats_minimal_adaptation \
  --doc_path docs/CTAF_STAGE14A_A2_MINIMAL_ADAPTATION_AUDIT.md \
  --latent_knn_k 5 \
  --pca_dim 256
```

## 7. Server Command for mode=a2_v63_pca_beats_fusion

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage14a_a2_beats_minimal_adaptation.py \
  --mode a2_v63_pca_beats_fusion \
  --data_root /data1/test/LAES_2025/data/dcase \
  --a0_root results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --v37_root results/v37_teacher_student_pretrain_seedpanel \
  --v61_root results/v61_cross_year_original_encoder_seedpanel \
  --seeds 13711,13721,13731 \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --output_root results/stage14a_a2_beats_minimal_adaptation \
  --doc_path docs/CTAF_STAGE14A_A2_MINIMAL_ADAPTATION_AUDIT.md \
  --latent_knn_k 5 \
  --pca_dim 256
```

## 8. Confirmation That No Experiment Was Run in Codex

- no `check` mode was executed in Codex
- no `a2_pca_beats` mode was executed in Codex
- no `a2_v63_pca_beats_fusion` mode was executed in Codex

## 9. Confirmation That No Training/Adaptation Beyond PCA/Whitening Was Added

- no BEATs fine-tuning
- no projection-head training
- no LoRA / adapter
- no MLP adaptation head
- no contrastive training
- no supervised labels
- only frozen embedding PCA/whitening support was added
