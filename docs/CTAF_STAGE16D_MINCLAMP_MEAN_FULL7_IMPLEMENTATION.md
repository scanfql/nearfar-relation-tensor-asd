# CTAF Stage16D MinClamp Mean Full7 Implementation

## 1. Modified File List

- `scripts/audit_stage16d_minclamp_mean_full7.py`
- `docs/CTAF_STAGE16D_MINCLAMP_MEAN_FULL7_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the fixed full-7 audit for the Stage16D candidate `min_clamp__mean`.
- It is strictly frozen clean `EAT` only:
  - no training
  - no fine-tuning
  - no LoRA
  - no adapter
  - no Conformer
  - no fusion
  - no alpha sweep
  - no per-machine selector
  - no oracle
- It supports three modes:
  - `--mode check`
  - `--mode extract_minclamp_mean`
  - `--mode score_minclamp_mean`
- `extract_minclamp_mean` is the only mode that loads clean EAT and runs frozen forward.
- `score_minclamp_mean` loads saved embeddings only and computes cosine `kNN, k=5`.

## 3. Fixed Candidate Definition

- preprocessing:
  - `min_clamp`
  - `x = where(abs(x) < 0.1, 0, x)`
- pooling:
  - temporal `mean` over clean EAT frame output `[T, 768]`
- candidate name:
  - `min_clamp__mean`

## 4. Fixed Scoring Protocol

- frozen clean EAT only
- all 7 machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- source train-normal memory only
- no target train-normal
- cosine `kNN`
- `k = 5`
- metrics per machine:
  - `hmean`
  - `pAUC`
  - `KS`
  - `TPR@95n`
  - `TPR@99n`
  - `q95`
  - `q99`

## 5. Expected Outputs

Base root:

- `results/stage16d_minclamp_mean_full7/`

Files:

- `check_status.json`
- `machine_check.csv`
- `embedding_summary.csv`
- `embeddings/<machine>_minclamp_mean_embeddings.npz`
- `scores/<machine>_minclamp_mean_scores.npz`
- `summary.csv`
- `per_machine.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `comparison_vs_p1.csv`
- `collapse_check.csv`
- `stage16d_full7_verdict.json`
- `missing_comparison_files.csv`
- `docs/CTAF_STAGE16D_MINCLAMP_MEAN_FULL7_AUDIT.md`

## 6. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16d_minclamp_mean_full7.py --mode check --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cpu --output_root results/stage16d_minclamp_mean_full7 --doc_path docs/CTAF_STAGE16D_MINCLAMP_MEAN_FULL7_AUDIT.md --latent_knn_k 5 --stage16b_p1_per_machine_path results/stage16b_clean_eat_full_frozen_knn/per_machine.csv
```

## 7. Extract MinClamp Mean Command

Use second GPU via `CUDA_VISIBLE_DEVICES=1` and script arg `--device cuda:0`.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage16d_minclamp_mean_full7.py --mode extract_minclamp_mean --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cuda:0 --output_root results/stage16d_minclamp_mean_full7 --doc_path docs/CTAF_STAGE16D_MINCLAMP_MEAN_FULL7_AUDIT.md --latent_knn_k 5 --stage16b_p1_per_machine_path results/stage16b_clean_eat_full_frozen_knn/per_machine.csv
```

## 8. Score MinClamp Mean Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16d_minclamp_mean_full7.py --mode score_minclamp_mean --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cpu --output_root results/stage16d_minclamp_mean_full7 --doc_path docs/CTAF_STAGE16D_MINCLAMP_MEAN_FULL7_AUDIT.md --latent_knn_k 5 --stage16b_p1_per_machine_path results/stage16b_clean_eat_full_frozen_knn/per_machine.csv
```

## 9. Confirmation

- no experiment was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
- no scoring was run in Codex
