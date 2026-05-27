# CTAF Stage18A-C ATST MinClamp 050 Full7 Implementation

## 1. Modified File List

- `scripts/audit_stage18ac_atst_minclamp_050_full7.py`
- `docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the full7 audit for:
  - ATST standalone full7
  - fixed fusion candidate `F_ATST_minclamp_050`
- It supports:
  - `--mode check`
  - `--mode extract_atst_embeddings`
  - `--mode score_atst`
  - `--mode fixed_fusion`
- It is frozen-only and score-level only beyond extraction:
  - no training
  - no fine-tuning
  - no LoRA
  - no adapter
  - no Conformer
  - no alpha sweep
  - no per-machine selector
  - no oracle

## 3. Fixed Candidate

- `F_ATST_minclamp_050 = 0.5 * rank_norm(ATST_knn) + 0.5 * rank_norm(min_clamp__mean)`

No other fusion candidates are added.

## 4. Expected Outputs

- `results/stage18ac_atst_minclamp_050_full7/check_status.json`
- `results/stage18ac_atst_minclamp_050_full7/machine_check.csv`
- `results/stage18ac_atst_minclamp_050_full7/embedding_summary.csv`
- `results/stage18ac_atst_minclamp_050_full7/embeddings/<machine>_atst_embeddings.npz`
- `results/stage18ac_atst_minclamp_050_full7/scores/<machine>_atst_knn_scores.npz`
- `results/stage18ac_atst_minclamp_050_full7/atst_summary.csv`
- `results/stage18ac_atst_minclamp_050_full7/atst_per_machine.csv`
- `results/stage18ac_atst_minclamp_050_full7/atst_tail_table.csv`
- `results/stage18ac_atst_minclamp_050_full7/atst_score_geometry.csv`
- `results/stage18ac_atst_minclamp_050_full7/fusion_scores/<machine>/<machine>_scores.npz`
- `results/stage18ac_atst_minclamp_050_full7/fusion_summary.csv`
- `results/stage18ac_atst_minclamp_050_full7/fusion_per_machine.csv`
- `results/stage18ac_atst_minclamp_050_full7/fusion_tail_table.csv`
- `results/stage18ac_atst_minclamp_050_full7/fusion_score_geometry.csv`
- `results/stage18ac_atst_minclamp_050_full7/fusion_comparison.csv`
- `results/stage18ac_atst_minclamp_050_full7/stage18ac_verdict.json`
- `docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_AUDIT.md`

## 5. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18ac_atst_minclamp_050_full7.py --mode check --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --data_root /data1/test/LAES_2025/data/dcase --minclamp_root results/stage16d_minclamp_mean_full7 --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cpu --output_root results/stage18ac_atst_minclamp_050_full7 --doc_path docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_AUDIT.md --latent_knn_k 5
```

## 6. Extract ATST Embeddings Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18ac_atst_minclamp_050_full7.py --mode extract_atst_embeddings --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --data_root /data1/test/LAES_2025/data/dcase --minclamp_root results/stage16d_minclamp_mean_full7 --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cuda:0 --output_root results/stage18ac_atst_minclamp_050_full7 --doc_path docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_AUDIT.md --latent_knn_k 5
```

## 7. Score ATST Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18ac_atst_minclamp_050_full7.py --mode score_atst --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --data_root /data1/test/LAES_2025/data/dcase --minclamp_root results/stage16d_minclamp_mean_full7 --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cpu --output_root results/stage18ac_atst_minclamp_050_full7 --doc_path docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_AUDIT.md --latent_knn_k 5
```

## 8. Fixed Fusion Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18ac_atst_minclamp_050_full7.py --mode fixed_fusion --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --data_root /data1/test/LAES_2025/data/dcase --minclamp_root results/stage16d_minclamp_mean_full7 --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --seed 13711 --device cpu --output_root results/stage18ac_atst_minclamp_050_full7 --doc_path docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_AUDIT.md --latent_knn_k 5
```

## 9. Confirmation

- no experiment was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
- no scoring was run in Codex
