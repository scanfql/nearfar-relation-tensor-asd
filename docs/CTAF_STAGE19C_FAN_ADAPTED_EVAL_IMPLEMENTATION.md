# CTAF Stage19C Fan Adapted Eval Implementation

## 1. Modified File List

- `scripts/audit_stage19c_fan_adapted_eval.py`
- `docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements fan-only evaluation of the Stage19B adapted checkpoint.
- It supports:
  - `--mode check`
  - `--mode inspect_checkpoint`
  - `--mode extract_adapted_embeddings`
  - `--mode score_adapted_fan`
- It loads:
  - clean EAT checkpoint
  - adapted checkpoint from Stage19B
- It extracts adapted fan embeddings using:
  - same 16kHz / 128-bin Kaldi fbank / 1024-frame normalization as clean EAT
  - same `min_clamp`
  - same `mean` pooling
- It evaluates exactly two fixed score candidates:
  - `adapted_baseline_knn_k5`
  - `adapted_ldn_diff_K1_scorek5`
- It does not do full7, no fusion, no selector, no oracle, and no test-label checkpoint selection.

## 3. Expected Outputs

- `results/stage19c_fan_adapted_eval/check_status.json`
- `results/stage19c_fan_adapted_eval/checkpoint_status.json`
- `results/stage19c_fan_adapted_eval/adapted_parameter_audit.csv`
- `results/stage19c_fan_adapted_eval/trainable_boundary_recap.json`
- `results/stage19c_fan_adapted_eval/embedding_summary.csv`
- `results/stage19c_fan_adapted_eval/embeddings/fan_adapted_minclamp_mean_embeddings.npz`
- `results/stage19c_fan_adapted_eval/scores/adapted_baseline_knn_k5/fan_scores.npz`
- `results/stage19c_fan_adapted_eval/scores/adapted_ldn_diff_K1_scorek5/fan_scores.npz`
- `results/stage19c_fan_adapted_eval/summary.csv`
- `results/stage19c_fan_adapted_eval/tail_table.csv`
- `results/stage19c_fan_adapted_eval/score_geometry.csv`
- `results/stage19c_fan_adapted_eval/comparison_vs_stage18_fan.csv`
- `results/stage19c_fan_adapted_eval/stage19c_fan_adapted_eval_verdict.json`
- `docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_AUDIT.md`

## 4. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage19c_fan_adapted_eval.py --mode check --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --adapted_ckpt_path results/stage19b_fan_dap_ssl_sanity/checkpoints/fan_stage19b_best.pt --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cpu --output_root results/stage19c_fan_adapted_eval --doc_path docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_AUDIT.md --stage19b_root results/stage19b_fan_dap_ssl_sanity --stage18e_root results/stage18e_local_density_protocol_full7 --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --latent_knn_k 5
```

## 5. Inspect Checkpoint Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage19c_fan_adapted_eval.py --mode inspect_checkpoint --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --adapted_ckpt_path results/stage19b_fan_dap_ssl_sanity/checkpoints/fan_stage19b_best.pt --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cuda:0 --output_root results/stage19c_fan_adapted_eval --doc_path docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_AUDIT.md --stage19b_root results/stage19b_fan_dap_ssl_sanity --stage18e_root results/stage18e_local_density_protocol_full7 --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --latent_knn_k 5
```

## 6. Extract Adapted Embeddings Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage19c_fan_adapted_eval.py --mode extract_adapted_embeddings --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --adapted_ckpt_path results/stage19b_fan_dap_ssl_sanity/checkpoints/fan_stage19b_best.pt --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cuda:0 --output_root results/stage19c_fan_adapted_eval --doc_path docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_AUDIT.md --stage19b_root results/stage19b_fan_dap_ssl_sanity --stage18e_root results/stage18e_local_density_protocol_full7 --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --latent_knn_k 5
```

## 7. Score Adapted Fan Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage19c_fan_adapted_eval.py --mode score_adapted_fan --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --adapted_ckpt_path results/stage19b_fan_dap_ssl_sanity/checkpoints/fan_stage19b_best.pt --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cpu --output_root results/stage19c_fan_adapted_eval --doc_path docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_AUDIT.md --stage19b_root results/stage19b_fan_dap_ssl_sanity --stage18e_root results/stage18e_local_density_protocol_full7 --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --latent_knn_k 5
```

## 8. Confirmation

- no experiment was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
- no scoring was run in Codex
