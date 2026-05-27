# CTAF Stage19B Fan DAP SSL Sanity Implementation

## 1. Modified File List

- `scripts/audit_stage19b_fan_dap_ssl_sanity.py`
- `docs/CTAF_STAGE19B_FAN_DAP_SSL_SANITY_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the Stage19B conservative one-machine training scaffold for `fan`.
- It is a scaffold only:
  - `check`
  - `inspect_trainable`
  - `train_fan_ssl`
- `check` verifies fan safe train-normal files, clean EAT checkpoint, imports, and Stage18F/G anchors.
- `inspect_trainable` loads clean EAT, freezes all parameters, enables only the last `N` transformer blocks plus a small reconstruction head, and writes parameter audits.
- `train_fan_ssl` implements:
  - safe `90/10` train/val split on fan source train-normal only
  - masked acoustic reconstruction on normalized 128-bin fbank
  - masked positions only MSE
  - optional anchor loss to frozen EAT pooled embedding, disabled by default
  - best-checkpoint selection by validation reconstruction loss
- This stage does not evaluate ASD scoring. Stage19C should do the backend evaluation.

## 3. Expected Outputs

- `results/stage19b_fan_dap_ssl_sanity/check_status.json`
- `results/stage19b_fan_dap_ssl_sanity/fan_split_inventory.csv`
- `results/stage19b_fan_dap_ssl_sanity/model_load_status.json`
- `results/stage19b_fan_dap_ssl_sanity/trainable_parameter_audit.csv`
- `results/stage19b_fan_dap_ssl_sanity/frozen_parameter_audit.csv`
- `results/stage19b_fan_dap_ssl_sanity/trainable_summary.json`
- `results/stage19b_fan_dap_ssl_sanity/train_log.csv`
- `results/stage19b_fan_dap_ssl_sanity/checkpoints/fan_stage19b_best.pt`
- `results/stage19b_fan_dap_ssl_sanity/stage19b_train_config.json`
- `results/stage19b_fan_dap_ssl_sanity/stage19b_training_verdict.json`
- `docs/CTAF_STAGE19B_FAN_DAP_SSL_SANITY_AUDIT.md`

## 4. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage19b_fan_dap_ssl_sanity.py --mode check --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cpu --output_root results/stage19b_fan_dap_ssl_sanity --doc_path docs/CTAF_STAGE19B_FAN_DAP_SSL_SANITY_AUDIT.md --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --stage18g_root results/stage18g_ldn_diff_k1_explainability --trainable_last_blocks 1 --mask_ratio 0.30 --batch_size 8 --epochs 5 --lr 1e-5 --weight_decay 1e-4 --grad_clip 1.0 --early_stop_patience 2 --anchor_loss_weight 0.0
```

## 5. Inspect Trainable Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage19b_fan_dap_ssl_sanity.py --mode inspect_trainable --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cuda:0 --output_root results/stage19b_fan_dap_ssl_sanity --doc_path docs/CTAF_STAGE19B_FAN_DAP_SSL_SANITY_AUDIT.md --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --stage18g_root results/stage18g_ldn_diff_k1_explainability --trainable_last_blocks 1 --mask_ratio 0.30 --batch_size 8 --epochs 5 --lr 1e-5 --weight_decay 1e-4 --grad_clip 1.0 --early_stop_patience 2 --anchor_loss_weight 0.0
```

## 6. Train Fan SSL Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage19b_fan_dap_ssl_sanity.py --mode train_fan_ssl --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine fan --seed 13711 --device cuda:0 --output_root results/stage19b_fan_dap_ssl_sanity --doc_path docs/CTAF_STAGE19B_FAN_DAP_SSL_SANITY_AUDIT.md --stage18f_root results/stage18f_ldn_diff_k1_route_freeze --stage18g_root results/stage18g_ldn_diff_k1_explainability --trainable_last_blocks 1 --mask_ratio 0.30 --batch_size 8 --epochs 5 --lr 1e-5 --weight_decay 1e-4 --grad_clip 1.0 --early_stop_patience 2 --anchor_loss_weight 0.0
```

## 7. Confirmation

- no experiment was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
