# CTAF Stage18A ATST One-Machine KNN Implementation

## 1. Modified File List

- `scripts/audit_stage18a_atst_one_machine_knn.py`
- `docs/CTAF_STAGE18A_ATST_ONE_MACHINE_KNN_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the first frozen ATST screening scaffold.
- It targets the official checkpoint:
  - `/data1/test/pretrained_models/atst_clean/atst_as2M.ckpt`
- It supports three modes:
  - `--mode check`
  - `--mode one_wav_sanity`
  - `--mode one_machine_knn`
- It does not train, fine-tune, fuse, run LoRA, or open 7-machine audits.

## 3. Loader Assumption

- The checkpoint is loaded as a frozen ViT-style frame encoder from the stored `encoder.encoder.*` weights.
- The screening scaffold uses a checkpoint-based minimal backbone loader:
  - no class token
  - 12 transformer blocks
  - embedding dim `768`
  - patch projection from `64 x 4 = 256` inputs to `768`
  - final per-frame normalization via `norm_frame`
- Input preprocessing is fixed to:
  - 64-bin Kaldi fbank
  - frame shift `10 ms`
  - pad / truncate to `1004` frames

This is intended for frozen screening only.

## 4. Expected Outputs

- `results/stage18a_atst_one_machine_knn/check_status.json`
- `results/stage18a_atst_one_machine_knn/one_wav_sanity.json`
- `results/stage18a_atst_one_machine_knn/embedding_summary.csv`
- `results/stage18a_atst_one_machine_knn/embeddings/<machine>_atst_embeddings.npz`
- `results/stage18a_atst_one_machine_knn/scores/<machine>_atst_knn_scores.npz`
- `results/stage18a_atst_one_machine_knn/summary.csv`
- `results/stage18a_atst_one_machine_knn/tail_table.csv`
- `results/stage18a_atst_one_machine_knn/score_geometry.csv`
- `docs/CTAF_STAGE18A_ATST_ONE_MACHINE_KNN_AUDIT.md`

## 5. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18a_atst_one_machine_knn.py --mode check --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage18a_atst_one_machine_knn --doc_path docs/CTAF_STAGE18A_ATST_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 6. One-Wav Sanity Command

Replace `YOUR_WAV_PATH.wav` with an explicit file path.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18a_atst_one_machine_knn.py --mode one_wav_sanity --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --wav_path YOUR_WAV_PATH.wav --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage18a_atst_one_machine_knn --doc_path docs/CTAF_STAGE18A_ATST_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 7. One-Machine KNN Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18a_atst_one_machine_knn.py --mode one_machine_knn --atst_ckpt_path /data1/test/pretrained_models/atst_clean/atst_as2M.ckpt --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage18a_atst_one_machine_knn --doc_path docs/CTAF_STAGE18A_ATST_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 8. Confirmation

- no experiment was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
