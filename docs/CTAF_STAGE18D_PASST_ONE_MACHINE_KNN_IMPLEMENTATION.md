# CTAF Stage18D PaSST One-Machine KNN Implementation

## 1. Modified File List

- `scripts/audit_stage18d_passt_one_machine_knn.py`
- `docs/CTAF_STAGE18D_PASST_ONE_MACHINE_KNN_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the first frozen PaSST screening scaffold.
- It targets:
  - checkpoint: `/data1/test/pretrained_models/passt_clean/passt-s-f128-p16-s10-ap.476-swa.pt`
- It supports three modes:
  - `--mode check`
  - `--mode one_wav_sanity`
  - `--mode one_machine_knn`
- It uses a checkpoint-based minimal PaSST ViT-style loader:
  - 128 mel bins
  - 32kHz audio
  - 16x16 patch projection
  - stride 10x10
  - 768-dim tokens
  - 12 transformer blocks
- For screening:
  - remove class/dist tokens
  - mean-pool patch tokens for utterance embedding
  - source train-normal only memory
  - cosine `kNN, k=5`
- It does not train, fine-tune, fuse, run LoRA, or open 7-machine audits.

## 3. Expected Outputs

- `results/stage18d_passt_one_machine_knn/check_status.json`
- `results/stage18d_passt_one_machine_knn/one_wav_sanity.json`
- `results/stage18d_passt_one_machine_knn/embedding_summary.csv`
- `results/stage18d_passt_one_machine_knn/embeddings/<machine>_passt_embeddings.npz`
- `results/stage18d_passt_one_machine_knn/scores/<machine>_passt_knn_scores.npz`
- `results/stage18d_passt_one_machine_knn/summary.csv`
- `results/stage18d_passt_one_machine_knn/tail_table.csv`
- `results/stage18d_passt_one_machine_knn/score_geometry.csv`
- `docs/CTAF_STAGE18D_PASST_ONE_MACHINE_KNN_AUDIT.md`

## 4. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18d_passt_one_machine_knn.py --mode check --passt_ckpt_path /data1/test/pretrained_models/passt_clean/passt-s-f128-p16-s10-ap.476-swa.pt --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage18d_passt_one_machine_knn --doc_path docs/CTAF_STAGE18D_PASST_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 5. One-Wav Sanity Command

Replace `YOUR_WAV_PATH.wav` with an explicit wav path.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18d_passt_one_machine_knn.py --mode one_wav_sanity --passt_ckpt_path /data1/test/pretrained_models/passt_clean/passt-s-f128-p16-s10-ap.476-swa.pt --wav_path YOUR_WAV_PATH.wav --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage18d_passt_one_machine_knn --doc_path docs/CTAF_STAGE18D_PASST_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 6. One-Machine KNN Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18d_passt_one_machine_knn.py --mode one_machine_knn --passt_ckpt_path /data1/test/pretrained_models/passt_clean/passt-s-f128-p16-s10-ap.476-swa.pt --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage18d_passt_one_machine_knn --doc_path docs/CTAF_STAGE18D_PASST_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 7. Confirmation

- no experiment was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
