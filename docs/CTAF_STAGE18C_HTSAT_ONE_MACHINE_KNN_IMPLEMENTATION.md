# CTAF Stage18C HTSAT One-Machine KNN Implementation

## 1. Modified File List

- `scripts/audit_stage18c_htsat_one_machine_knn.py`
- `docs/CTAF_STAGE18C_HTSAT_ONE_MACHINE_KNN_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements the first frozen HTSAT screening scaffold.
- It targets:
  - repo: `/data1/test/pretrained_models/htsat_clean/HTS-Audio-Transformer`
  - checkpoint: `/data1/test/pretrained_models/htsat_clean/HTSAT_AudioSet_Saved_1.ckpt`
- It supports three modes:
  - `--mode check`
  - `--mode one_wav_sanity`
  - `--mode one_machine_knn`
- It loads the official HTSAT backbone through the official repo code and strips the `sed_model.` checkpoint prefix.
- For screening:
  - use `latent_output` from HTSAT as utterance embedding
  - source train-normal only memory
  - cosine `kNN, k=5`
- It does not train, fine-tune, fuse, run LoRA, or open 7-machine audits.

## 3. Expected Outputs

- `results/stage18c_htsat_one_machine_knn/check_status.json`
- `results/stage18c_htsat_one_machine_knn/one_wav_sanity.json`
- `results/stage18c_htsat_one_machine_knn/embedding_summary.csv`
- `results/stage18c_htsat_one_machine_knn/embeddings/<machine>_htsat_embeddings.npz`
- `results/stage18c_htsat_one_machine_knn/scores/<machine>_htsat_knn_scores.npz`
- `results/stage18c_htsat_one_machine_knn/summary.csv`
- `results/stage18c_htsat_one_machine_knn/tail_table.csv`
- `results/stage18c_htsat_one_machine_knn/score_geometry.csv`
- `docs/CTAF_STAGE18C_HTSAT_ONE_MACHINE_KNN_AUDIT.md`

## 4. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage18c_htsat_one_machine_knn.py --mode check --htsat_repo_path /data1/test/pretrained_models/htsat_clean/HTS-Audio-Transformer --htsat_ckpt_path /data1/test/pretrained_models/htsat_clean/HTSAT_AudioSet_Saved_1.ckpt --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage18c_htsat_one_machine_knn --doc_path docs/CTAF_STAGE18C_HTSAT_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 5. One-Wav Sanity Command

Replace `YOUR_WAV_PATH.wav` with an explicit wav path.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18c_htsat_one_machine_knn.py --mode one_wav_sanity --htsat_repo_path /data1/test/pretrained_models/htsat_clean/HTS-Audio-Transformer --htsat_ckpt_path /data1/test/pretrained_models/htsat_clean/HTSAT_AudioSet_Saved_1.ckpt --wav_path YOUR_WAV_PATH.wav --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage18c_htsat_one_machine_knn --doc_path docs/CTAF_STAGE18C_HTSAT_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 6. One-Machine KNN Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage18c_htsat_one_machine_knn.py --mode one_machine_knn --htsat_repo_path /data1/test/pretrained_models/htsat_clean/HTS-Audio-Transformer --htsat_ckpt_path /data1/test/pretrained_models/htsat_clean/HTSAT_AudioSet_Saved_1.ckpt --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage18c_htsat_one_machine_knn --doc_path docs/CTAF_STAGE18C_HTSAT_ONE_MACHINE_KNN_AUDIT.md --latent_knn_k 5
```

## 7. Confirmation

- no experiment was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
