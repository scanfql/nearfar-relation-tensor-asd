# CTAF Stage16C2 Clean EAT Segment-Chunk One-Machine Implementation

## 1. Modified File List

- `scripts/audit_stage16c2_clean_eat_segment_chunk_one_machine.py`
- `docs/CTAF_STAGE16C2_CLEAN_EAT_SEGMENT_CHUNK_ONE_MACHINE_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements Stage16C-2 one-machine sanity for frozen clean `EAT` local chunk scoring on `gearbox`.
- It is strictly frozen-EAT only:
  - no training
  - no fine-tuning
  - no LoRA
  - no adapter
  - no projection head
  - no A1/V63 fusion
  - no alpha sweep
  - no per-machine route
  - no oracle
- The script is split into:
  - `--mode check`
  - `--mode extract_chunk_cache`
  - `--mode score_chunk`
- `extract_chunk_cache` is the only mode that loads clean EAT and runs frozen forward.
- `score_chunk` uses saved chunk caches only and does not load the model.

## 3. Fixed Chunk Configs

- `C08S04`
  - `chunk_len = 8`
  - `chunk_stride = 4`
- `C16S08`
  - `chunk_len = 16`
  - `chunk_stride = 8`
- `C32S16`
  - `chunk_len = 32`
  - `chunk_stride = 16`

Chunk embedding construction:

- input frame output:
  - `[T, 768]`
- per chunk:
  - `concat(mean_T(chunk_frames), std_T(chunk_frames))`
- chunk embedding dim:
  - `1536`
- if `T < chunk_len`:
  - use one chunk over all available frames

## 4. Fixed Aggregation Rules

For each chunk config, per-file chunk scores are aggregated with:

- `top1`
  - max chunk score
- `top5mean`
  - mean of highest `5%` chunk scores
  - at least `1` chunk
- `top10mean`
  - mean of highest `10%` chunk scores
  - at least `1` chunk

Candidate names:

- `C08S04_top1`
- `C08S04_top5mean`
- `C08S04_top10mean`
- `C16S08_top1`
- `C16S08_top5mean`
- `C16S08_top10mean`
- `C32S16_top1`
- `C32S16_top5mean`
- `C32S16_top10mean`

## 5. Expected Inputs

- clean EAT checkpoint path:
  - `/data1/test/pretrained_models/eat_clean/checkpoints`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- machine:
  - `gearbox`
- pooling root for runtime `P1_mean_std` baseline if available:
  - `results/stage16b_clean_eat_pooling_fusion`
- fixed scoring:
  - cosine `kNN`
  - `k = 5`

## 6. Expected Outputs

Base output root:

- `results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/`

Expected files:

- `check_status.json`
- `machine_check.csv`
- `chunk_cache_summary.csv`
- `chunk_memory_summary.csv`
- `chunk_cache/C08S04_gearbox_chunks.npz`
- `chunk_cache/C16S08_gearbox_chunks.npz`
- `chunk_cache/C32S16_gearbox_chunks.npz`
- `scores/C08S04_top1/gearbox_scores.npz`
- `scores/C08S04_top5mean/gearbox_scores.npz`
- `scores/C08S04_top10mean/gearbox_scores.npz`
- `scores/C16S08_top1/gearbox_scores.npz`
- `scores/C16S08_top5mean/gearbox_scores.npz`
- `scores/C16S08_top10mean/gearbox_scores.npz`
- `scores/C32S16_top1/gearbox_scores.npz`
- `scores/C32S16_top5mean/gearbox_scores.npz`
- `scores/C32S16_top10mean/gearbox_scores.npz`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `reference_comparison.csv`
- `segment_chunk_verdict.json`
- `docs/CTAF_STAGE16C2_CLEAN_EAT_SEGMENT_CHUNK_ONE_MACHINE_GEARBOX_AUDIT.md`

## 7. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16c2_clean_eat_segment_chunk_one_machine.py --mode check --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox --doc_path docs/CTAF_STAGE16C2_CLEAN_EAT_SEGMENT_CHUNK_ONE_MACHINE_GEARBOX_AUDIT.md --pooling_root results/stage16b_clean_eat_pooling_fusion --latent_knn_k 5
```

## 8. Extract Chunk Cache Command

Use second GPU via `CUDA_VISIBLE_DEVICES=1` and script arg `--device cuda:0`.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage16c2_clean_eat_segment_chunk_one_machine.py --mode extract_chunk_cache --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox --doc_path docs/CTAF_STAGE16C2_CLEAN_EAT_SEGMENT_CHUNK_ONE_MACHINE_GEARBOX_AUDIT.md --pooling_root results/stage16b_clean_eat_pooling_fusion --latent_knn_k 5
```

## 9. Score Chunk Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16c2_clean_eat_segment_chunk_one_machine.py --mode score_chunk --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox --doc_path docs/CTAF_STAGE16C2_CLEAN_EAT_SEGMENT_CHUNK_ONE_MACHINE_GEARBOX_AUDIT.md --pooling_root results/stage16b_clean_eat_pooling_fusion --latent_knn_k 5
```

## 10. Confirmation

- no experiment was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
- no scoring was run in Codex
