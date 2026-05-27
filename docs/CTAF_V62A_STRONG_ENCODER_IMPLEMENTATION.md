# CTAF V62A Strong Encoder Implementation

## 1. Modified File List

- `src/models/conv_ae.py`
- `scripts/run_v62a_strong_encoder.py`
- `docs/CTAF_V62A_STRONG_ENCODER_IMPLEMENTATION.md`

## 2. Encoder Architecture Summary

Implemented strong encoder path:

- encoder type: `full_5view Conformer-style`
- input: full_5view tensor `[B, 5, 128, T]`
- per-view stem:
  - `Conv2d(1 -> 16, stride=2)`
  - `Conv2d(16 -> 32, stride=2)`
- late fusion:
  - concatenate all 5 views → `160` channels
  - `1x1` fusion projection to `hidden_dim = 128`
- sequence backbone:
  - `4` Conformer blocks
  - `4` attention heads
  - conformer depthwise convolution kernel = `15`
- decoder:
  - fused latent map `128 -> 160`
  - split back to 5 views
  - per-view transpose-conv decoder to reconstruction

Global summary remains fixed:

- latent_summary = `mean + std + max`

## 3. Parameter Count

- total parameter count: `1,639,413`
- target range check `1M–5M`: `PASS`

## 4. Input Shape

Shape check used:

- input shape: `[2, 5, 128, 313]`

## 5. Output Shape

- latent map shape: `[2, 128, 32, 79]`
- reconstruction output shape: `[2, 5, 128, 313]`

The output matches the input tensor shape after decoder crop alignment.

## 6. Latent Summary Dimension

- latent_summary shape: `[2, 384]`
- latent_summary dimension: `384`

This keeps the existing `latent_summary_knn` interface compatible while changing only encoder capacity.

## 7. shape_check Command

```bash
cd /Users/apple/Project/codex/LAES_2025/new_projects/clean_tensor_asd && \
python scripts/run_v62a_strong_encoder.py \
  --mode shape_check \
  --output_root results/v62a_strong_encoder_shape_check \
  --device cpu
```

Result:

- import check: `PASS`
- one-batch forward: `PASS`
- output/input shape compatibility: `PASS`
- latent_summary shape check: `PASS`
- parameter range check: `PASS`

## 8. Gearbox dryrun_train Command

This command is provided for manual server execution only.
It was **not** run in Codex.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/run_v62a_strong_encoder.py \
  --mode dryrun_train \
  --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machine gearbox \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/v62a_strong_encoder_dryrun \
  --baseline_dir results/v61_cross_year_original_encoder_seedpanel/seed_13711 \
  --epochs 50 \
  --batch_size 32 \
  --latent_knn_k 5 \
  --hidden_dim 128 \
  --num_layers 4 \
  --num_heads 4 \
  --conformer_conv_kernel 15
```

## 9. Full 7-Machine Single-Seed Command

This command is provided for manual server execution only.
It was **not** run in Codex.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/run_v62a_strong_encoder.py \
  --mode full_train \
  --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/v62a_strong_encoder_full_seed13711 \
  --baseline_dir results/v61_cross_year_original_encoder_seedpanel/seed_13711 \
  --epochs 50 \
  --batch_size 32 \
  --latent_knn_k 5 \
  --hidden_dim 128 \
  --num_layers 4 \
  --num_heads 4 \
  --conformer_conv_kernel 15
```

## 10. Expected Output Paths

shape_check:

- `results/v62a_strong_encoder_shape_check/shape_check_summary.json`

dryrun_train:

- `results/v62a_strong_encoder_dryrun/summary.csv`
- `results/v62a_strong_encoder_dryrun/tail_table.csv`
- `results/v62a_strong_encoder_dryrun/score_geometry.csv`
- `results/v62a_strong_encoder_dryrun/models/*.pt`
- `results/v62a_strong_encoder_dryrun/scores/*.npz`

full_train:

- `results/v62a_strong_encoder_full_seed13711/summary.csv`
- `results/v62a_strong_encoder_full_seed13711/tail_table.csv`
- `results/v62a_strong_encoder_full_seed13711/score_geometry.csv`
- `results/v62a_strong_encoder_full_seed13711/models/*.pt`
- `results/v62a_strong_encoder_full_seed13711/scores/*.npz`

## 11. Confirmation That No Full Training Was Run

- No `dryrun_train` was executed in Codex
- No `full_train` was executed in Codex
- Only `shape_check` was executed

## 12. Confirmation That No Protocol Changes Were Made

- full_5view frontend unchanged
- manifest unchanged
- scoring unchanged: `latent_summary_knn, k=5`
- latent summary rule unchanged: `mean + std + max`
- no loss changes introduced
- no BEATs / AST / EAT
- no external pretrained weights
- no UACS
- no score fusion
- no per-machine selection
- no V63 opened
