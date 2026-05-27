# CTAF Stage16D Clean EAT Temporal Pooling Revisit One-Machine Implementation

## 1. Modified File List

- `scripts/audit_stage16d_clean_eat_temporal_pooling_revisit_one_machine.py`
- `docs/CTAF_STAGE16D_CLEAN_EAT_TEMPORAL_POOLING_REVISIT_ONE_MACHINE_IMPLEMENTATION.md`

## 2. Implementation Summary

- This script implements Stage16D one-machine training-free temporal pooling revisit on frozen clean `EAT` for `gearbox`.
- It is strictly frozen-EAT only:
  - no training
  - no fine-tuning
  - no LoRA
  - no adapter
  - no projection head
  - no Conformer
  - no A1/V63 fusion
  - no alpha sweep
  - no per-machine route
  - no oracle
  - no 7-machine run
- The script has three modes:
  - `--mode check`
  - `--mode extract_pooling_embeddings`
  - `--mode score_pooling`
- `extract_pooling_embeddings` is the only mode that loads clean EAT and runs frozen forward.
- `score_pooling` uses saved pooled embeddings only and does not load the model.

## 3. Fixed Preprocessing Variants

- `none`
- `min_clamp`
  - `x = where(abs(x) < 0.1, 0, x)`
- `spike_suppression`
  - if `x > 0.5`, set `x = 0.5 + tanh(x - 0.5)`
  - if `x < -0.5`, set `x = -0.5 + tanh(x + 0.5)`
  - values in `[-0.5, 0.5]` unchanged
- `min_clamp_spike_suppression`
  - apply `min_clamp` first
  - then `spike_suppression`

## 4. Fixed Pooling Candidates

Pooling methods:

- `mean`
  - temporal mean over `T`
  - dim `768`
- `mean_std`
  - concat temporal mean and std
  - dim `1536`
- `gem_p3`
  - GeM with `p = 3`
  - on `max(0, x)`
  - dim `768`
- `rdp_g1`
  - RDP with `gamma = 1`
  - weights based on normalized temporal deviations from mean
  - dim `768`
- `rdp_gem_p3_g1`
  - weighted GeM with `p = 3`
  - weights from `RDP gamma = 1`
  - dim `768`

Candidate naming:

- `<preprocess>__<pooling>`

Examples:

- `none__mean`
- `none__mean_std`
- `min_clamp__mean`
- `spike_suppression__gem_p3`
- `min_clamp_spike_suppression__rdp_gem_p3_g1`

## 5. Scoring Protocol

- extract pooled embeddings for gearbox source train-normal only as memory
- extract pooled embeddings for gearbox test
- use cosine `kNN`
- `k = 5`
- no target train-normal
- no test labels in scoring
- metrics reported after scoring:
  - `hmean`
  - `pAUC`
  - `KS`
  - `TPR@95n`
  - `TPR@99n`
  - `q95`
  - `q99`

## 6. Expected Outputs

Base output root:

- `results/stage16d_clean_eat_temporal_pooling_revisit_one_machine_gearbox/`

Files:

- `check_status.json`
- `machine_check.csv`
- `pooling_embedding_summary.csv`
- `embeddings/<candidate>_gearbox_embeddings.npz`
- `scores/<candidate>/gearbox_scores.npz`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `reference_comparison.csv`
- `temporal_pooling_verdict.json`
- `docs/CTAF_STAGE16D_CLEAN_EAT_TEMPORAL_POOLING_REVISIT_ONE_MACHINE_GEARBOX_AUDIT.md`

## 7. Check Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16d_clean_eat_temporal_pooling_revisit_one_machine.py --mode check --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage16d_clean_eat_temporal_pooling_revisit_one_machine_gearbox --doc_path docs/CTAF_STAGE16D_CLEAN_EAT_TEMPORAL_POOLING_REVISIT_ONE_MACHINE_GEARBOX_AUDIT.md --p1_score_path results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz --latent_knn_k 5
```

## 8. Extract Pooling Embeddings Command

Use second GPU via `CUDA_VISIBLE_DEVICES=1` and script arg `--device cuda:0`.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage16d_clean_eat_temporal_pooling_revisit_one_machine.py --mode extract_pooling_embeddings --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cuda:0 --output_root results/stage16d_clean_eat_temporal_pooling_revisit_one_machine_gearbox --doc_path docs/CTAF_STAGE16D_CLEAN_EAT_TEMPORAL_POOLING_REVISIT_ONE_MACHINE_GEARBOX_AUDIT.md --p1_score_path results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz --latent_knn_k 5
```

## 9. Score Pooling Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && source /home/yinpin/miniconda3/etc/profile.d/conda.sh && conda activate scanf && python scripts/audit_stage16d_clean_eat_temporal_pooling_revisit_one_machine.py --mode score_pooling --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints --data_root /data1/test/LAES_2025/data/dcase --machine gearbox --seed 13711 --device cpu --output_root results/stage16d_clean_eat_temporal_pooling_revisit_one_machine_gearbox --doc_path docs/CTAF_STAGE16D_CLEAN_EAT_TEMPORAL_POOLING_REVISIT_ONE_MACHINE_GEARBOX_AUDIT.md --p1_score_path results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz --latent_knn_k 5
```

## 10. Confirmation

- no experiment was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
- no scoring was run in Codex
