# CTAF Stage16B Clean EAT Pooling Fusion Implementation

## 1. Modified File List

- `scripts/audit_stage16b_clean_eat_pooling_fusion.py`
- `docs/CTAF_STAGE16B_CLEAN_EAT_POOLING_FUSION_IMPLEMENTATION.md`

## 2. Script Modes

- `--mode check`
  - checks:
    - EAT model path exists
    - data_root exists
    - all machines exist
    - source train-normal files exist
    - test files exist
    - V63 root exists
    - Stage14A/A1 root exists
    - imports available
  - no model load
  - no forward
  - no embedding extraction
  - no scoring
  - no fusion

- `--mode extract_pooling_embeddings`
  - loads clean EAT from local path
  - `model.eval()`
  - `torch.no_grad()`
  - extracts frame outputs `[B, T, 768]`
  - computes:
    - `P0_mean`
    - `P1_mean_std`
    - `P2_mean_max`
    - `P3_mean_std_max`
  - saves per-machine embedding caches
  - no scoring in this mode

- `--mode score_pooling_knn`
  - loads saved pooling embeddings
  - computes `kNN, k=5`
  - writes standalone pooling metrics

- `--mode fusion_pooling`
  - loads pooling scores
  - loads `V63 / A0 raw BEATs / A1`
  - alignment check
  - computes fixed fusion candidates
  - writes summary and per-machine results

## 3. Fixed Pooling Candidates

- `P0_mean`
- `P1_mean_std`
- `P2_mean_max`
- `P3_mean_std_max`

## 4. Fixed Fusion Candidates

For each pooling candidate `P`:

- `F_P_A1_050 = 0.5 * rank_norm(P) + 0.5 * rank_norm(A1)`
- `F_P_A1_065 = 0.65 * rank_norm(P) + 0.35 * rank_norm(A1)`
- `F_P_V63_050 = 0.5 * rank_norm(P) + 0.5 * rank_norm(V63)`
- `F_P_TRI = (rank_norm(P) + rank_norm(V63) + rank_norm(A0_raw_BEATs)) / 3`

No learned weights. No per-machine weights.

## 5. Expected Inputs

- EAT model path:
  - `/data1/test/pretrained_models/eat_clean/checkpoints`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seed:
  - `13711`
- V63 root:
  - `results/v63_fixed_v37_v61_rankfusion`
- Stage14A root:
  - `results/stage14a_frozen_beats_v63_fusion`

## 6. Expected Outputs

- `results/stage16b_clean_eat_pooling_fusion/check_status.json`
- `results/stage16b_clean_eat_pooling_fusion/machine_check.csv`
- `results/stage16b_clean_eat_pooling_fusion/embedding_summary.csv`
- `results/stage16b_clean_eat_pooling_fusion/pooling_summary.csv`
- `results/stage16b_clean_eat_pooling_fusion/pooling_per_machine.csv`
- `results/stage16b_clean_eat_pooling_fusion/pooling_tail_table.csv`
- `results/stage16b_clean_eat_pooling_fusion/fusion_summary.csv`
- `results/stage16b_clean_eat_pooling_fusion/fusion_per_machine.csv`
- `results/stage16b_clean_eat_pooling_fusion/fusion_tail_table.csv`
- `results/stage16b_clean_eat_pooling_fusion/alignment_summary.csv`
- `results/stage16b_clean_eat_pooling_fusion/complementarity_summary.csv`
- `results/stage16b_clean_eat_pooling_fusion/score_geometry.csv`
- `results/stage16b_clean_eat_pooling_fusion/embeddings/<pooling>/<machine>_embeddings.npz`
- `results/stage16b_clean_eat_pooling_fusion/scores/<candidate>/<machine>_scores.npz`
- `docs/CTAF_STAGE16B_CLEAN_EAT_POOLING_FUSION_AUDIT.md`

## 7. Server Command for check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_pooling_fusion.py \
  --mode check \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cpu \
  --output_root results/stage16b_clean_eat_pooling_fusion \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_POOLING_FUSION_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 8. Server Command for extract_pooling_embeddings

Use second GPU via `CUDA_VISIBLE_DEVICES=1` and script arg `--device cuda:0`.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage16b_clean_eat_pooling_fusion.py \
  --mode extract_pooling_embeddings \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/stage16b_clean_eat_pooling_fusion \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_POOLING_FUSION_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 9. Server Command for score_pooling_knn

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_pooling_fusion.py \
  --mode score_pooling_knn \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cpu \
  --output_root results/stage16b_clean_eat_pooling_fusion \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_POOLING_FUSION_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 10. Server Command for fusion_pooling

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_pooling_fusion.py \
  --mode fusion_pooling \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cpu \
  --output_root results/stage16b_clean_eat_pooling_fusion \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_POOLING_FUSION_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 11. Confirmation

- no command was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
