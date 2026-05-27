# CTAF Stage16B Clean EAT Full Frozen KNN Implementation

## 1. Modified File List

- `scripts/audit_stage16b_clean_eat_full_frozen_knn.py`
- `docs/CTAF_STAGE16B_CLEAN_EAT_FULL_FROZEN_KNN_IMPLEMENTATION.md`

## 2. Script Modes

- `--mode check`
  - verifies:
    - EAT model path exists
    - data_root exists
    - all requested machines exist
    - source train-normal files exist for all machines
    - test files exist for all machines
    - required imports are available
  - no model load
  - no forward
  - no embedding extraction
  - no scoring

- `--mode full_frozen_knn`
  - loads clean EAT from local path only
  - `model.eval()`
  - `torch.no_grad()`
  - extracts source train-normal and test embeddings for all 7 machines
  - mean-pools over time
  - computes frozen kNN with `k=5`
  - writes per-machine and overall metrics
  - writes optional baseline comparisons when available

## 3. Fixed Protocol

- clean EAT model path:
  - `/data1/test/pretrained_models/eat_clean/checkpoints`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seed reference:
  - `13711`
- frame output:
  - `[B, T, 768]`
- utterance embedding:
  - mean pooling over `T`
- memory:
  - current-year source train-normal only
- test:
  - current-year test set
- scoring:
  - `kNN, k=5`
- no training
- no fine-tuning
- no projection head
- no PCA / whitening
- no fusion
- no per-machine selection

## 4. Expected Inputs

- `--eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints`
- `--data_root /data1/test/LAES_2025/data/dcase`
- `--machines bearing,fan,gearbox,slider,toycar,toytrain,valve`
- `--seed 13711`
- `--device cuda:0/cpu`
- `--output_root results/stage16b_clean_eat_full_frozen_knn`
- optional baseline roots:
  - `results/v63_fixed_v37_v61_rankfusion`
  - `results/stage14a_frozen_beats_v63_fusion`

## 5. Expected Outputs

- `results/stage16b_clean_eat_full_frozen_knn/check_status.json`
- `results/stage16b_clean_eat_full_frozen_knn/machine_check.csv`
- `results/stage16b_clean_eat_full_frozen_knn/embedding_summary.csv`
- `results/stage16b_clean_eat_full_frozen_knn/summary.csv`
- `results/stage16b_clean_eat_full_frozen_knn/per_machine.csv`
- `results/stage16b_clean_eat_full_frozen_knn/tail_table.csv`
- `results/stage16b_clean_eat_full_frozen_knn/score_geometry.csv`
- `results/stage16b_clean_eat_full_frozen_knn/baseline_comparison.csv`
- `results/stage16b_clean_eat_full_frozen_knn/embeddings/<machine>_clean_eat_embeddings.npz`
- `results/stage16b_clean_eat_full_frozen_knn/scores/<machine>_clean_eat_knn_scores.npz`
- `docs/CTAF_STAGE16B_CLEAN_EAT_FULL_FROZEN_KNN_AUDIT.md`

## 6. Server Command for check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_full_frozen_knn.py \
  --mode check \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cpu \
  --output_root results/stage16b_clean_eat_full_frozen_knn \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_FULL_FROZEN_KNN_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 7. Server Command for full_frozen_knn

Use second GPU via `CUDA_VISIBLE_DEVICES=1` and script arg `--device cuda:0`.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage16b_clean_eat_full_frozen_knn.py \
  --mode full_frozen_knn \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/stage16b_clean_eat_full_frozen_knn \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_FULL_FROZEN_KNN_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 8. Confirmation

- no command was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
