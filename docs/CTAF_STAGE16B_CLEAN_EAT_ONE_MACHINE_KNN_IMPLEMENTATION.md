# CTAF Stage16B Clean EAT One-Machine KNN Implementation

## 1. Modified File List

- `scripts/audit_stage16b_clean_eat_one_machine_knn.py`
- `docs/CTAF_STAGE16B_CLEAN_EAT_ONE_MACHINE_KNN_IMPLEMENTATION.md`

## 2. Script Modes

- `--mode check`
  - verifies:
    - EAT model path exists
    - data_root exists
    - requested machine exists
    - source train-normal files exist
    - test files exist
    - required imports are available
  - no model load
  - no forward
  - no embedding extraction
  - no scoring

- `--mode one_machine_knn`
  - loads clean EAT from local path only
  - `model.eval()`
  - `torch.no_grad()`
  - extracts frame output for current-year source train-normal and test files
  - uses mean pooling over time
  - computes frozen kNN with `k=5`
  - writes one-machine metrics and optional baseline comparisons

## 3. Fixed Protocol

- clean EAT model path:
  - `/data1/test/pretrained_models/eat_clean/checkpoints`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- machine:
  - `gearbox` (recommended first sanity machine)
- frozen EAT only
- utterance embedding:
  - mean pooling over time from `[B, T, 768]`
- memory:
  - current-year source train-normal only
- test:
  - current-year test split
- scoring:
  - kNN with `k=5`
- no training
- no fine-tuning
- no projection head
- no PCA / whitening
- no fusion tuning
- no per-machine selection

## 4. Expected Inputs

- `--eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints`
- `--data_root /data1/test/LAES_2025/data/dcase`
- `--machine gearbox`
- `--device cpu/cuda:0/cuda:1`
- `--output_root results/stage16b_clean_eat_one_machine_knn`
- optional baseline roots:
  - `results/v63_fixed_v37_v61_rankfusion`
  - `results/stage14a_frozen_beats_v63_fusion`

## 5. Expected Outputs

- `results/stage16b_clean_eat_one_machine_knn/check_status.json`
- `results/stage16b_clean_eat_one_machine_knn/embedding_summary.csv`
- `results/stage16b_clean_eat_one_machine_knn/embeddings/<machine>_clean_eat_embeddings.npz`
- `results/stage16b_clean_eat_one_machine_knn/scores/<machine>_clean_eat_knn_scores.npz`
- `results/stage16b_clean_eat_one_machine_knn/summary.csv`
- `results/stage16b_clean_eat_one_machine_knn/tail_table.csv`
- `results/stage16b_clean_eat_one_machine_knn/score_geometry.csv`
- `results/stage16b_clean_eat_one_machine_knn/baseline_comparison.csv` (if available)
- `docs/CTAF_STAGE16B_CLEAN_EAT_ONE_MACHINE_KNN_AUDIT.md`

## 6. Server Command for check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_one_machine_knn.py \
  --mode check \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machine gearbox \
  --seed 13711 \
  --device cpu \
  --output_root results/stage16b_clean_eat_one_machine_knn \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_ONE_MACHINE_KNN_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 7. Server Command for one_machine_knn on gearbox

Use second GPU by setting `CUDA_VISIBLE_DEVICES=1` and script argument `--device cuda:0`.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage16b_clean_eat_one_machine_knn.py \
  --mode one_machine_knn \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machine gearbox \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/stage16b_clean_eat_one_machine_knn \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_ONE_MACHINE_KNN_AUDIT.md \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --latent_knn_k 5
```

## 8. Confirmation

- no command was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
