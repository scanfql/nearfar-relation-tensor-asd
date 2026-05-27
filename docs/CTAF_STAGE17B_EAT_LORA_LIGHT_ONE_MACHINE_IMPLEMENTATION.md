# CTAF Stage17B EAT LoRA-Light One-Machine Implementation

## 1. Modified File List

- `scripts/audit_stage17b_eat_lora_light_one_machine.py`
- `docs/CTAF_STAGE17B_EAT_LORA_LIGHT_ONE_MACHINE_IMPLEMENTATION.md`

## 2. Fixed Stage17B First-Sanity Decisions

- machine:
  - `gearbox`
- backbone:
  - clean `EAT`
  - local checkpoint only
  - `/data1/test/pretrained_models/eat_clean/checkpoints`
- data root:
  - `/data1/test/LAES_2025/data/dcase`
- target modules:
  - attention `qkv/proj` linear only
  - `model.blocks.*.attn.qkv`
  - `model.blocks.*.attn.proj`
- forbidden tuning:
  - no `MLP` / `FFN`
  - no patch embedding
  - no normalization
  - no classifier head
  - no full fine-tuning
- LoRA:
  - `rank = 4`
  - `alpha = 8`
  - `dropout = 0.05`
- pooling:
  - `P1_mean_std`
  - `concat(mean_T(frame_output), std_T(frame_output))`
  - embedding dim = `1536`
- train / val / memory / test:
  - current-year `source train-normal` only for train/val
  - `90% / 10%` split by seed `13711`
  - source train-normal memory for kNN
  - current-year test split for evaluation
- scoring:
  - cosine `kNN`
  - `k = 5`
- no fusion:
  - no `A1` / `V63`
  - no alpha sweep
  - no per-machine selection
  - no 7-machine run
  - no Stage17C

## 3. Fixed Stage17B First-Sanity Objective

Loss is applied on normalized pooled embeddings, not directly on frame output.

Definitions:

- `t0 = L2Norm(P1_mean_std(teacher(x0)))`
- `s0 = L2Norm(P1_mean_std(student(x0)))`
- `s1 = L2Norm(P1_mean_std(student(x1)))`

Weak paired-view construction:

- `x0 = original waveform`
- `x1 = weakly augmented waveform from the same file`

Allowed weak augmentations, all fixed and applied together:

1. gain jitter
   - multiplier uniformly sampled from `[0.9, 1.1]`
2. additive Gaussian noise
   - fixed `SNR = 30 dB`
3. small time shift
   - integer shift sampled from `[-160, 160]` at `16 kHz`
   - zero padding only
   - no circular shift

Not allowed:

- no SpecAugment
- no time masking
- no frequency masking
- no strong augmentation
- no pseudo anomaly
- no mixup
- no cutmix

Fixed loss:

`L = 1.0 * L_anchor0 + 1.0 * L_anchor1 + 0.2 * L_view + 0.1 * L_rel`

where:

- `L_anchor0 = mean(1 - cosine(s0, t0))`
- `L_anchor1 = mean(1 - cosine(s1, t0))`
- `L_view = mean(1 - cosine(s0, s1))`
- `L_rel = MSE(offdiag(sim_s), offdiag(sim_t))`
- `sim_t = cosine_similarity_matrix(t0)`
- `sim_s = cosine_similarity_matrix(s0)`
- if batch size is `1`, then `L_rel = 0`

Not added:

- no center loss
- no compactness loss
- no CE
- no SupCon
- no pseudo-anomaly loss

## 4. Script Modes

- `--mode check`
  - verifies clean EAT checkpoint path exists
  - verifies `data_root` exists
  - verifies `gearbox` current-year source train-normal and test files exist
  - checks import availability
  - performs static source-file inspection for `qkv` / `proj` / attention tokens
  - does not load the model
  - no forward
  - no training
  - no extraction
  - no scoring

- `--mode inspect_lora_targets`
  - loads clean EAT from local checkpoint package only
  - enumerates all `Linear` module names
  - selects only attention `qkv` / `proj` linear modules
  - injects LoRA-light wrappers for parameter accounting only
  - freezes all base EAT parameters
  - audits that only LoRA parameters remain trainable
  - no forward
  - no training
  - no extraction
  - no scoring

- `--mode one_machine_lora_sanity`
  - loads frozen teacher clean EAT
  - loads student clean EAT
  - injects LoRA-light into selected attention `qkv/proj` linear layers
  - freezes all base EAT parameters
  - trains only LoRA parameters
  - uses teacher-anchored weak-augmentation consistency on `P1_mean_std`
  - saves best checkpoint by lowest validation total loss
  - reloads best checkpoint
  - extracts adapted student `P1_mean_std` embeddings for:
    - gearbox source train-normal memory
    - gearbox test split
  - computes cosine `kNN` with `k = 5`
  - computes fixed metrics
  - computes frozen runtime `P1_mean_std` reference in the same output bundle
  - no fusion

## 5. Fixed One-Machine Training Configuration

- epochs:
  - `5`
- batch size:
  - `8`
- lr:
  - `1e-4`
- weight decay:
  - `1e-4`
- grad clip:
  - `1.0`
- early stop patience:
  - `2`
- checkpoint selection:
  - lowest validation total loss
- best checkpoint path:
  - `checkpoints/gearbox_lora_light_best.pt`

## 6. Implemented Output Files

Base output root:

- `results/stage17b_eat_lora_light_one_machine/`

From `--mode check`:

- `check_status.json`
- `static_target_discovery.json`
- `gearbox_split_inventory.csv`
- audit markdown via `--doc_path`

From `--mode inspect_lora_targets`:

- `target_module_inventory.csv`
- `parameter_audit.csv`
- `inspect_lora_targets.json`
- audit markdown via `--doc_path`

From `--mode one_machine_lora_sanity`:

- `train_log.csv`
- `lora_gradient_audit.csv`
- `parameter_audit_after_training.csv`
- `checkpoints/gearbox_lora_light_best.pt`
- `embeddings/gearbox_lora_p1_embeddings.npz`
- `scores/gearbox_lora_p1_scores.npz`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `frozen_reference_comparison.csv`
- `one_machine_lora_sanity_result.json`
- audit markdown via `--doc_path`

## 7. Metrics and Comparison Targets

After training, the script reports:

- `hmean`
- `pAUC`
- `KS`
- `TPR@95n`
- `TPR@99n`
- `q95`
- `q99`

Registered frozen gearbox `P1_mean_std` reference:

- `hmean = 0.561801`
- `pAUC = 0.493158`
- `TPR@99n = 0.020000`

Registered previous gearbox `F_P1_A1_065` reference:

- `hmean = 0.572219`
- `pAUC = 0.519474`
- `TPR@99n = 0.030000`

The script writes these comparisons into:

- `frozen_reference_comparison.csv`
- `one_machine_lora_sanity_result.json`
- audit markdown

## 8. Server Command for `--mode check`

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage17b_eat_lora_light_one_machine.py \
  --mode check \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machine gearbox \
  --seed 13711 \
  --device cpu \
  --output_root results/stage17b_eat_lora_light_one_machine \
  --doc_path docs/CTAF_STAGE17B_EAT_LORA_LIGHT_ONE_MACHINE_AUDIT.md \
  --latent_knn_k 5 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05
```

## 9. Server Command for `--mode inspect_lora_targets`

Use second GPU as requested.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage17b_eat_lora_light_one_machine.py \
  --mode inspect_lora_targets \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machine gearbox \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/stage17b_eat_lora_light_one_machine \
  --doc_path docs/CTAF_STAGE17B_EAT_LORA_LIGHT_ONE_MACHINE_AUDIT.md \
  --latent_knn_k 5 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05
```

## 10. Server Command for `--mode one_machine_lora_sanity`

Use second GPU as requested.

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
CUDA_VISIBLE_DEVICES=1 python scripts/audit_stage17b_eat_lora_light_one_machine.py \
  --mode one_machine_lora_sanity \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machine gearbox \
  --seed 13711 \
  --device cuda:0 \
  --output_root results/stage17b_eat_lora_light_one_machine \
  --doc_path docs/CTAF_STAGE17B_EAT_LORA_LIGHT_ONE_MACHINE_AUDIT.md \
  --latent_knn_k 5 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05
```

## 11. Confirmation

- no experiment was run in Codex
- no forward was run in Codex
- no training was run in Codex
- no evaluation was run in Codex
- no embedding extraction was run in Codex
