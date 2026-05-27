# CTAF Stage16A1 External Checkpoint Provenance Implementation

## 1. Modified File List

- `scripts/audit_stage16a1_external_checkpoint_provenance.py`
- `docs/CTAF_STAGE16A1_EXTERNAL_CHECKPOINT_PROVENANCE_IMPLEMENTATION.md`

## 2. Expected Inputs

- EAT root:
  - `/data1/test/asd_eat_repro/`
- BEATs LoRA root:
  - `/data1/test/dacase2023_task2_baseline/models/saved_model/`
- BEATs official root:
  - `/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/`

## 3. Expected Outputs

Output root:

- `results/stage16a1_external_checkpoint_provenance/`

Files:

- `eat_checkpoint_inventory.csv`
- `eat_config_log_summary.csv`
- `beats_lora_checkpoint_inventory.csv`
- `risk_classification.csv`
- `recommended_next_steps.md`
- `docs/CTAF_STAGE16A1_EXTERNAL_CHECKPOINT_PROVENANCE_AUDIT.md`

## 4. Server Command

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16a1_external_checkpoint_provenance.py \
  --eat_root /data1/test/asd_eat_repro \
  --beats_lora_root /data1/test/dacase2023_task2_baseline/models/saved_model \
  --beats_official_root /data1/test/dacase2023_task2_baseline/pretrained_models/beats_official \
  --output_root results/stage16a1_external_checkpoint_provenance \
  --doc_path docs/CTAF_STAGE16A1_EXTERNAL_CHECKPOINT_PROVENANCE_AUDIT.md
```

## 5. Confirmation

- no training code path
- no model forward
- no embedding extraction
- no full DCASE evaluation
- no checkpoint mutation
- no download logic
