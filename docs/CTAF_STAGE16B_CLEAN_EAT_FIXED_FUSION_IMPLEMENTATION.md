# CTAF Stage16B Clean EAT Fixed Fusion Implementation

## 1. Modified File List

- `scripts/audit_stage16b_clean_eat_fixed_fusion.py`
- `docs/CTAF_STAGE16B_CLEAN_EAT_FIXED_FUSION_IMPLEMENTATION.md`

## 2. Script Modes

- `--mode check`
  - checks:
    - clean EAT score files exist
    - V63 score files exist
    - A0 raw BEATs score files exist
    - A1 score files exist
    - machines/seeds availability
  - no metric computation
  - no fusion
  - no model load
  - no forward

- `--mode fixed_fusion`
  - performs alignment check
  - computes fixed rank-normalized fusions only
  - no training
  - no model forward

## 3. Fixed Candidates

Base candidates:

- `clean_EAT`
- `V63`
- `A0_raw_BEATs`
- `A1`

Fixed fusion candidates:

- `F_EAT_V63 = 0.5 * rank_norm(clean_EAT) + 0.5 * rank_norm(V63)`
- `F_EAT_BEATs = 0.5 * rank_norm(clean_EAT) + 0.5 * rank_norm(A0_raw_BEATs)`
- `F_EAT_A1 = 0.5 * rank_norm(clean_EAT) + 0.5 * rank_norm(A1)`
- `F_TRI_EAT_V63_BEATs = (rank_norm(clean_EAT) + rank_norm(V63) + rank_norm(A0_raw_BEATs)) / 3`
- `F_HMEAN_ORIENTED_EAT_A1 = 0.65 * rank_norm(clean_EAT) + 0.35 * rank_norm(A1)`
- `F_BALANCED_EAT_A1 = 0.35 * rank_norm(clean_EAT) + 0.65 * rank_norm(A1)`

## 4. Expected Inputs

- clean EAT root:
  - `results/stage16b_clean_eat_full_frozen_knn`
- V63 root:
  - `results/v63_fixed_v37_v61_rankfusion`
- A0/A1 root:
  - `results/stage14a_frozen_beats_v63_fusion`
- machines:
  - `bearing,fan,gearbox,slider,toycar,toytrain,valve`
- seeds:
  - `13711,13721,13731`

## 5. Expected Outputs

- `results/stage16b_clean_eat_fixed_fusion/check_status.json`
- `results/stage16b_clean_eat_fixed_fusion/alignment_summary.csv`
- `results/stage16b_clean_eat_fixed_fusion/summary.csv`
- `results/stage16b_clean_eat_fixed_fusion/per_machine.csv`
- `results/stage16b_clean_eat_fixed_fusion/tail_table.csv`
- `results/stage16b_clean_eat_fixed_fusion/complementarity_summary.csv`
- `results/stage16b_clean_eat_fixed_fusion/score_geometry.csv`
- `results/stage16b_clean_eat_fixed_fusion/baseline_comparison.csv`
- `results/stage16b_clean_eat_fixed_fusion/scores/<candidate>/<machine>_scores.npz`
- `docs/CTAF_STAGE16B_CLEAN_EAT_FIXED_FUSION_AUDIT.md`

## 6. Server Command for check

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_fixed_fusion.py \
  --mode check \
  --data_root /data1/test/LAES_2025/data/dcase \
  --eat_root results/stage16b_clean_eat_full_frozen_knn \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seeds 13711,13721,13731 \
  --output_root results/stage16b_clean_eat_fixed_fusion \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_FIXED_FUSION_AUDIT.md
```

## 7. Server Command for fixed_fusion

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_fixed_fusion.py \
  --mode fixed_fusion \
  --data_root /data1/test/LAES_2025/data/dcase \
  --eat_root results/stage16b_clean_eat_full_frozen_knn \
  --v63_root results/v63_fixed_v37_v61_rankfusion \
  --stage14a_root results/stage14a_frozen_beats_v63_fusion \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --seeds 13711,13721,13731 \
  --output_root results/stage16b_clean_eat_fixed_fusion \
  --doc_path docs/CTAF_STAGE16B_CLEAN_EAT_FIXED_FUSION_AUDIT.md
```

## 8. Confirmation

- no command was run in Codex
- no training was run in Codex
- no model forward was run in Codex
- no embedding extraction was run in Codex
