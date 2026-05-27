#!/bin/bash
# Full run: 7 machines, 4 experiments, 50 epochs
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_experiment \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines bearing,fan,gearbox,slider,toycar,toytrain,valve \
  --experiments logmel_only,dynamic_3view,edge_3view,full_5view \
  --epochs 50 \
  --batch_size 32 \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/full
