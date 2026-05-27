#!/bin/bash
# Debug run: 3 machines, 2 experiments, 3 epochs
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_experiment \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines gearbox,slider,toycar \
  --experiments logmel_only,full_5view \
  --epochs 3 \
  --batch_size 16 \
  --seed 13711 \
  --device cuda:0 \
  --output_dir results/debug
