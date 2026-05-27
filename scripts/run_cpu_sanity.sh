#!/bin/bash
# CPU sanity test: 1 machine, 2 experiments, 1 epoch
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_experiment \
  --data_root /data1/test/LAES_2025/data/dcase \
  --machines gearbox \
  --experiments logmel_only,full_5view \
  --epochs 1 \
  --batch_size 4 \
  --seed 13711 \
  --device cpu \
  --output_dir results/cpu_test
