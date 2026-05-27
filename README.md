# CTAF — Clean Tensor Acoustic Frontend for Label-Free ASD

Label-free multi-view tensor autoencoder for machine anomalous sound detection.

## Run

```bash
# Debug (3 machines, 3 epochs)
bash scripts/run_debug.sh

# Full experiment (7 machines, 50 epochs)
bash scripts/run_full.sh

# CPU sanity check
bash scripts/run_cpu_sanity.sh
```

## Outputs

- `results/*/summary.csv` — aggregate experiment results
- `results/*/per_machine.csv` — per-machine metrics
- `results/*/score_geometry.csv` — score distribution analysis
- `results/*/figures/` — score histograms
- `docs/RESULTS_SUMMARY.md` — human-readable report

## Current Status

- `docs/CTAF_V1_FULL_RUN_ANALYSIS_AND_V2_PROTOCOL.md` — V1 full-run conclusion and protocol-safe V2 plan.

## Protocol

- **Label-free**: No anomaly labels in training, feature extraction, or model selection.
- **Training**: Normal-only source-domain data, MSE reconstruction.
- **Scoring**: Mean squared reconstruction error (direction fixed: larger = anomalous).
- **No fusion**: Single model per machine per experiment. No weight search.
- **Fixed seed** 13711 for reproducibility.
