# CTAF Experiment Protocol

## Research Question

Does a multi-view time-frequency tensor frontend improve unsupervised machine
anomaly detection over a clean log-Mel baseline, under a fully label-free protocol?

## Experiment Groups

| ID | Name | Views | V |
|---|---|---|---|
| A | logmel_only | log-Mel | 1 |
| B | dynamic_3view | log-Mel + delta + delta-delta | 3 |
| C | edge_3view | log-Mel + edge + enhanced | 3 |
| D | full_5view | all five views | 5 |

## Success Criteria

- **Pass**: full_5view hmean > logmel_only hmean AND full_5view pAUC >= logmel_only pAUC
- **Worth developing**: >= 4/7 machines improved, source-target KS not worsened
- **Stop**: full_5view <= logmel_only, or pAUC drops, or only 1 machine improves

## What Is Forbidden

- Label-dependent score direction (reverse/forward selection)
- Weight search using anomaly labels
- Machine-specific architecture tuning
- Official-score-driven model selection
- Per-machine experiment group selection

## What Is Allowed

- Normal-only training data
- Source/target domain metadata
- Fixed pre-registered hyperparameters
- Post-hoc evaluation using anomaly labels (after scores are fixed)
