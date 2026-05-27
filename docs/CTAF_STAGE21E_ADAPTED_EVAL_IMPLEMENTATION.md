# CTAF Stage21E Adapted Eval Implementation

## Scope

This stage evaluates the checkpoint produced by Stage21D using the frozen Stage18 backend:

- `min_clamp__mean + ldn_diff_K1_scorek5`

It is machine-local and sequential:

- evaluate `fan` first
- evaluate `gearbox` second

No full7 conclusion is made in this stage.

## Implemented script

- `scripts/audit_stage21e_adapted_eval.py`

## Modes

### `--mode check`

Checks:

- clean EAT checkpoint exists
- Stage21D adapted checkpoint exists
- current benchmark data root exists
- Stage21D root exists
- Stage18E / Stage18F roots exist
- machine-local train/test inventory is visible
- required imports are available

No model load, no forward, no extraction, no scoring.

### `--mode inspect_checkpoint`

Allowed:

- load the Stage21D adapted checkpoint
- verify `student_state_dict`
- verify `recon_head_state_dict`
- verify `train_config`
- recap trainable boundary cleanliness from Stage21D

No data forward and no scoring.

### `--mode extract_adapted_embeddings`

Allowed:

- load clean EAT + Stage21D adapted checkpoint
- extract current-benchmark train/test embeddings for one machine
- apply:
  - 16 kHz
  - 128-bin Kaldi fbank
  - target length 1024
  - Stage16D min-clamp
  - mean pooling

No scoring in this mode.

### `--mode score_adapted_machine`

Allowed:

- read extracted embedding NPZ
- compute only:
  - `adapted_baseline_knn_k5`
  - `adapted_ldn_diff_K1_scorek5`
- compare against machine-local Stage18 references

No new candidates, no selector, no full7.

## Output structure

Outputs are written per machine:

- `results/stage21e_adapted_eval/fan/...`
- `results/stage21e_adapted_eval/gearbox/...`

## Expected outputs per machine

- `check_status.json`
- `checkpoint_status.json`
- `adapted_parameter_audit.csv`
- `trainable_boundary_recap.json`
- `embedding_summary.csv`
- `embeddings/<machine>_adapted_minclamp_mean_embeddings.npz`
- `scores/adapted_baseline_knn_k5/<machine>_scores.npz`
- `scores/adapted_ldn_diff_K1_scorek5/<machine>_scores.npz`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `comparison_vs_stage18_machine.csv`
- `stage21e_<machine>_adapted_eval_verdict.json`

## Machine-specific logic

### Fan

Primary gate:

- `adapted_ldn_diff_K1_scorek5 hmean > Stage18 fan ldn hmean`

Strong gate:

- `adapted_ldn_diff_K1_scorek5 hmean >= Stage18 fan baseline hmean`

### Gearbox

Primary guardrail:

- `adapted_ldn_diff_K1_scorek5 hmean >= Stage18 gearbox ldn hmean`

Secondary guardrail:

- `adapted_baseline_knn_k5 hmean >= Stage18 gearbox baseline hmean`

## Important boundary

This stage uses the frozen backend only.

It does not:

- tune K
- tune score protocol
- add fusion
- make any full7 claim
