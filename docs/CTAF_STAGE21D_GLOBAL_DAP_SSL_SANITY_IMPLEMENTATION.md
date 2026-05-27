# CTAF Stage21D Global DAP SSL Sanity Implementation

## Scope

This stage implements the first **global normal-only DAP training scaffold** using the frozen manifest from Stage21C:

- `results/stage21c_dcase2022_additional_integrity_audit/allowed_train_normal_filelist.csv`

It deliberately stays conservative:

- clean EAT initialization
- masked acoustic reconstruction
- most of the backbone frozen
- only the last 1 or last 2 transformer blocks trainable
- no ASD scoring in this stage

The goal is to produce a clean adapted checkpoint for later evaluation.

## Implemented script

- `scripts/audit_stage21d_global_dap_ssl_sanity.py`

## Modes

### `--mode check`

Checks:

- clean EAT checkpoint exists
- Stage21C allowlist exists
- Stage21B / Stage21C roots exist
- Stage21C integrity verdict exists
- allowlist row counts and basic summaries are readable
- output root is writable
- required imports are available

No model loading, no forward, no training.

### `--mode inspect_trainable`

Allowed:

- load clean EAT
- freeze all parameters
- enable only last `N` blocks plus a small reconstruction head
- write trainable/frozen parameter audits

No forward and no training.

### `--mode train_global_ssl`

Allowed:

- load clean EAT teacher/student
- read only the Stage21C allowlist
- perform stratified train/val split over `(machine, domain, section)`
- train masked acoustic reconstruction
- optionally add anchor loss (disabled by default)
- save best checkpoint by validation loss

Not allowed:

- no ASD scoring
- no fan/gearbox evaluation in this stage
- no backend tuning
- no selector/oracle logic

## Output root

- `results/stage21d_global_dap_ssl_sanity/`

## Expected outputs

- `check_status.json`
- `allowlist_group_summary.csv`
- `allowlist_machine_summary.csv`
- `global_split_inventory.csv`
- `model_load_status.json`
- `trainable_parameter_audit.csv`
- `frozen_parameter_audit.csv`
- `trainable_summary.json`
- `train_log.csv`
- `checkpoints/stage21d_global_best.pt`
- `stage21d_train_config.json`
- `stage21d_training_verdict.json`
- `docs/CTAF_STAGE21D_GLOBAL_DAP_SSL_SANITY_AUDIT.md`

## Key boundary

This stage must consume only:

- `allowed_train_normal_filelist.csv`

It must not re-scan the raw dataset directory and re-derive its own allowlist.

## Next stage

If Stage21D training is healthy, Stage21E should:

- extract adapted embeddings
- evaluate `fan` first
- evaluate `gearbox` second
- keep the backend frozen to `min_clamp__mean + ldn_diff_K1_scorek5`
