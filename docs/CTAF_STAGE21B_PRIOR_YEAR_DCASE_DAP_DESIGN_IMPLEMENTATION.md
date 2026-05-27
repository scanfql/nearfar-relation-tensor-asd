# CTAF Stage21B Prior-Year Official DCASE DAP Design Implementation

## Scope

This stage freezes the **first high-ceiling external-data training design** after Stage21A confirmed that prior-year official DCASE normal-only corpora are safe and justified.

Stage21B is **design-only**:

- no training
- no model loading
- no forward
- no embedding extraction
- no ASD scoring

The purpose is to freeze:

- which prior-year corpora are enabled first
- which corpora are deferred
- which leakage rules remain hard constraints
- how the next training/evaluation stages should be sequenced

## Implemented script

- `scripts/audit_stage21b_prior_year_dcase_dap_design.py`

## Modes

### `--mode check`

Checks:

- Stage21A root exists
- Stage18F root exists
- Stage18G root exists
- required Stage21A / Stage18F / Stage18G anchor files exist
- output root is writable
- lightweight imports are available

No design outputs are created beyond path/import validation.

### `--mode design`

Reads the Stage21A recommendation package and freezes the first prior-year DCASE DAP plan.

Current first implementation target:

- `DCASE 2022 Challenge Task 2 Additional Training Dataset`

Current design logic:

- enable only the smallest, best-matched, prior-year same-family corpus by default
- keep `DCASE 2022 Development` as the first extension
- keep `2021/2020` as later ablations
- defer `2023/2024/2025` safe corpora from the first pass to preserve causal attribution

## Output root

- `results/stage21b_prior_year_dcase_dap_design/`

## Expected outputs

- `check_status.json`
- `path_check.csv`
- `selected_prior_year_corpora.csv`
- `deferred_safe_corpora.csv`
- `data_usage_policy.csv`
- `stage21b_training_schedule.csv`
- `stage21b_risk_register.csv`
- `stage21b_protocol_freeze.json`
- `stage21b_training_route_plan.md`
- `docs/CTAF_STAGE21B_PRIOR_YEAR_DCASE_DAP_DESIGN_AUDIT.md`

## Frozen decisions

- first implementation corpus:
  - `DCASE 2022 Challenge Task 2 Additional Training Dataset`
- first evaluation machine:
  - `fan`
- second evaluation machine:
  - `gearbox`
- downstream backend remains:
  - `min_clamp__mean + ldn_diff_K1_scorek5`

## Hard exclusions

- any evaluation/test audio
- any anomaly labels for training
- ToyADMOS family
- MIMII family
- selector/oracle routes
- extra fusion added only to chase score

## Why this stage exists

Stage18F/G already froze the strongest current route.
Stage19/20 showed that low-risk protocol tweaks and local adaptation probes were not enough.
Stage21A showed that the next justified lever is **external prior-year official normal-only DCASE data**.

Stage21B therefore turns that conclusion into a strict, reproducible first training design rather than another informal idea.
