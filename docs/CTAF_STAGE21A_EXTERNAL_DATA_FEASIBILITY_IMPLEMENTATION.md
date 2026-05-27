# CTAF Stage21A External Data Feasibility Implementation

## Scope

This stage implements a reproducible **external normal-only data feasibility audit** for the next high-ceiling training phase.

It is intentionally **planning-only**:

- no dataset download
- no training
- no model loading
- no forward
- no embedding extraction
- no score recomputation

The goal is to freeze a **data-layer recommendation** before opening any new Stage21B training route.

## Implemented script

- `scripts/audit_stage21a_external_data_feasibility.py`

## Modes

### `--mode check`

Checks:

- Stage18F root exists
- Stage18G root exists
- required Stage18F / Stage18G anchor files exist
- output root is writable
- lightweight imports are available

No audit logic is executed beyond path / import validation.

### `--mode audit`

Reads Stage18F / Stage18G anchor verdicts and emits a curated feasibility package for external normal-only machine-sound data.

This stage uses a fixed registry of candidate datasets and challenge-rule compatibility notes.

## Output root

- `results/stage21a_external_data_feasibility_audit/`

## Expected outputs

- `check_status.json`
- `path_check.csv`
- `external_dataset_candidates.csv`
- `recommended_safe_datasets.csv`
- `rejected_or_high_risk_datasets.csv`
- `dcase_rule_notes.md`
- `stage21a_data_feasibility_verdict.json`
- `docs/CTAF_STAGE21A_EXTERNAL_DATA_FEASIBILITY_AUDIT.md`

## Current recommendation logic

The current first recommendation is:

- `DCASE 2022 Challenge Task 2 Additional Training Dataset`

Reason:

- direct machine-family overlap with the current benchmark
- contains fan / gearbox / bearing / slide rail / toy / valve style data
- compatible with current DCASE-style usage if restricted to train-normal clips and all evaluation/test clips are excluded

## Hard exclusions

The current audit marks these as high-risk / rejected:

- `ToyADMOS`
- `ToyADMOS2`
- `MIMII`
- `MIMII DUE`
- `MIMII DG`

Reason:

- they are explicitly incompatible with the challenge-rule framing used by the current project

## Why this stage matters

Stage18F/G established the strongest frozen protocol route:

- `EAT min_clamp__mean + ldn_diff_K1_scorek5`

Stage19 / Stage20 showed that low-risk local tweaks and small training probes did not produce a reliable next jump.

Therefore Stage21A moves the search up one level:

- from protocol tweaks
- to data-layer feasibility

before any new high-risk training design is allowed.
