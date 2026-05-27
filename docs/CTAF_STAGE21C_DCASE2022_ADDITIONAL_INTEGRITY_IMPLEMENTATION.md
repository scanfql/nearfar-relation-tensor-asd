# CTAF Stage21C DCASE2022 Additional Integrity Implementation

## Scope

This stage converts the Stage21B data plan into a **real file-level audit** for the primary corpus:

- `DCASE 2022 Challenge Task 2 Additional Training Dataset`

The purpose is to answer:

- is the dataset actually downloaded?
- is it unpacked?
- which files are truly safe train-normal files?
- which files are risky or forbidden?
- do fan and gearbox both exist in the allowed manifest?

This stage is **integrity-only**:

- no training
- no model loading
- no forward
- no embedding extraction
- no ASD scoring

## Implemented script

- `scripts/audit_stage21c_dcase2022_additional_integrity.py`

## Modes

### `--mode check`

Checks:

- data roots exist
- Stage21B root exists
- required Stage21B protocol files exist
- output root is writable
- lightweight imports are available

No dataset scan beyond path existence.

### `--mode audit`

Performs the real integrity audit:

- attempts to discover the dataset root from known aliases or an explicit override
- distinguishes downloaded-but-not-unpacked from fully unpacked
- scans wav files under the selected dataset root
- infers:
  - machine
  - split group
  - normal/anomaly
  - source/target
  - section
- freezes:
  - `allowed_train_normal_filelist.csv`
  - `rejected_or_risky_filelist.csv`

## Output root

- `results/stage21c_dcase2022_additional_integrity_audit/`

## Expected outputs

- `check_status.json`
- `path_check.csv`
- `dataset_presence_check.json`
- `raw_file_inventory.csv`
- `summary_by_machine_domain_section.csv`
- `allowed_train_normal_filelist.csv`
- `rejected_or_risky_filelist.csv`
- `unexpected_structure_report.csv`
- `integrity_verdict.json`
- `docs/CTAF_STAGE21C_DCASE2022_ADDITIONAL_INTEGRITY_AUDIT.md`

## Allowlist rule

A file is allowed only if all are true:

- machine is recognized
- split is `train`
- inferred label is `normal`
- section is recognized

Everything else is rejected with an explicit `risk_tag`.

## Important design decision

The next training stage must consume:

- `allowed_train_normal_filelist.csv`

It must **not** re-glob the raw dataset directory and re-infer rules again.

This keeps the Stage21 pipeline traceable and leakage-resistant.
