# CTAF V60B Cross-Year Normal Manifest

## 1. Why V60B is needed after V60 limited

V60 showed that cross-year data exists and is mostly compatible, but the protocol was still limited because 2025 contains eval-related branches and 2026 contains Emu/new-setting taxonomy.
V60B converts that audit into a reproducible manifest lock: exactly which files enter V61-clean, and exactly which files are excluded.

## 2. Exact include/exclude rules

Included:

- 2023 train normal only
- 2024 train normal only
- 2025 current/dev train normal only
- metadata fields retained: year, path, machine_raw, machine_canonical, split, domain, section, duration, sample_rate

Excluded:

- all test split files
- all anomaly clips
- all 2025 eval_data_* branches
- all 2026 Emu / new-setting taxonomy files
- ambiguous or failed files

## 3. Per-year usable normal count

- 2023: `7000`
- 2024: `7000`
- 2025: `7700`
- 2026: `0`

## 4. Per-machine usable normal count

| year | machine | usable normal count |
|---|---|---:|
| 2023 | bearing | 1000 |
| 2023 | fan | 1000 |
| 2023 | gearbox | 1000 |
| 2023 | slider | 1000 |
| 2023 | toycar | 1000 |
| 2023 | toytrain | 1000 |
| 2023 | valve | 1000 |
| 2024 | bearing | 1000 |
| 2024 | fan | 1000 |
| 2024 | gearbox | 1000 |
| 2024 | slider | 1000 |
| 2024 | toycar | 1000 |
| 2024 | toytrain | 1000 |
| 2024 | valve | 1000 |
| 2025 | bearing | 1100 |
| 2025 | fan | 1100 |
| 2025 | gearbox | 1100 |
| 2025 | slider | 1100 |
| 2025 | toycar | 1100 |
| 2025 | toytrain | 1100 |
| 2025 | valve | 1100 |

## 5. Excluded count by reason

| reason | count |
|---|---:|
| exclude_2025_eval_data_branch | 10400 |
| exclude_2026_reserved_for_future_protocol | 2400 |
| exclude_anomaly_clip | 2100 |
| exclude_test_split | 2100 |

## 6. full_5view tensor feasibility on selected manifest

| year | selected count | tensor checked | tensor failed | full_5view feasible |
|---|---:|---:|---:|---|
| 2023 | 7000 | 7000 | 0 | True |
| 2024 | 7000 | 7000 | 0 | True |
| 2025 | 7700 | 7700 | 0 | True |
| 2026 | 0 | 0 | 0 | True |

## 7. Leakage risk assessment

| year | risk summary |
|---|---|
| 2023 | none |
| 2024 | none |
| 2025 | contains_eval_data_branches_excluded |
| 2026 | 2026_reserved_for_future_noise_aware_protocol |

## 8. Final verdict

`V60B limited`

Manifest exists, but manual cleaning / mapping is still required before training.

## 9. Confirmation that no training occurred

No model was trained. No checkpoint was modified. No anomaly label was used beyond exclusion filtering.
