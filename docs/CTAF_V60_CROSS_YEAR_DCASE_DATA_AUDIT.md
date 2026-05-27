# CTAF V60 Cross-Year DCASE Data Audit

## 1. Stage 13A-0 motivation

Stage 13A-0 checks only whether a multi-year normal-only data protocol is safe and unified enough to define.
It does not answer whether the protocol will improve performance.

## 2. Why this is not V39-V59 version search

- no model is trained
- no scoring rule is changed
- no frontend is changed
- only dataset schema, legality, overlap, and leakage risk are audited

## 3. Multi-year normal-only protocol definition

- Protocol A: current-year-only CTAF baseline
- Protocol B: multi-year normal-only CTAF pretraining corpus
- Protocol B may use only normal clips from allowed roots after strict filtering
- anomaly clips, anomaly labels, and test labels are excluded

## 4. Per-year dataset schema table

| year | root | n_machine_types | contains train | contains test | has source/target | has section metadata | sample rates | duration mean | structure |
|---|---|---:|---|---|---|---|---|---:|---|
| 2023 | /data1/test/LAES_2025/data/dcase2023_dev | 7 | True | True | True | True | 16000 | 12.0000 | train/test dev-style |
| 2024 | /data1/test/LAES_2025/data/dcase2024_dev | 7 | True | True | True | True | 16000 | 12.0000 | train/test dev-style |
| 2025 | /data1/test/LAES_2025/data/dcase | 15 | True | True | True | True | 16000 | 12.0000 | dev_current + eval_train_extra + eval_test_extra |
| 2026 | /data1/test/LAES_2025/data/dcase2026_dev | 2 | True | True | True | True | 16000 | 11.0000 | train/test dev-style |

## 5. Per-year normal/anomaly filtering table

| year | total normal-like | total anomaly | anomaly present | anomaly filterable | usable normal-only count | excluded count |
|---|---:|---:|---|---|---:|---:|
| 2023 | 7700 | 700 | True | True | 6930 | 1470 |
| 2024 | 7700 | 700 | True | True | 6930 | 1470 |
| 2025 | 15700 | 700 | True | True | 7630 | 11870 |
| 2026 | 2200 | 200 | True | True | 1980 | 420 |

## 6. Machine type mapping table

| year | machine | normal count | anomaly count | train count | test count | source count | target count | overlap with 2024 target | emu/new setting |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2023 | bearing | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2023 | fan | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2023 | gearbox | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2023 | slider | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2023 | toycar | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2023 | toytrain | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2023 | valve | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | bearing | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | fan | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | gearbox | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | slider | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | toycar | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | toytrain | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2024 | valve | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2025 | autotrash | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | bandsealer | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | bearing | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2025 | coffeegrinder | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | fan | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2025 | gearbox | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2025 | homecamera | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | polisher | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | screwfeeder | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | slider | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2025 | toycar | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2025 | toypet | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | toyrccar | 1000 | 0 | 1000 | 300 | 1290 | 10 | False | False |
| 2025 | toytrain | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2025 | valve | 1100 | 100 | 1000 | 300 | 1190 | 110 | True | False |
| 2026 | fan | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |
| 2026 | toycar | 1100 | 100 | 1000 | 200 | 1090 | 110 | True | False |

## 7. source/target/section metadata compatibility

| year | overlap with current 2024 target machines | all targets present | has source/target | has section metadata | machine taxonomy issue |
|---|---|---|---|---|---|
| 2023 | bearing,fan,gearbox,slider,toycar,toytrain,valve | True | True | True | False |
| 2024 | bearing,fan,gearbox,slider,toycar,toytrain,valve | True | True | True | False |
| 2025 | bearing,fan,gearbox,slider,toycar,toytrain,valve | True | True | True | False |
| 2026 | fan,toycar | False | True | True | False |

## 8. full_5view tensor feasibility

| year | tensor checks | success | failures | full_5view feasible |
|---|---:|---:|---:|---|
| 2023 | 20 | 20 | 0 | True |
| 2024 | 20 | 20 | 0 | True |
| 2025 | 20 | 20 | 0 | True |
| 2026 | 20 | 20 | 0 | True |

## 9. Data leakage risk assessment

| year | risk flags |
|---|---|
| 2023 | anomaly_clips_present_but_filterable |
| 2024 | anomaly_clips_present_but_filterable |
| 2025 | current_year_eval_trees_present;anomaly_clips_present_but_filterable |
| 2026 | anomaly_clips_present_but_filterable |

## 10. usable normal-only data count

- total usable normal-only files: `23470`

## 11. excluded data count and reason

- total excluded files: `15230`
- main reasons include:
  - anomaly clips present but filtered out
  - test splits excluded
  - 2025 eval_data_* branches treated as non-baseline external trees
  - 2026 `Emu` / new-setting machine taxonomy flagged separately

## 12. final verdict

`V60 limited`

Data exists but needs manual cleaning/mapping before training.

## 13. confirmation that no training occurred

No model was trained. No checkpoint was loaded for optimization. No anomaly label was used in training or model selection.

V60 only answers whether multi-year normal-only DCASE data can safely enter a unified CTAF protocol.
