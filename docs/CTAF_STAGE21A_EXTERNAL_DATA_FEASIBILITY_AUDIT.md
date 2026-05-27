# CTAF Stage21A External Normal-Only Data Feasibility Audit

## Goal

Find official, public, machine-audio datasets that can support future normal-only pretraining.

## Bottom line

- Best first dataset: `DCASE 2022 Challenge Task 2 Additional Training Dataset`
- Best use: train on source-normal and target-normal only
- Why: exact machine-family match to the current benchmark and clean normal-only training splits

## Safe candidates

- DCASE 2022 Challenge Task 2 Additional Training Dataset
- DCASE 2022 Challenge Task 2 Development Dataset
- DCASE 2025 Challenge Task 2 Development Dataset
- DCASE 2025 Challenge Task 2 Additional Training Dataset
- DCASE 2024 Challenge Task 2 Development Dataset
- DCASE 2024 Challenge Task 2 Additional Training Dataset
- DCASE 2023 Challenge Task 2 Development Dataset
- DCASE 2023 Challenge Task 2 Additional Training Dataset
- DCASE 2021 Challenge Task 2 Development Dataset
- DCASE 2021 Challenge Task 2 Additional Training Dataset
- DCASE 2020 Challenge Task 2 Development Dataset
- DCASE 2020 Challenge Task 2 Additional Training Dataset

## Rejected or high risk

- ToyADMOS
- ToyADMOS2
- MIMII Dataset
- MIMII DUE dataset
- MIMII DG dataset

## Rule note

The current DCASE 2025 rule set allows prior DCASE training/additional-training data if test/evaluation audio is excluded, but explicitly forbids the ToyADMOS/MIMII-family datasets listed above.

