# CTAF V61 Cross-Year Original Encoder Dry Run

## 1. Scope

One-machine dry run only. This step validates pipeline compatibility for
`cross-year pretrain -> current-year adapt -> V37 latent_summary_knn eval`.

## 2. Manifest and Current-Year Root

- manifest: `results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv`
- current target-year data root: `/data1/test/LAES_2025/data/dcase`
- machine: `gearbox`
- seed: `13711`

## 3. Included / Excluded Confirmation

- manifest selected rows for `gearbox`: `3100`
- per-year counts: `{"2023": 1000, "2024": 1000, "2025": 1100}`
- per-domain counts: `{"source": 3070, "target": 30}`
- 2025 eval_data_* branches: excluded by V60B manifest
- 2026 Emu / noise-aware branches: excluded by V60B manifest
- anomaly clips: excluded by V60B manifest
- test files: excluded from cross-year pretraining manifest

## 4. Phase Status

- Phase 1 pretrain: success, epochs=50, best_val=0.157531
- Phase 2 teacher adaptation: success, epochs=50, best_val=0.161480
- Phase 2 student distillation: success, epochs=50, best_val=0.054319

## 5. Checkpoints and Scores

- phase1 teacher checkpoint: `results/v61_cross_year_original_encoder_dryrun/models/gearbox_phase1_teacher_seed13711.pt`
- phase2 teacher checkpoint: `results/v61_cross_year_original_encoder_dryrun/models/gearbox_phase2_teacher_seed13711.pt`
- phase2 student checkpoint: `results/v61_cross_year_original_encoder_dryrun/models/gearbox_phase2_student_seed13711.pt`
- score file: `results/v61_cross_year_original_encoder_dryrun/scores/gearbox_full_5view_latent_summary_knn_scores.npz`

## 6. Gearbox Dry-Run Metrics

- hmean: `0.535612`
- pAUC: `0.518947`
- KS: `0.160000`
- q95: `0.111547`
- q99: `0.120759`
- TPR@95n: `0.080000`
- TPR@99n: `0.050000`

## 7. V37 Gearbox Baseline Comparison

- baseline hmean: `0.522195`
- baseline pAUC: `0.520000`
- baseline KS: `0.100000`
- baseline q95: `0.117578`
- baseline q99: `0.129624`
- baseline TPR@95n: `0.100000`
- baseline TPR@99n: `0.060000`
- delta hmean: `0.013417`
- delta pAUC: `-0.001053`
- delta KS: `0.060000`

## 8. Final Verdict

`dryrun pass`

Pipeline is compatible with the V61 prereg logic on one-machine dry run.
This verdict does not change protocol or authorize V62.

## 9. Confirmation

- no encoder architecture change
- no frontend change
- no loss change beyond prereg V37 pretrain/adapt logic
- no anomaly label used in training
- no test label used in training
- no 2025 eval_data_* used
- no 2026 used
- no V62 opened
