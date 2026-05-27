# CTAF V62A Strong Encoder Dryrun

## 1. V62A Engineering Status

V62A only implemented strong-encoder functionality and minimal engineering sanity.
No full V62 training was run.

Current status:

- `shape_check`: `PASS`
- parameter count: `1,639,413`
- latent_summary dim: `384`
- gearbox `dryrun_train`: completed
- checkpoint files generated
- score file generated

Shape-check reference:

- input shape: `[2, 5, 128, 313]`
- latent map shape: `[2, 128, 32, 79]`
- output shape: `[2, 5, 128, 313]`
- latent_summary shape: `[2, 384]`

Dryrun output paths:

- `results/v62a_strong_encoder_dryrun/models/gearbox_phase1_teacher_seed13711.pt`
- `results/v62a_strong_encoder_dryrun/models/gearbox_phase2_teacher_seed13711.pt`
- `results/v62a_strong_encoder_dryrun/models/gearbox_phase2_student_seed13711.pt`
- `results/v62a_strong_encoder_dryrun/scores/gearbox_full_5view_latent_summary_knn_scores.npz`

## 2. Gearbox Comparison

| Variant | hmean | pAUC | q99 | TPR@99n |
|---|---:|---:|---:|---:|
| V37 | 0.5222 | 0.5200 | 0.1296 | 0.0600 |
| V61 | 0.5356 | 0.5189 | 0.1208 | 0.0500 |
| V62A | 0.4963 | 0.5184 | 0.0481 | 0.0300 |

Additional V62A tail values:

- `q95 = 0.0422`
- `anomaly_median = 0.0335`
- `anomaly_q75 = 0.0377`
- `KS = 0.18`

## 3. Mechanistic Interpretation

The strong encoder clearly compresses normal scores:

- `V37 q99 = 0.1296`
- `V61 q99 = 0.1208`
- `V62A q99 = 0.0481`

However, anomaly scores are also compressed:

- `anomaly_median = 0.0335`
- `anomaly_q75 = 0.0377`

Both are below:

- `normal q99 = 0.0481`

So the result is not improved low-FPR separation.
It is stronger score compression without corresponding anomaly-tail lift.

Therefore:

- normal scores move down
- anomaly scores also move down
- `TPR@99n` drops from `0.0500` in V61 to `0.0300` in V62A
- low-FPR behavior is worse, not better

## 4. Final Verdict

- `V62A implementation sanity: PASS`
- `V62A method dryrun: FAIL / early-stop`

Decision:

- do **not** run full 7-machine V62
- do **not** open V63
- do **not** tune architecture

This dryrun is sufficient to conclude that the current V62A Conformer-style
strong encoder does not justify a full single-seed 7-machine execution.

## 5. Confirmation

- no protocol change
- no data leakage
- no anomaly/test label use in training
- no BEATs
- no UACS
- no score fusion
