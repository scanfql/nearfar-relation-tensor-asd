# CTAF V1 Full-Run Analysis and V2 Protocol

Last updated: 2026-05-12

## 1. Scope

This document records the first full CTAF result and defines the next protocol-safe
development step.

CTAF remains a clean label-free project. It is not V27, not a route-search
pipeline, and not a supervised score-fusion system.

The controlling project plan is:

```text
NEW_PROJECT_CLEAN_TENSOR_FRONTEND_PLAN_FOR_CODEX.md
```

Its constraints remain binding:

- no reverse score selection
- no anomaly-label-driven fusion
- no machine-specific tuning
- no official-score-driven model or route selection
- anomaly labels are used only after scores are fixed

## 2. V1 Full-Run Setting

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/full_7machines_4groups
```

Dataset protocol:

- 7 machines: bearing, fan, gearbox, slider, toycar, toytrain, valve
- training: source-domain normal data only
- evaluation: fixed test split after all scores are generated
- per machine test split: source normal 50, source anomaly 50, target normal 50, target anomaly 50

## 3. V1 Full-Run Summary

| Experiment | Views | Mean source AUC | Mean target AUC | Mean pAUC | Mean hmean | Mean DCASE AUC | Mean source-target KS | Improved vs logmel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logmel_only | 1 | 0.482000 | 0.541429 | 0.509398 | 0.502470 | 0.511714 | 0.3457 | 0 |
| dynamic_3view | 3 | 0.474400 | 0.527943 | 0.503534 | 0.491669 | 0.501171 | 0.2914 | 3 |
| edge_3view | 3 | 0.482000 | 0.502914 | 0.505113 | 0.483899 | 0.492457 | 0.3314 | 2 |
| full_5view | 5 | 0.462057 | 0.545714 | 0.501654 | 0.493440 | 0.503886 | 0.2800 | 3 |

## 4. V1 Decision

V1 fails the pre-registered minimum success criterion.

Required:

```text
full_5view hmean > logmel_only hmean
and
full_5view pAUC >= logmel_only pAUC
```

Observed:

```text
full_5view hmean = 0.493440 < logmel_only hmean = 0.502470
full_5view pAUC  = 0.501654 < logmel_only pAUC  = 0.509398
```

Therefore, CTAF V1 must be reported as a clean negative result:

```text
Naive full multi-view tensor stacking does not improve over the clean log-Mel
autoencoder baseline under the 50-epoch label-free protocol.
```

## 5. Useful Signal Retained From V1

V1 is not useless. It shows a consistent reliability clue:

```text
full_5view source-target normal KS = 0.2800
logmel_only source-target normal KS = 0.3457
```

The multi-view tensor representation reduces source-target normal score drift,
but this drift improvement does not translate into better anomaly discrimination.

This becomes the V2 research question:

```text
Can CTAF keep the source-target stability benefit while recovering anomaly
separation, without using anomaly labels for score direction, score choice, or
machine-specific tuning?
```

## 6. V1 Failure Pattern

Per-machine hmean deltas versus logmel_only:

| Machine | dynamic_3view | edge_3view | full_5view | Main observation |
|---|---:|---:|---:|---|
| bearing | -0.029013 | -0.076100 | -0.074766 | multi-view hurts a strong logmel baseline |
| fan | -0.026894 | -0.041858 | -0.011933 | full is closest but still below logmel |
| gearbox | -0.019957 | -0.008914 | -0.000005 | full nearly ties logmel |
| slider | +0.011466 | -0.051449 | -0.026336 | dynamic helps; edge/full hurt target behavior |
| toycar | +0.020034 | +0.033751 | +0.013103 | all multi-view variants help |
| toytrain | -0.033046 | +0.071797 | +0.025478 | edge is very strong |
| valve | +0.001800 | -0.057228 | +0.011250 | full helps weakly |

The failure is not uniform. It is not correct to say "multi-view has no signal."
The correct statement is:

```text
The current V1 aggregation and reconstruction score do not convert multi-view
structure into stable discrimination across machines.
```

## 7. V2 Protocol-Safe Direction

V2 should not add supervised search. The next allowed change is:

```text
view-wise residual normalization and fixed aggregation
```

The V1 score is:

```text
score = mean residual over all views, frequency bins, and time frames
```

This can let high-variance or low-value views dominate the score. V2 should
compute per-view residuals first:

```text
e_v(x) = mean((X_v - Xhat_v)^2)
```

Then normalize each view using train-normal residual statistics only:

```text
z_v(x) = (e_v(x) - median_train(e_v)) / (IQR_train(e_v) + eps)
```

The main V2 score must use a fixed pre-registered aggregation rule. Allowed
first candidates:

```text
score_view_mean = mean_v z_v
score_view_top2 = mean of the largest 2 z_v values
```

Important: these are not selected per machine. If both are evaluated, they must
be treated as fixed experiment groups and reported honestly, not chosen by
which one wins a machine.

## 8. V2 Forbidden Items

Do not implement:

- `score` versus `-score` selection
- per-machine best experiment selection
- per-machine view subset selection
- AUC/Hmean/pAUC-driven score aggregation search
- machine-specific model architecture
- machine-specific normalization
- route search or fusion weight search

If a score rule is added, it must be globally defined before looking at its
evaluation metrics.

## 9. Proposed V2 Experiment Groups

Keep V1 groups for comparability:

```text
logmel_only
dynamic_3view
edge_3view
full_5view
```

Add fixed score variants only if implemented globally:

```text
mean_mse             # V1 baseline score
view_z_mean          # train-normal view-wise robust residual aggregation
view_z_top2          # train-normal robust top-2 view aggregation
```

The main V2 comparison should be:

```text
logmel_only / mean_mse
vs
full_5view / view_z_mean
vs
edge_3view / view_z_mean
```

`view_z_top2` is diagnostic unless pre-registered as a main V2 score before
running the full experiment.

## 10. V2 Success Criteria

The 0.65 target remains aspirational. The immediate V2 gate is:

```text
mean_hmean > 0.502470
mean_pauc  >= 0.509398
source-target KS not worse than logmel_only
at least 4/7 machines improve over logmel_only
```

The long-run target remains:

```text
mean_hmean >= 0.650000
```

If V2 improves stability but not hmean, it is a diagnostic success but not a
method success.

## 11. Immediate Next Task

Before implementing advanced models, implement score diagnostics that expose:

- per-view train residual median/IQR
- per-view test residual distributions
- view-wise normalized scores
- fixed aggregated scores

This is the next protocol-safe step because it tests whether V1 failed due to
score aggregation rather than because the tensor frontend itself has no signal.

Recommended V2 debug command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 3 --batch_size 16 --seed 13711 --device cuda:0 --output_dir results/v2_debug_view_scores --score_modes mean_mse,view_z_mean,view_z_top2
```

Recommended V2 full command only after debug review:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v2_full_view_scores --score_modes mean_mse,view_z_mean,view_z_top2
```

## 12. V3 Protocol-Safe Training Checkpoint

The V2 full run showed that view-wise residual aggregation improves the
full_5view score over V1 but does not beat logmel_only after 50 epochs.

The next allowed training-side change is:

```text
restore the checkpoint with the best normal-validation reconstruction loss
```

This is protocol-safe because:

- the validation split comes only from train-normal data
- anomaly labels are not used to choose the epoch
- AUC, pAUC, Hmean, and official score are not used for checkpoint selection
- the same patience and min_delta apply to every machine and experiment

This is not reverse selection, route search, or machine-specific tuning.

Recommended V3 debug command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v3_debug_earlystop_view_scores --score_modes mean_mse,view_z_mean,view_z_top2 --early_stopping --patience 8 --min_delta 0.0
```

If V3 debug passes the immediate gate, run the same command with a final output
directory:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v3_full_earlystop_view_scores --score_modes mean_mse,view_z_mean,view_z_top2 --early_stopping --patience 8 --min_delta 0.0
```

## 13. V4 Protocol-Safe Denoising Training

The V3 run showed that normal-validation checkpointing does not materially
change the result: best epochs are mostly near the final epoch. The next
protocol-safe change is to make the autoencoder learn normal structure rather
than simply copy inputs.

Allowed V4 change:

```text
train-time random time-frequency masking with clean tensor reconstruction target
```

Implementation rule:

```text
x_input = mask(x_normal_train)
target  = x_normal_train
score input at test time = clean x_test
```

This remains label-free because:

- corruption is applied only to normal training tensors
- the target is the same normal tensor
- anomaly labels are not used
- mask probability is a global pre-registered parameter
- scoring still uses fixed reconstruction residuals and fixed score modes

Forbidden V4 variants:

- choosing mask probability by AUC/pAUC/Hmean
- choosing different mask probabilities per machine
- masking only selected views per machine
- reversing scores after seeing metrics

Recommended V4 debug command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v4_debug_tfmask015_view_scores --score_modes mean_mse,view_z_mean,view_z_top2 --noise_mode tf_mask --mask_prob 0.15
```

If V4 debug improves over `logmel_only/mean_mse` on hmean and pAUC, run the
same command with:

```text
--output_dir results/v4_full_tfmask015_view_scores
```

## 14. V5 Protocol-Safe Latent Memory Score

V4 improved pAUC and source-target stability but still did not beat
`logmel_only/mean_mse` on hmean. The next allowed score-side change is a fixed
train-normal latent memory distance.

V5 score:

```text
1. Train the same Conv2D AE on train-normal tensors.
2. Extract encoder latent vectors for train-normal tensors.
3. Extract encoder latent vectors for test tensors.
4. score = mean distance to k nearest train-normal latent vectors.
5. Larger distance = more anomalous.
```

Protocol constraints:

- train-normal data only builds the memory bank
- anomaly labels are not used
- k is fixed globally before evaluation
- no per-machine k
- no reverse score direction
- no choice between latent and reconstruction score per machine

Recommended fixed value:

```text
k = 5
```

Recommended V5 debug command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v5_debug_latent_knn --score_modes mean_mse,view_z_top2,latent_knn --latent_knn_k 5
```

If latent memory scoring improves over `logmel_only/mean_mse`, run the same
command with:

```text
--output_dir results/v5_full_latent_knn
```

## 15. Current Best Result: V5 Full Latent Memory

V5 full run completed with the same command as V5 debug, using:

```text
output_dir = results/v5_full_latent_knn
score_modes = mean_mse,view_z_top2,latent_knn
latent_knn_k = 5
noise_mode = none
mask_prob = 0.0
```

Main comparison:

| Method | Mean source AUC | Mean target AUC | Mean pAUC | Mean hmean | Mean DCASE AUC | Mean source-target KS | Improved machines |
|---|---:|---:|---:|---:|---:|---:|---:|
| logmel_only / mean_mse | 0.481257 | 0.544457 | 0.509850 | 0.502846 | 0.512857 | 0.3371 | 0 |
| full_5view / latent_knn | 0.550171 | 0.540057 | 0.516617 | 0.526239 | 0.545114 | 0.3171 | 6 |

Observed V5 gains:

```text
hmean: +0.023393
pAUC : +0.006767
KS   : -0.0200
```

V5 is the first CTAF variant that passes the immediate clean gate:

```text
mean_hmean > logmel_only/mean_mse
mean_pauc  >= logmel_only/mean_mse
source-target KS not worse than logmel_only/mean_mse
at least 4/7 machines improve
```

Current best method statement:

```text
CTAF V5 = full_5view tensor frontend + Conv2D AE encoder latent representation
          + fixed train-normal kNN memory distance, k=5.
```

This remains label-free because:

- the AE is trained only on train-normal source-domain data
- the latent memory bank is built only from train-normal samples
- k is globally fixed at 5
- larger latent distance is the pre-registered anomaly direction
- no reverse selection, score-fusion search, or machine-specific route selection is used

## 16. V5 Per-Machine Pattern

V5 full_5view/latent_knn hmean by machine:

| Machine | logmel_only/mean_mse hmean | full_5view/latent_knn hmean | Delta |
|---|---:|---:|---:|
| bearing | 0.569549 | 0.592235 | +0.022686 |
| fan | 0.568844 | 0.586757 | +0.017914 |
| gearbox | 0.549074 | 0.557924 | +0.008850 |
| slider | 0.511285 | 0.442804 | -0.068481 |
| toycar | 0.459461 | 0.517523 | +0.058062 |
| toytrain | 0.543906 | 0.592536 | +0.048631 |
| valve | 0.317803 | 0.393894 | +0.076091 |

Interpretation:

```text
V5 improves 6/7 machines. The dominant failure is slider.
```

Do not fix slider with a machine-specific route. The next step must use a
global rule that applies to every machine.

## 17. Current Gap to Target

The project target remains:

```text
mean_hmean >= 0.650000
```

Current best:

```text
mean_hmean = 0.526239
gap        = 0.123761
```

V5 is a valid clean method candidate, but it is not close enough to the final
target. Further work is justified.

## 18. Next Step: V6 tf_mask + latent_knn

Rationale:

- V4 showed `tf_mask` improves pAUC and source-target drift, but reconstruction
  scores still did not beat logmel hmean.
- V5 showed `latent_knn` is much stronger than reconstruction MSE.
- V6 combines the two protocol-safe ideas:

```text
train with tf_mask denoising
score with fixed train-normal latent_knn
```

This remains label-free if `mask_prob=0.15` and `latent_knn_k=5` remain fixed
global parameters.

Run next:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v6_tfmask015_latent_knn --score_modes mean_mse,view_z_top2,latent_knn --latent_knn_k 5 --noise_mode tf_mask --mask_prob 0.15
```

V6 decision checks:

```text
full_5view/latent_knn hmean > 0.526239
full_5view/latent_knn pAUC  >= 0.516617
slider full_5view/latent_knn improves over 0.442804
source-target KS not worse than 0.3171
```

Recommended summary commands after V6 finishes:

```bash
cat results/v6_tfmask015_latent_knn/summary.csv
grep "full_5view,latent_knn" results/v6_tfmask015_latent_knn/per_machine.csv
grep "latent_knn" results/v6_tfmask015_latent_knn/score_geometry.csv
```

## 19. Handoff Rule for Future Conversations

For a new conversation, read this file first:

```text
new_projects/clean_tensor_asd/docs/CTAF_V1_FULL_RUN_ANALYSIS_AND_V2_PROTOCOL.md
```

Then continue from the last numbered section. When a new version finishes,
append:

1. command used
2. `summary.csv` key rows
3. best method and baseline comparison
4. per-machine failure pattern
5. next protocol-safe step

Do not start from old LTFD/V24/V25/V26 route-search logic. CTAF remains governed
by `NEW_PROJECT_CLEAN_TENSOR_FRONTEND_PLAN_FOR_CODEX.md`.

## 20. V6 Result: tf_mask + latent_knn

V6 command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v6_tfmask015_latent_knn --score_modes mean_mse,view_z_top2,latent_knn --latent_knn_k 5 --noise_mode tf_mask --mask_prob 0.15
```

Main comparison inside V6:

| Method | Mean source AUC | Mean target AUC | Mean pAUC | Mean hmean | Mean DCASE AUC | Mean source-target KS | Improved machines |
|---|---:|---:|---:|---:|---:|---:|---:|
| logmel_only / mean_mse | 0.502171 | 0.542257 | 0.510752 | 0.509246 | 0.522214 | 0.3286 | 0 |
| full_5view / latent_knn | 0.560800 | 0.537257 | 0.504737 | 0.524280 | 0.549029 | 0.2714 | 4 |

V6 compared with the current V5 best:

| Method | Mean pAUC | Mean hmean | Mean source-target KS |
|---|---:|---:|---:|
| V5 full_5view / latent_knn | 0.516617 | 0.526239 | 0.3171 |
| V6 full_5view / latent_knn | 0.504737 | 0.524280 | 0.2714 |

Decision:

```text
V6 does not replace V5 as the best method.
```

V6 decision checks:

```text
full_5view/latent_knn hmean > 0.526239      FAIL  (0.524280)
full_5view/latent_knn pAUC  >= 0.516617     FAIL  (0.504737)
slider improves over 0.442804               PASS  (0.454131)
source-target KS not worse than 0.3171      PASS  (0.2714)
```

Interpretation:

```text
tf_mask improves source-target stability and partially repairs slider, but it
reduces pAUC and does not improve overall hmean. V5 remains the current best.
```

Per-machine V6 full_5view/latent_knn hmean:

| Machine | V5 full_5view/latent_knn | V6 full_5view/latent_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.592235 | 0.593889 | +0.001654 |
| fan | 0.586757 | 0.554151 | -0.032607 |
| gearbox | 0.557924 | 0.551734 | -0.006190 |
| slider | 0.442804 | 0.454131 | +0.011326 |
| toycar | 0.517523 | 0.518119 | +0.000596 |
| toytrain | 0.592536 | 0.593269 | +0.000732 |
| valve | 0.393894 | 0.404669 | +0.010775 |

The main V6 damage is fan pAUC/hmean; the main benefit is lower source-target
drift.

## 21. Current Best After V6

Keep V5 as the current best:

```text
CTAF V5 = full_5view tensor frontend + Conv2D AE latent vectors
          + train-normal latent kNN memory score, k=5
```

Current best metrics:

```text
mean_hmean = 0.526239
mean_pAUC  = 0.516617
mean_KS    = 0.3171
```

V6 is useful as a diagnostic but not as the main method.

## 22. Next Protocol-Safe Direction: V7

The current bottleneck is no longer the raw tensor frontend. The evidence so far:

- reconstruction MSE is weak
- view-wise residual scoring helps only modestly
- latent kNN memory distance is clearly stronger
- tf_mask improves drift but hurts pAUC

The next protocol-safe step should improve the latent normal-memory score while
keeping a fixed, label-free rule.

Recommended V7 direction:

```text
latent_knn on pooled encoder statistics
```

Instead of flattening the entire latent map, compute fixed latent summary
features:

```text
latent_mean over spatial positions
latent_std over spatial positions
latent_max over spatial positions
feature = concat(mean, std, max)
```

Then score exactly as in V5:

```text
score = mean distance to k=5 nearest train-normal latent-summary vectors
larger distance = more anomalous
```

Why this is protocol-safe:

- the summary rule is global and fixed
- the memory bank still uses train-normal samples only
- k remains fixed at 5
- no machine-specific selection
- no reverse direction
- no AUC/pAUC/Hmean-driven route choice

Rationale:

```text
Flattened latent maps may encode nuisance time-position details and domain
shift. Pooled latent statistics may preserve machine-state structure while
reducing position/domain sensitivity.
```

Do not tune pooling variants per machine. If multiple latent summary modes are
added, report each as a fixed score mode.

Implemented score mode name:

```text
latent_summary_knn
```

V7 command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v7_latent_summary_knn --score_modes mean_mse,latent_knn,latent_summary_knn --latent_knn_k 5
```

V7 decision checks:

```text
full_5view/latent_summary_knn hmean > 0.526239
full_5view/latent_summary_knn pAUC  >= 0.516617
slider full_5view/latent_summary_knn improves over 0.442804
source-target KS not worse than 0.3171
```

Recommended summary commands:

```bash
cat results/v7_latent_summary_knn/summary.csv
grep "full_5view,latent_summary_knn" results/v7_latent_summary_knn/per_machine.csv
grep "latent_summary_knn" results/v7_latent_summary_knn/score_geometry.csv
```

## 23. V7 Result: latent_summary_knn

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v7_latent_summary_knn --score_modes mean_mse,latent_knn,latent_summary_knn --latent_knn_k 5
```

Main summary rows:

| Experiment / score | Mean source AUC | Mean target AUC | Mean pAUC | Mean hmean | Mean DCASE AUC | Mean source-target KS | Improved vs logmel |
|---|---:|---:|---:|---:|---:|---:|---:|
| logmel_only / mean_mse | 0.476229 | 0.540171 | 0.508947 | 0.499804 | 0.508200 | 0.3371 | 0 |
| full_5view / latent_knn | 0.555143 | 0.537829 | 0.516165 | 0.526727 | 0.546486 | 0.3057 | 6 |
| full_5view / latent_summary_knn | 0.540514 | 0.552057 | 0.529398 | 0.536338 | 0.546286 | 0.2829 | 6 |
| edge_3view / latent_summary_knn | 0.590400 | 0.558057 | 0.538421 | 0.552497 | 0.574229 | 0.3486 | 6 |

V7 decision checks for the pre-registered main target
`full_5view / latent_summary_knn`:

```text
full_5view/latent_summary_knn hmean > 0.526239      PASS  (0.536338)
full_5view/latent_summary_knn pAUC  >= 0.516617     PASS  (0.529398)
slider full_5view/latent_summary_knn > 0.442804     PASS  (0.490213)
source-target KS not worse than 0.3171              PASS  (0.2829)
```

Verdict:

```text
V7 passes all pre-registered decision checks.
```

The pooled latent summary score does what V7 was supposed to test:

- it improves full_5view hmean over V5
- it improves full_5view pAUC over V5
- it further reduces source-target normal drift
- it keeps the score fully label-free and globally fixed

Compared with the frozen V5 reference:

```text
V5 full_5view/latent_knn:
mean_hmean = 0.526239
mean_pAUC  = 0.516617
mean_KS    = 0.3171

V7 full_5view/latent_summary_knn:
mean_hmean = 0.536338   delta = +0.010099
mean_pAUC  = 0.529398   delta = +0.012781
mean_KS    = 0.2829     delta = -0.0342
```

## 24. Per-Machine V7 vs V5

`full_5view / latent_summary_knn` hmean versus the frozen
`V5 full_5view / latent_knn` mainline:

| Machine | V5 full_5view / latent_knn hmean | V7 full_5view / latent_summary_knn hmean | Delta |
|---|---:|---:|---:|
| bearing | 0.592235 | 0.597747 | +0.005512 |
| fan | 0.586757 | 0.562146 | -0.024611 |
| gearbox | 0.557924 | 0.549726 | -0.008198 |
| slider | 0.442804 | 0.490213 | +0.047409 |
| toycar | 0.517523 | 0.526583 | +0.009060 |
| toytrain | 0.592536 | 0.543640 | -0.048896 |
| valve | 0.393894 | 0.484312 | +0.090418 |

Interpretation:

- V7 improves 5/7 machines versus the frozen V5 mainline
- the largest gains are `valve` and `slider`
- the main regressions are `toytrain` and `fan`
- versus `logmel_only / mean_mse`, V7 full_5view still improves 6/7 machines

## 25. Current Best After V7

Update the full_5view mainline to:

```text
CTAF V7 = full_5view tensor frontend
          + Conv2D AE latent summary vectors (mean/std/max)
          + train-normal latent kNN memory score, k=5
```

Current mainline metrics:

```text
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829
```

This is the first CTAF full_5view result that clearly beats the frozen V5
latent-kNN mainline on both discrimination and drift.

Important side result:

```text
edge_3view / latent_summary_knn reaches
mean_hmean = 0.552497
mean_pAUC  = 0.538421
mean_KS    = 0.3486
```

So the strongest raw discrimination in the fixed V7 grid comes from
`edge_3view / latent_summary_knn`, but its source-target KS is worse than both
`logmel_only / mean_mse` and `full_5view / latent_summary_knn`.

Therefore the cleanest CTAF conclusion after V7 is:

```text
pooled latent summary kNN is a real improvement over flattened latent kNN,
and full_5view remains the best balanced mainline when discrimination and
source-target stability are considered together.
```

## 26. V8 Result: latent_summary_robust_knn

V8 tested a stronger train-normal normalization rule on the same frozen V7
latent-summary vectors:

```text
fit per-dimension train-normal median/IQR
robust-z every latent-summary dimension
then run the same fixed k=5 latent kNN
```

Why this was reasonable:

- `mean/std/max` summary blocks have very different scales
- V7 still uses only global L2 normalization
- a train-normal feature normalization is label-free and globally defined

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_robust_knn | 0.505564 | 0.520744 | 0.2571 |
| full_5view / latent_summary_robust_knn | 0.496241 | 0.499776 | 0.2629 |

Verdict:

```text
V8 is a negative result.
```

Interpretation:

- per-dimension robust normalization is too aggressive
- it removes useful absolute latent magnitude structure
- it hurts both logmel_only and full_5view discrimination
- although KS improves, the anomaly separation collapses

So V8 must not replace V7.

## 27. V9 Result: latent_summary_blockscale_knn

Because V8 over-normalized, V9 tested a softer fixed rule:

```text
split latent summary into 3 blocks: mean | std | max
fit one train-normal robust scale per block
scale blocks only, without per-dimension centering
then run the same fixed k=5 latent kNN
```

This preserves within-block magnitude structure while reducing domination by
the large-scale `max` block.

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_blockscale_knn | 0.535188 | 0.554940 | 0.4629 |
| full_5view / latent_summary_blockscale_knn | 0.533158 | 0.545104 | 0.3743 |

Important comparison:

```text
V7 full_5view / latent_summary_knn:
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829

V9 full_5view / latent_summary_blockscale_knn:
mean_hmean = 0.545104
mean_pAUC  = 0.533158
mean_KS    = 0.3743
```

So V9 raises raw `full_5view` discrimination above V7, but two things prevent
it from becoming the new CTAF mainline:

1. It worsens source-target drift substantially.
2. Under the same V9 score rule, `logmel_only` is still better than `full_5view`:

```text
logmel_only / latent_summary_blockscale_knn hmean = 0.554940
full_5view / latent_summary_blockscale_knn  hmean = 0.545104
```

Therefore V9 is not evidence that the tensor frontend beats the clean logmel
baseline under this score mode.

Verdict:

```text
V9 is a useful discrimination-side diagnostic, but not a valid replacement
for the V7 full_5view mainline.
```

Current status after V9:

```text
Best balanced CTAF mainline:
  V7 full_5view / latent_summary_knn

Best raw full_5view hmean so far:
  V9 full_5view / latent_summary_blockscale_knn
  but with worse KS and no win over same-mode logmel_only
```

## 28. Next Run Started: V10 tf_mask + latent_summary Family

The most reasonable next step after V7/V8/V9 is to combine:

- V6's lower source-target drift from `tf_mask`
- V7's balanced pooled latent-summary kNN
- V9's stronger block-scaled latent-summary discrimination

So the next full server run has been started with one fixed global setting:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v10_tfmask015_latent_summary --score_modes mean_mse,latent_knn,latent_summary_knn,latent_summary_blockscale_knn --latent_knn_k 5 --noise_mode tf_mask --mask_prob 0.15
```

V10 purpose:

```text
test whether tf_mask can recover the drift damage of V9
while preserving the stronger latent-summary discrimination
```

## 29. V10 Result: tf_mask + latent_summary Family

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,dynamic_3view,edge_3view,full_5view --epochs 50 --batch_size 32 --seed 13711 --device cuda:0 --output_dir results/v10_tfmask015_latent_summary --score_modes mean_mse,latent_knn,latent_summary_knn,latent_summary_blockscale_knn --latent_knn_k 5 --noise_mode tf_mask --mask_prob 0.15
```

Main summary rows:

| Experiment / score | Mean source AUC | Mean target AUC | Mean pAUC | Mean hmean | Mean DCASE AUC | Mean source-target KS | Improved vs logmel |
|---|---:|---:|---:|---:|---:|---:|---:|
| logmel_only / mean_mse | 0.504514 | 0.545343 | 0.509323 | 0.511402 | 0.524929 | 0.3229 | 0 |
| logmel_only / latent_summary_knn | 0.582457 | 0.567829 | 0.534286 | 0.553611 | 0.575143 | 0.3514 | 4 |
| logmel_only / latent_summary_blockscale_knn | 0.595943 | 0.570286 | 0.528421 | 0.561523 | 0.583114 | 0.4314 | 6 |
| full_5view / latent_summary_knn | 0.535600 | 0.551543 | 0.518872 | 0.528971 | 0.543571 | 0.3314 | 4 |
| full_5view / latent_summary_blockscale_knn | 0.534514 | 0.539829 | 0.532105 | 0.525830 | 0.537171 | 0.3971 | 6 |
| edge_3view / latent_summary_blockscale_knn | 0.595657 | 0.567486 | 0.532707 | 0.560101 | 0.581571 | 0.4657 | 7 |

Main V10 takeaways:

- `tf_mask + latent_summary_knn` does **not** beat the frozen V7 mainline
- `tf_mask + latent_summary_blockscale_knn` does **not** beat the V9 raw
  full_5view blockscale result
- `edge_3view / latent_summary_blockscale_knn` reaches the highest V10 hmean,
  but at the cost of very poor source-target KS

Direct comparison against the frozen V7 mainline:

```text
V7 full_5view / latent_summary_knn:
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829

V10 full_5view / latent_summary_knn:
mean_hmean = 0.528971   delta = -0.007367
mean_pAUC  = 0.518872   delta = -0.010526
mean_KS    = 0.3314     delta = +0.0485
```

So the intended V10 goal fails:

```text
tf_mask does not improve the pooled latent-summary mainline.
```

Per-machine `full_5view / latent_summary_knn` hmean versus V7:

| Machine | V7 full_5view / latent_summary_knn | V10 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.597747 | 0.528899 | -0.068847 |
| fan | 0.562146 | 0.546212 | -0.015935 |
| gearbox | 0.549726 | 0.539339 | -0.010387 |
| slider | 0.490213 | 0.494667 | +0.004454 |
| toycar | 0.526583 | 0.548415 | +0.021832 |
| toytrain | 0.543640 | 0.552814 | +0.009174 |
| valve | 0.484312 | 0.492450 | +0.008138 |

Interpretation:

- `tf_mask` helps `slider`, `toycar`, `toytrain`, and `valve`
- but it hurts `bearing` badly and also weakens `fan` and `gearbox`
- the large `bearing` regression is enough to destroy the V7 gain

Direct comparison for the block-scale branch:

```text
V9 full_5view / latent_summary_blockscale_knn:
mean_hmean = 0.545104
mean_pAUC  = 0.533158
mean_KS    = 0.3743

V10 full_5view / latent_summary_blockscale_knn:
mean_hmean = 0.525830   delta = -0.019274
mean_pAUC  = 0.532105   delta = -0.001053
mean_KS    = 0.3971     delta = +0.0228
```

So `tf_mask` also fails to rescue the block-scale branch. It does not improve
drift, and it loses hmean.

Per-machine `full_5view / latent_summary_blockscale_knn` hmean versus V9:

| Machine | V9 full_5view / latent_summary_blockscale_knn | V10 full_5view / latent_summary_blockscale_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.619224 | 0.551723 | -0.067501 |
| fan | 0.592422 | 0.560385 | -0.032037 |
| gearbox | 0.593550 | 0.598435 | +0.004886 |
| slider | 0.511784 | 0.504408 | -0.007377 |
| toycar | 0.553362 | 0.524385 | -0.028977 |
| toytrain | 0.557029 | 0.565406 | +0.008377 |
| valve | 0.388359 | 0.376070 | -0.012289 |

Verdict:

```text
V10 is a negative result for the full_5view CTAF mainline.
```

## 30. Current Best After V10

Keep the current mainline unchanged:

```text
CTAF V7 = full_5view tensor frontend
          + Conv2D AE latent summary vectors (mean/std/max)
          + train-normal latent kNN memory score, k=5
```

Current best balanced metrics remain:

```text
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829
```

What V10 changes scientifically:

- it strengthens the conclusion that `tf_mask` is not the path forward for the
  pooled latent-summary mainline
- it shows that `tf_mask` does not repair the drift problem of the block-scale
  score branch
- it narrows the next search space: future gains are more likely to come from
  better latent geometry than from denoising corruption

## 31. V11 Result: latent_summary_whiten_knn

After V8/V9/V10, the next protocol-safe score-side question was whether the
latent-summary feature cloud needed correlation correction rather than only
scale correction.

V11 tested a fixed train-normal covariance whitener:

```text
fit train-normal mean and covariance on latent summaries
rotate into covariance eigenbasis
scale by inverse sqrt eigenvalues
then run the same fixed k=5 latent kNN
```

This is still fully label-free and globally fixed.

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_whiten_knn | 0.504286 | 0.505057 | 0.2400 |
| full_5view / latent_summary_whiten_knn | 0.508496 | 0.495106 | 0.2829 |

Direct comparison against the V7 mainline:

```text
V7 full_5view / latent_summary_knn:
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829

V11 full_5view / latent_summary_whiten_knn:
mean_hmean = 0.495106   delta = -0.041232
mean_pAUC  = 0.508496   delta = -0.020902
mean_KS    = 0.2829     delta = +0.0000
```

Per-machine `full_5view` hmean versus V7:

| Machine | V7 full_5view / latent_summary_knn | V11 full_5view / latent_summary_whiten_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.597747 | 0.467148 | -0.130598 |
| fan | 0.562146 | 0.489945 | -0.072201 |
| gearbox | 0.549726 | 0.430669 | -0.119057 |
| slider | 0.490213 | 0.573310 | +0.083098 |
| toycar | 0.526583 | 0.448438 | -0.078146 |
| toytrain | 0.543640 | 0.517030 | -0.026610 |
| valve | 0.484312 | 0.539202 | +0.054890 |

Interpretation:

- whitening helps `slider` and `valve`
- but it damages `bearing`, `gearbox`, and `toycar` too severely
- it does not improve the overall drift metric

Verdict:

```text
V11 is a strong negative result.
```

So full covariance whitening is not the path forward for CTAF latent-summary
scoring.

## 32. V12 Result: Wider Encoder (`latent_channels=128`)

After multiple score-side failures, the next protocol-safe question was whether
the latent representation itself was too small.

V12 changed only one global capacity parameter:

```text
Conv2D AE latent_channels: 64 -> 128
```

To keep the test focused and cheap, V12 ran only:

```text
logmel_only
full_5view
```

with score modes:

```text
mean_mse
latent_summary_knn
```

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 128 --seed 13711 --device cuda:0 --output_dir results/v12_latent128_summary --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.521654 | 0.534864 | 0.4229 |
| full_5view / latent_summary_knn | 0.523835 | 0.517595 | 0.2943 |

Direct comparison against the V7 mainline:

```text
V7 full_5view / latent_summary_knn:
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829

V12 full_5view / latent_summary_knn:
mean_hmean = 0.517595   delta = -0.018743
mean_pAUC  = 0.523835   delta = -0.005563
mean_KS    = 0.2943     delta = +0.0114
```

Per-machine `full_5view` hmean versus V7:

| Machine | V7 full_5view / latent_summary_knn | V12 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.597747 | 0.582416 | -0.015331 |
| fan | 0.562146 | 0.548790 | -0.013356 |
| gearbox | 0.549726 | 0.556492 | +0.006765 |
| slider | 0.490213 | 0.467397 | -0.022816 |
| toycar | 0.526583 | 0.472797 | -0.053786 |
| toytrain | 0.543640 | 0.528704 | -0.014936 |
| valve | 0.484312 | 0.466573 | -0.017739 |

Interpretation:

- widening the encoder to 128 channels does not improve the full_5view mainline
- it helps `gearbox` slightly
- but it hurts 6/7 machines, especially `toycar`
- it still does not let `full_5view` beat same-mode `logmel_only`

Verdict:

```text
V12 is a negative result for simple capacity scaling.
```

## 33. Current Best After V12

Keep the current best CTAF mainline unchanged:

```text
CTAF V7 = full_5view tensor frontend
          + Conv2D AE latent summary vectors (mean/std/max)
          + train-normal latent kNN memory score, k=5
```

Current best balanced metrics remain:

```text
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829
```

The cumulative evidence after V8-V12 is now:

- score normalization alone is not enough
- `tf_mask` is not the right training-side fix
- simple latent width scaling is not enough

So if the project continues toward the long-run `0.70` target, the next gain
likely requires a more meaningful representation change than:

- post-hoc score geometry tricks
- denoising corruption
- wider versions of the same AE

## 34. V13 Result: Viewwise Late-Fusion AE

After V8-V12, the core hypothesis shifted:

```text
the problem may be early view mixing rather than weak score geometry
```

The baseline AE treats CTAF views like ordinary image channels and mixes them
from the first convolution. But the views are not RGB-like; they have different
signal semantics.

V13 therefore changed the representation model itself:

```text
encode each view with a shared per-view stem
delay cross-view mixing until a deeper fusion layer
decode back through the symmetric late-fusion structure
score with the same fixed latent_summary_knn rule
```

This remains protocol-safe because:

- the architecture is global and fixed
- no anomaly labels are used
- scoring remains train-normal latent-summary kNN with fixed `k=5`
- no machine-specific branches or selection rules are introduced

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --model_type viewwise_late_fusion --seed 13711 --device cuda:0 --output_dir results/v13_viewwise_late_fusion --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed summary:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.522556 | 0.531876 | 0.3943 |
| full_5view / latent_summary_knn | 0.551654 | 0.563192 | 0.3114 |

Direct comparison against the frozen V7 mainline:

```text
V7 full_5view / latent_summary_knn:
mean_hmean = 0.536338
mean_pAUC  = 0.529398
mean_KS    = 0.2829

V13 full_5view / latent_summary_knn:
mean_hmean = 0.563192   delta = +0.026854
mean_pAUC  = 0.551654   delta = +0.022256
mean_KS    = 0.3114     delta = +0.0285
```

This is the first post-V7 architecture change that clearly improves both hmean
and pAUC.

Equally important, under the same model and score rule, `full_5view` now also
beats `logmel_only`:

```text
V13 logmel_only / latent_summary_knn:
mean_hmean = 0.531876
mean_pAUC  = 0.522556
mean_KS    = 0.3943

V13 full_5view / latent_summary_knn:
mean_hmean = 0.563192   delta = +0.031316
mean_pAUC  = 0.551654   delta = +0.029098
mean_KS    = 0.3114     delta = -0.0829
```

So the full CTAF question finally gets a clean positive answer:

```text
with a view-aware late-fusion encoder, the full_5view tensor frontend
outperforms the corresponding clean logmel baseline.
```

Per-machine `full_5view / latent_summary_knn` hmean versus V7:

| Machine | V7 full_5view / latent_summary_knn | V13 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.597747 | 0.648936 | +0.051189 |
| fan | 0.562146 | 0.560465 | -0.001681 |
| gearbox | 0.549726 | 0.578515 | +0.028789 |
| slider | 0.490213 | 0.494056 | +0.003843 |
| toycar | 0.526583 | 0.506567 | -0.020016 |
| toytrain | 0.543640 | 0.547962 | +0.004323 |
| valve | 0.484312 | 0.605844 | +0.121532 |

Interpretation:

- V13 improves 5/7 machines versus the frozen V7 mainline
- the biggest gains are `valve`, `bearing`, and `gearbox`
- `slider` no longer collapses
- the only notable regression is `toycar`
- `fan` is effectively flat

Tradeoff:

```text
V13 improves discrimination strongly,
but KS is worse than V7 (0.3114 vs 0.2829).
```

This is still acceptable because:

- the KS increase is modest compared with the large hmean/pAUC gain
- V13 remains clearly better than the matched `logmel_only` baseline
- it is a real architecture-side improvement, not a score-selection artifact

Verdict:

```text
V13 is the new current best CTAF mainline.
```

## 35. Current Best After V13

Update the CTAF mainline to:

```text
CTAF V13 = full_5view tensor frontend
           + viewwise late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics:

```text
mean_hmean = 0.563192
mean_pAUC  = 0.551654
mean_KS    = 0.3114
```

## 64. V25 Result: Use Target-Domain Normal Train Audio

V25 tested the largest remaining clean data-regime lever:

```text
use_target_train = True
```

This expands training from source normal only to source + target normal train
audio while keeping the V16 architecture fixed.

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.514286 | 0.534021 | 0.3086 |
| full_5view / latent_summary_knn | 0.527368 | 0.547062 | 0.2543 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V25 full_5view / latent_summary_knn:
mean_hmean = 0.547062   delta = -0.017232
mean_pAUC  = 0.527368   delta = -0.026091
mean_KS    = 0.2543     delta = -0.0343
```

Interpretation:

- adding target-domain normal train audio improves drift materially
- but it hurts hmean and pAUC too much
- the extra domain coverage is not enough to justify the discrimination loss

Verdict:

```text
V25 is a useful domain-shift diagnostic, but not a new best model.
```

## 65. V26 Result: Train Multi-View Input, Reconstruct Logmel Only

V26 changed the training objective rather than the architecture:

```text
loss_view_mode = logmel_only
```

The model still sees the full multi-view tensor as input, but the
reconstruction loss only supervises the core logmel view. The hypothesis is
that auxiliary views may help explain the primary acoustic structure without
wasting capacity on reconstructing all derived views.

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.522556 | 0.531876 | 0.3943 |
| full_5view / latent_summary_knn | 0.536917 | 0.545047 | 0.3200 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V26 full_5view / latent_summary_knn:
mean_hmean = 0.545047   delta = -0.019247
mean_pAUC  = 0.536917   delta = -0.016542
mean_KS    = 0.3200     delta = +0.0314
```

Interpretation:

- focusing reconstruction on logmel alone does not improve the latent memory
  representation
- the model likely loses useful pressure to preserve multi-view structure
- this objective change is weaker than the original all-view reconstruction

Verdict:

```text
V26 is a negative result for logmel-only reconstruction supervision.
```

## 66. Current Best After V26

Keep the CTAF mainline unchanged:

```text
CTAF V16 = full_5view tensor frontend
           + viewwise unshared late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

## 67. Next Run Started: V27 use_target_train + logmel-only Loss

Since V25 and V26 each attack different nuisance sources, the next queued run
combines them:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --use_target_train --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode logmel_only --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v27_use_target_train_logmel_target_unshared_late_fusion --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

This is still far from the long-run `0.70` goal, but it is the clearest sign
so far that the bottleneck was architectural:

- not score normalization
- not denoising corruption
- not simple width scaling
- but the way multi-view information is fused

This makes the next direction much clearer:

```text
continue along view-aware representation learning,
not along more score-side post-processing.
```

## 36. V14 Result: Viewwise Late-Fusion AE + Wider Latent

The next question after V13 was whether the new late-fusion architecture still
benefits from a larger latent bottleneck once the representation is structured
correctly.

V14 kept the V13 architecture and score rule fixed, and changed only:

```text
latent_channels: 64 -> 128
```

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 128 --model_type viewwise_late_fusion --seed 13711 --device cuda:0 --output_dir results/v14_viewwise_late_fusion_latent128 --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.526992 | 0.529086 | 0.4200 |
| full_5view / latent_summary_knn | 0.545639 | 0.552212 | 0.3000 |

Direct comparison against the V13 mainline:

```text
V13 full_5view / latent_summary_knn:
mean_hmean = 0.563192
mean_pAUC  = 0.551654
mean_KS    = 0.3114

V14 full_5view / latent_summary_knn:
mean_hmean = 0.552212   delta = -0.010980
mean_pAUC  = 0.545639   delta = -0.006015
mean_KS    = 0.3000     delta = -0.0114
```

So the tradeoff is:

- `V14` improves drift slightly over `V13`
- but loses both hmean and pAUC

It still beats its matched `logmel_only` baseline:

```text
V14 logmel_only / latent_summary_knn:
mean_hmean = 0.529086
mean_pAUC  = 0.526992
mean_KS    = 0.4200

V14 full_5view / latent_summary_knn:
mean_hmean = 0.552212   delta = +0.023126
mean_pAUC  = 0.545639   delta = +0.018647
mean_KS    = 0.3000     delta = -0.1200
```

Per-machine `full_5view` hmean versus V13:

| Machine | V13 full_5view / latent_summary_knn | V14 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.648936 | 0.613913 | -0.035023 |
| fan | 0.560465 | 0.556404 | -0.004061 |
| gearbox | 0.578515 | 0.517948 | -0.060567 |
| slider | 0.494056 | 0.518315 | +0.024259 |
| toycar | 0.506567 | 0.512225 | +0.005657 |
| toytrain | 0.547962 | 0.554936 | +0.006973 |
| valve | 0.605844 | 0.591741 | -0.014103 |

Interpretation:

- width helps `slider`, `toycar`, and `toytrain`
- but it damages `bearing` and especially `gearbox`
- net effect: the wider late-fusion model is more stable, but less accurate

Verdict:

```text
V14 is a useful regularized variant, but it does not replace V13.
```

## 37. Current Best After V14

Keep the current best CTAF mainline unchanged:

```text
CTAF V13 = full_5view tensor frontend
           + viewwise late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.563192
mean_pAUC  = 0.551654
mean_KS    = 0.3114
```

## 57. V21 Result: Unshared Late Fusion + Dropout 0.05

V21 repeated the V20 idea with weaker regularization:

```text
dropout_p = 0.05
model_type = viewwise_unshared_late_fusion
latent_channels = 64
stem_channels = 8
per_view_channels = 16
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.528045 | 0.536166 | 0.4000 |
| full_5view / latent_summary_knn | 0.544135 | 0.557568 | 0.2943 |

Comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V21 full_5view / latent_summary_knn:
mean_hmean = 0.557568   delta = -0.006726
mean_pAUC  = 0.544135   delta = -0.009324
mean_KS    = 0.2943     delta = +0.0057
```

Comparison against V20:

```text
V20 full_5view / latent_summary_knn:
mean_hmean = 0.554774
mean_pAUC  = 0.541203
mean_KS    = 0.2971

V21 full_5view / latent_summary_knn:
mean_hmean = 0.557568   delta = +0.002794
mean_pAUC  = 0.544135   delta = +0.002932
mean_KS    = 0.2943     delta = -0.0028
```

Interpretation:

- `dropout_p=0.05` is clearly better than `dropout_p=0.10`
- but it still does not recover the no-dropout V16 mainline
- regularization may help calibration, but it is not the main unlock

Verdict:

```text
V21 is a useful calibration result, but not a new best model.
```

## 60. V22 Result: Unshared Gated Late Fusion

V22 tested a more adaptive fusion mechanism on top of the current best
representation family:

```text
keep unshared stems
learn sample-wise view gates before fusion
score with the same latent_summary_knn rule
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.530226 | 0.543339 | 0.3571 |
| full_5view / latent_summary_knn | 0.535865 | 0.549257 | 0.2714 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V22 full_5view / latent_summary_knn:
mean_hmean = 0.549257   delta = -0.015037
mean_pAUC  = 0.535865   delta = -0.017594
mean_KS    = 0.2714     delta = -0.0172
```

Comparison inside the same V22 model:

```text
V22 full_5view / latent_summary_knn:
mean_hmean = 0.549257
mean_pAUC  = 0.535865

V22 logmel_only / latent_summary_knn:
mean_hmean = 0.543339
mean_pAUC  = 0.530226
```

Interpretation:

- learned view gates help source-target alignment
- but the current gate mechanism loses too much anomaly discrimination
- the direction is not useless, but this implementation is not yet competitive

Verdict:

```text
V22 is an informative adaptive-fusion diagnostic, but not a new best model.
```

## 62. V24 Result: Gated Late Fusion + View Dropout 0.2

V24 combined the two most plausible but individually insufficient ideas:

```text
unshared gated late fusion
+ view_dropout training (mask_prob = 0.2)
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.540902 | 0.564417 | 0.3657 |
| full_5view / latent_summary_knn | 0.541053 | 0.539668 | 0.2971 |

Interpretation:

- the combination keeps KS in a reasonable range
- but it collapses hmean relative to V16
- and it even loses to its matched `logmel_only` baseline on hmean

Verdict:

```text
V24 is a negative result for the gated-plus-view-dropout combination.
```

## 63. V25 Result: Use Target-Domain Normal Train Audio

V25 tested the largest remaining clean label-free data-regime lever:

```text
use_target_train = True
```

This expands training from source normal only to:

```text
source normal train
+ target normal train
```

while keeping the V16 architecture fixed.

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.514286 | 0.534021 | 0.3086 |
| full_5view / latent_summary_knn | 0.527368 | 0.547062 | 0.2543 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V25 full_5view / latent_summary_knn:
mean_hmean = 0.547062   delta = -0.017232
mean_pAUC  = 0.527368   delta = -0.026091
mean_KS    = 0.2543     delta = -0.0343
```

So adding target-domain normal training audio helps drift a lot, but hurts
discrimination too much.

Inside the same V25 setting, `full_5view` still beats `logmel_only`:

```text
V25 full_5view / latent_summary_knn:
mean_hmean = 0.547062
mean_pAUC  = 0.527368

V25 logmel_only / latent_summary_knn:
mean_hmean = 0.534021
mean_pAUC  = 0.514286
```

Interpretation:

- target-train normals reduce source-target drift materially
- but the extra target data also blurs anomaly separation
- this suggests domain adaptation alone is not the missing piece

Verdict:

```text
V25 is a useful domain-shift diagnostic, but not a new best model.
```

## 61. V23 Result: Unshared Late Fusion + View Dropout 0.2

V23 tested a more semantically meaningful corruption than spatial dropout:

```text
model_type = viewwise_unshared_late_fusion
noise_mode = view_dropout
mask_prob = 0.2
```

The intent was to force the representation to remain useful even when some
views are missing, rather than merely adding local feature noise.

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.528045 | 0.536166 | 0.4229 |
| full_5view / latent_summary_knn | 0.536692 | 0.544378 | 0.2914 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V23 full_5view / latent_summary_knn:
mean_hmean = 0.544378   delta = -0.019916
mean_pAUC  = 0.536692   delta = -0.016767
mean_KS    = 0.2914     delta = +0.0028
```

Comparison inside the same V23 model:

```text
V23 full_5view / latent_summary_knn:
mean_hmean = 0.544378
mean_pAUC  = 0.536692

V23 logmel_only / latent_summary_knn:
mean_hmean = 0.536166
mean_pAUC  = 0.528045
```

Interpretation:

- view dropout is less destructive than the wider-local-branches variant
- but it still does not beat the no-corruption V16 mainline
- the corruption seems to regularize too much and weakens several machines

Per-machine `full_5view` hmean versus V16:

| Machine | V16 full_5view / latent_summary_knn | V23 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.619938 | 0.613765 | -0.006173 |
| fan | 0.599299 | 0.556039 | -0.043260 |
| gearbox | 0.543746 | 0.502961 | -0.040785 |
| slider | 0.529575 | 0.543154 | +0.013579 |
| toycar | 0.562915 | 0.529888 | -0.033027 |
| toytrain | 0.529068 | 0.507743 | -0.021324 |
| valve | 0.565519 | 0.557096 | -0.008423 |

Verdict:

```text
V23 is a negative result for view-dropout training at mask_prob=0.2.
```

## 59. V22 Result: Unshared Gated Late Fusion

V22 tested a more adaptive version of the current best architecture:

```text
keep unshared stems
learn sample-wise view gates before fusion
score with the same latent_summary_knn rule
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.530226 | 0.543339 | 0.3571 |
| full_5view / latent_summary_knn | 0.535865 | 0.549257 | 0.2714 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V22 full_5view / latent_summary_knn:
mean_hmean = 0.549257   delta = -0.015037
mean_pAUC  = 0.535865   delta = -0.017594
mean_KS    = 0.2714     delta = -0.0172
```

So the gating model gives a clear tradeoff:

- better source-target alignment than V16
- but noticeably worse discrimination

Inside the same V22 model, `full_5view` still beats the matched logmel
baseline:

```text
V22 full_5view / latent_summary_knn:
mean_hmean = 0.549257
mean_pAUC  = 0.535865

V22 logmel_only / latent_summary_knn:
mean_hmean = 0.543339
mean_pAUC  = 0.530226
```

Interpretation:

- learned view gates are useful for drift control
- but the current gating mechanism is too aggressive or too restrictive
- it reduces useful anomaly-discriminative view information along with nuisance
  variation

Verdict:

```text
V22 is an informative adaptive-fusion diagnostic, but not a new best model.
```

## 49. V19 Result: Wider Local Branches

After V18, the next architecture question was whether the best current model
needed more capacity in the per-view local pathway rather than in the global
latent bottleneck.

V19 kept the V16 architecture but widened only the local branches:

```text
stem_channels: 8  -> 16
per_view_channels: 16 -> 32
latent_channels stays 64
model_type = viewwise_unshared_late_fusion
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.524662 | 0.543632 | 0.3714 |
| full_5view / latent_summary_knn | 0.539549 | 0.532775 | 0.3029 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V19 full_5view / latent_summary_knn:
mean_hmean = 0.532775   delta = -0.031519
mean_pAUC  = 0.539549   delta = -0.013910
mean_KS    = 0.3029     delta = +0.0143
```

This is a clear negative result:

- larger local branches degrade both hmean and pAUC
- drift also worsens
- the wider-local model even loses to its matched `logmel_only` baseline on hmean

Verdict:

```text
V19 is a negative result for local branch width scaling.
```

## 50. V20 Result: Unshared Late Fusion + Dropout 0.1

Because V19 looked overfit rather than underfit, V20 tested a lightweight
structural regularization:

```text
dropout_p = 0.1
model_type = viewwise_unshared_late_fusion
latent_channels = 64
stem_channels = 8
per_view_channels = 16
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.539624 | 0.547613 | 0.4143 |
| full_5view / latent_summary_knn | 0.541203 | 0.554774 | 0.2971 |

Comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V20 full_5view / latent_summary_knn:
mean_hmean = 0.554774   delta = -0.009520
mean_pAUC  = 0.541203   delta = -0.012256
mean_KS    = 0.2971     delta = +0.0085
```

Comparison inside the same V20 model:

```text
V20 logmel_only / latent_summary_knn:
mean_hmean = 0.547613
mean_pAUC  = 0.539624
mean_KS    = 0.4143

V20 full_5view / latent_summary_knn:
mean_hmean = 0.554774   delta = +0.007161
mean_pAUC  = 0.541203   delta = +0.001579
mean_KS    = 0.2971     delta = -0.1172
```

So V20 does not beat V16, but it is not useless:

- it remains better than the matched logmel baseline
- it repairs some of the overfit pattern seen in V19
- it improves several machines that V16 struggled on

Per-machine `full_5view` hmean versus V16:

| Machine | V16 full_5view / latent_summary_knn | V20 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.619938 | 0.616871 | -0.003067 |
| fan | 0.599299 | 0.547694 | -0.051605 |
| gearbox | 0.543746 | 0.521704 | -0.022042 |
| slider | 0.529575 | 0.590842 | +0.061266 |
| toycar | 0.562915 | 0.507970 | -0.054945 |
| toytrain | 0.529068 | 0.520221 | -0.008846 |
| valve | 0.565519 | 0.578116 | +0.012596 |

Interpretation:

- `dropout_p=0.1` is too strong globally
- it helps `slider` and `valve`
- but it hurts `fan`, `gearbox`, and `toycar` too much

Verdict:

```text
V20 is a useful regularization diagnostic, but not a new best model.
```

## 51. Current Best After V20

Keep the CTAF mainline unchanged:

```text
CTAF V16 = full_5view tensor frontend
           + viewwise unshared late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

## 53. V19 Result: Wider Local Branches

The next question after V18 was whether the unshared late-fusion model simply
needed more local capacity before fusion.

V19 widened only the local branches:

```text
stem_channels: 8 -> 16
per_view_channels: 16 -> 32
latent_channels stays 64
model_type = viewwise_unshared_late_fusion
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.524662 | 0.543632 | 0.3714 |
| full_5view / latent_summary_knn | 0.539549 | 0.532775 | 0.3029 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V19 full_5view / latent_summary_knn:
mean_hmean = 0.532775   delta = -0.031519
mean_pAUC  = 0.539549   delta = -0.013910
mean_KS    = 0.3029     delta = +0.0143
```

Verdict:

```text
V19 is a negative result for local branch width scaling.
```

It not only loses to V16, it also loses to its matched `logmel_only` baseline
on hmean.

## 54. V20 Result: Unshared Late Fusion + Dropout 0.1

Because V19 looked more like overfitting than under-capacity, V20 tested
lightweight structural regularization:

```text
dropout_p = 0.1
model_type = viewwise_unshared_late_fusion
latent_channels = 64
stem_channels = 8
per_view_channels = 16
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.539624 | 0.547613 | 0.4143 |
| full_5view / latent_summary_knn | 0.541203 | 0.554774 | 0.2971 |

Direct comparison against V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V20 full_5view / latent_summary_knn:
mean_hmean = 0.554774   delta = -0.009520
mean_pAUC  = 0.541203   delta = -0.012256
mean_KS    = 0.2971     delta = +0.0085
```

But unlike V19, V20 still beats the matched logmel baseline:

```text
V20 logmel_only / latent_summary_knn:
mean_hmean = 0.547613
mean_pAUC  = 0.539624
mean_KS    = 0.4143

V20 full_5view / latent_summary_knn:
mean_hmean = 0.554774   delta = +0.007161
mean_pAUC  = 0.541203   delta = +0.001579
mean_KS    = 0.2971     delta = -0.1172
```

Interpretation:

- `dropout_p=0.1` is too strong to become the best mainline
- but it does suggest that some regularization can help the current
  representation family
- the effect is much less damaging than V19

Verdict:

```text
V20 is a useful regularization diagnostic, but not a new best model.
```

## 55. Current Best After V20

Keep the CTAF mainline unchanged:

```text
CTAF V16 = full_5view tensor frontend
           + viewwise unshared late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

## 56. Next Run Started: V21 unshared late fusion + Dropout 0.05

V20 suggests the direction might be correct but the regularization strength is
too large, so the next run has been started with:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.05 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v21_unshared_late_fusion_dropout005 --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

## 52. Next Run Started: V21 unshared late fusion + Dropout 0.05

V20 suggests the right direction may be regularization, but `dropout_p=0.1`
looks too aggressive. So the next run has been started with a softer global
setting:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.05 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v21_unshared_late_fusion_dropout005 --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

## 47. V18 Result: edge_3view vs full_5view Under Unshared Late Fusion

V18 repeated the edge-versus-full comparison, but now under the current best
fusion architecture:

```text
model_type = viewwise_unshared_late_fusion
latent_channels = 64
edge_3view vs full_5view
```

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,edge_3view,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v18_viewwise_unshared_edge_vs_full --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed summary:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.522556 | 0.531876 | 0.3943 |
| edge_3view / latent_summary_knn | 0.549098 | 0.560391 | 0.3086 |
| full_5view / latent_summary_knn | 0.553459 | 0.564294 | 0.2886 |

Key comparison inside the same architecture:

```text
V18 edge_3view / latent_summary_knn:
mean_hmean = 0.560391
mean_pAUC  = 0.549098
mean_KS    = 0.3086

V18 full_5view / latent_summary_knn:
mean_hmean = 0.564294   delta = +0.003903
mean_pAUC  = 0.553459   delta = +0.004361
mean_KS    = 0.2886     delta = -0.0200
```

So even under the best current architecture:

- `full_5view` still beats `edge_3view`
- `full_5view` also has better source-target KS

Per-machine hmean under unshared late fusion:

| Machine | edge_3view / latent_summary_knn | full_5view / latent_summary_knn |
|---|---:|---:|
| bearing | 0.641783 | 0.619938 |
| fan | 0.587207 | 0.599299 |
| gearbox | 0.587967 | 0.543746 |
| slider | 0.511296 | 0.529575 |
| toycar | 0.500598 | 0.562915 |
| toytrain | 0.496972 | 0.529068 |
| valve | 0.596915 | 0.565519 |

Interpretation:

- `edge_3view` wins on `bearing`, `gearbox`, and `valve`
- `full_5view` wins on `fan`, `slider`, `toycar`, and `toytrain`
- the aggregate result still favors `full_5view`, and by a wider margin on the
  current best architecture than in the shared-stem case

Verdict:

```text
V18 confirms that the current best frontend remains full_5view, not edge_3view,
even after moving to unshared late fusion.
```

## 48. Current Best After V18

Keep the CTAF mainline unchanged:

```text
CTAF V16 = full_5view tensor frontend
           + viewwise unshared late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

## 44. V17 Result: Unshared Late Fusion + Wider Latent

V17 asked whether the now-best unshared-stem late-fusion architecture can
benefit from a larger latent bottleneck:

```text
model_type = viewwise_unshared_late_fusion
latent_channels: 64 -> 128
```

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 128 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v17_viewwise_unshared_late_fusion_latent128 --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed result:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.526992 | 0.529086 | 0.4200 |
| full_5view / latent_summary_knn | 0.540752 | 0.552119 | 0.2886 |

Direct comparison against the V16 mainline:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V17 full_5view / latent_summary_knn:
mean_hmean = 0.552119   delta = -0.012175
mean_pAUC  = 0.540752   delta = -0.012707
mean_KS    = 0.2886     delta = +0.0000
```

So the larger latent bottleneck does not help once the unshared-stem structure
is already in place.

It still beats the matched logmel baseline:

```text
V17 logmel_only / latent_summary_knn:
mean_hmean = 0.529086
mean_pAUC  = 0.526992
mean_KS    = 0.4200

V17 full_5view / latent_summary_knn:
mean_hmean = 0.552119   delta = +0.023033
mean_pAUC  = 0.540752   delta = +0.013760
mean_KS    = 0.2886     delta = -0.1314
```

Per-machine `full_5view` hmean versus V16:

| Machine | V16 full_5view / latent_summary_knn | V17 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.619938 | 0.621067 | +0.001129 |
| fan | 0.599299 | 0.594158 | -0.005141 |
| gearbox | 0.543746 | 0.546323 | +0.002577 |
| slider | 0.529575 | 0.507990 | -0.021585 |
| toycar | 0.562915 | 0.530877 | -0.032038 |
| toytrain | 0.529068 | 0.520675 | -0.008393 |
| valve | 0.565519 | 0.543745 | -0.021774 |

Interpretation:

- width gives tiny gains on `bearing` and `gearbox`
- but it hurts `slider`, `toycar`, and `valve`
- the aggregate result is a clear regression from V16

Verdict:

```text
V17 is a negative result for width scaling inside the unshared-stem architecture.
```

## 45. Current Best After V17

Keep the CTAF mainline unchanged:

```text
CTAF V16 = full_5view tensor frontend
           + viewwise unshared late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

## 46. Next Run Started: V18 edge_3view vs full_5view Under Unshared Late Fusion

The next unresolved question is whether `edge_3view` becomes stronger than
`full_5view` once the best current fusion architecture is used:

```text
model_type = viewwise_unshared_late_fusion
latent_channels = 64
edge_3view vs full_5view
```

So the next focused run has been started with:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,edge_3view,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v18_viewwise_unshared_edge_vs_full --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

## 41. V16 Result: Viewwise Unshared Late-Fusion AE

V13 used a shared per-view stem before late fusion. The next protocol-safe
question was whether even that shared stem was too restrictive:

```text
should each acoustic view have its own local encoder/decoder stem
before fusion?
```

V16 tested exactly that:

```text
separate stem per view
late fusion in the bottleneck
same latent summary score rule
same latent_channels = 64
```

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v16_viewwise_unshared_late_fusion --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed summary:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.522556 | 0.531876 | 0.3943 |
| full_5view / latent_summary_knn | 0.553459 | 0.564294 | 0.2886 |

Direct comparison against the V13 mainline:

```text
V13 full_5view / latent_summary_knn:
mean_hmean = 0.563192
mean_pAUC  = 0.551654
mean_KS    = 0.3114

V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294   delta = +0.001102
mean_pAUC  = 0.553459   delta = +0.001805
mean_KS    = 0.2886     delta = -0.0228
```

This is a real improvement, even if modest:

- hmean improves
- pAUC improves
- source-target KS also improves

And, as before, `full_5view` still beats the matched `logmel_only` baseline:

```text
V16 logmel_only / latent_summary_knn:
mean_hmean = 0.531876
mean_pAUC  = 0.522556
mean_KS    = 0.3943

V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294   delta = +0.032418
mean_pAUC  = 0.553459   delta = +0.030903
mean_KS    = 0.2886     delta = -0.1057
```

Per-machine `full_5view` hmean versus V13:

| Machine | V13 full_5view / latent_summary_knn | V16 full_5view / latent_summary_knn | Delta |
|---|---:|---:|---:|
| bearing | 0.648936 | 0.619938 | -0.028997 |
| fan | 0.560465 | 0.599299 | +0.038834 |
| gearbox | 0.578515 | 0.543746 | -0.034769 |
| slider | 0.494056 | 0.529575 | +0.035519 |
| toycar | 0.506567 | 0.562915 | +0.056348 |
| toytrain | 0.547962 | 0.529068 | -0.018895 |
| valve | 0.605844 | 0.565519 | -0.040325 |

Interpretation:

- unshared stems help `fan`, `slider`, and especially `toycar`
- they hurt `bearing`, `gearbox`, and `valve`
- but the net effect is still positive at the 7-machine mean level
- importantly, the model is more balanced in drift than V13

Verdict:

```text
V16 is the new current best CTAF mainline.
```

## 42. Current Best After V16

Update the CTAF mainline to:

```text
CTAF V16 = full_5view tensor frontend
           + viewwise unshared late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

This confirms the next architectural lesson:

```text
the acoustic views do not want a shared low-level encoder stem.
```

## 43. Next Run Started: V17 unshared late fusion + latent128

The next focused question is whether the unshared-stem architecture can use a
larger latent bottleneck better than the shared-stem architecture did.

So the next run has been started with:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 128 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v17_viewwise_unshared_late_fusion_latent128 --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

V17 purpose:

```text
test whether larger latent capacity helps once both
view-specific stems and late fusion are in place
```

## 38. Next Run Started: V15 edge_3view vs full_5view Under Late Fusion

V13/V14 together suggest the main gain comes from view-aware fusion rather than
from simply widening the bottleneck. The next unresolved question is:

```text
under the correct late-fusion architecture,
is edge_3view actually stronger than full_5view?
```

So the next focused run has been started with:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,edge_3view,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --model_type viewwise_late_fusion --seed 13711 --device cuda:0 --output_dir results/v15_viewwise_late_fusion_edge_vs_full --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

V15 purpose:

```text
test whether the best late-fusion CTAF representation is
the full 5-view tensor or the structural 3-view edge branch
```

## 39. V15 Result: edge_3view vs full_5view Under Late Fusion

V15 compared the two most plausible late-fusion frontends under the same model
and score rule:

```text
edge_3view
vs
full_5view
```

Server command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,edge_3view,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --model_type viewwise_late_fusion --seed 13711 --device cuda:0 --output_dir results/v15_viewwise_late_fusion_edge_vs_full --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Observed summary:

| Experiment / score | Mean pAUC | Mean hmean | Mean KS |
|---|---:|---:|---:|
| logmel_only / latent_summary_knn | 0.522556 | 0.531876 | 0.3943 |
| edge_3view / latent_summary_knn | 0.549098 | 0.556579 | 0.3343 |
| full_5view / latent_summary_knn | 0.551654 | 0.563192 | 0.3114 |

Key comparison inside the same architecture:

```text
V15 edge_3view / latent_summary_knn:
mean_hmean = 0.556579
mean_pAUC  = 0.549098
mean_KS    = 0.3343

V15 full_5view / latent_summary_knn:
mean_hmean = 0.563192   delta = +0.006613
mean_pAUC  = 0.551654   delta = +0.002556
mean_KS    = 0.3114     delta = -0.0229
```

So under the correct late-fusion encoder:

- `full_5view` is still better than `edge_3view`
- the gain is modest but consistent
- `full_5view` also has better source-target KS

Per-machine hmean under late fusion:

| Machine | edge_3view / latent_summary_knn | full_5view / latent_summary_knn |
|---|---:|---:|
| bearing | 0.642547 | 0.648936 |
| fan | 0.596401 | 0.560465 |
| gearbox | 0.554105 | 0.578515 |
| slider | 0.496282 | 0.494056 |
| toycar | 0.494583 | 0.506567 |
| toytrain | 0.520542 | 0.547962 |
| valve | 0.591592 | 0.605844 |

Interpretation:

- `edge_3view` wins only weakly on `fan` and `slider`
- `full_5view` wins clearly on `gearbox`, `toycar`, `toytrain`, and `valve`
- the aggregate result confirms that the extra views are useful once fusion is
  done correctly

Verdict:

```text
V15 confirms that the current best frontend remains full_5view, not edge_3view.
```

## 40. Current Best After V15

Keep the CTAF mainline unchanged:

```text
CTAF V13 = full_5view tensor frontend
           + viewwise late-fusion Conv2D AE
           + latent summary vectors (mean/std/max)
           + train-normal latent kNN memory score, k=5
```

Current best metrics remain:

```text
mean_hmean = 0.563192
mean_pAUC  = 0.551654
mean_KS    = 0.3114
```

## V27/V28 Audit

### 1. V16 Frozen Baseline

Frozen mainline:

```text
CTAF V16 =
full_5view tensor frontend
+ viewwise_unshared_late_fusion Conv2D AE
+ latent summary vectors (mean/std/max)
+ train-normal latent kNN memory score
+ k=5
+ source-only normal training
```

Frozen metrics:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
```

### 2. V27 Command and Result

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --use_target_train --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode logmel_only --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v27_use_target_train_logmel_target_unshared_late_fusion --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Result summary for `full_5view / latent_summary_knn`:

```text
mean_hmean = 0.530462
mean_pAUC  = 0.527368
mean_KS    = 0.2200
matched logmel_only mean_hmean = 0.534021
matched logmel_only mean_pAUC  = 0.514286
```

Audit against frozen V16 criteria:

- `hmean > 0.564294`: FAIL
- `pAUC >= 0.553459`: FAIL
- `KS <= 0.2886`: PASS
- `full_5view` beats matched `logmel_only`: FAIL on hmean, PASS on pAUC

### 3. V28 Command and Result

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --memory_domains source_target_train --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v28_source_target_memory_unshared_late_fusion --score_modes mean_mse,latent_summary_knn --latent_knn_k 5
```

Result summary for `full_5view / latent_summary_knn`:

```text
mean_hmean = 0.544567
mean_pAUC  = 0.537218
mean_KS    = 0.3000
matched logmel_only mean_hmean = 0.535396
matched logmel_only mean_pAUC  = 0.513233
```

Audit against frozen V16 criteria:

- `hmean > 0.564294`: FAIL
- `pAUC >= 0.553459`: FAIL
- `KS <= 0.2886`: FAIL
- `at least 4/7 machines not below V16`: FAIL, only 3/7
- `full_5view` beats matched `logmel_only`: PASS

### 4. Overall Comparison Table

| version | training data | memory bank | loss view mode | hmean | pAUC | KS | verdict |
|---|---|---|---|---:|---:|---:|---|
| V16 | source-only normal train | source_train normals | all_views | 0.564294 | 0.553459 | 0.2886 | frozen baseline |
| V27 | source+target normal train | source_train normals | logmel_only | 0.530462 | 0.527368 | 0.2200 | diagnostic only |
| V28 | source-only normal train | source+target train normals | all_views | 0.544567 | 0.537218 | 0.3000 | diagnostic only |

### 5. Matched Baseline Comparison

| version | logmel_only hmean | full_5view hmean | delta | logmel_only pAUC | full_5view pAUC | delta |
|---|---:|---:|---:|---:|---:|---:|
| V27 | 0.534021 | 0.530462 | -0.003559 | 0.514286 | 0.527368 | +0.013082 |
| V28 | 0.535396 | 0.544567 | +0.009171 | 0.513233 | 0.537218 | +0.023985 |

### 6. Per-Machine Comparison Against V16

| machine | V16 | V27 | V28 | best diagnostic observation |
|---|---:|---:|---:|---|
| bearing | 0.619938 | 0.601644 | 0.639293 | V28>=V16 |
| fan | 0.599299 | 0.578882 | 0.513091 | both below V16 |
| gearbox | 0.543746 | 0.559513 | 0.552338 | V27>=V16; V28>=V16 |
| slider | 0.529575 | 0.486723 | 0.545622 | V28>=V16 |
| toycar | 0.562915 | 0.443887 | 0.473314 | both below V16 |
| toytrain | 0.529068 | 0.501999 | 0.527474 | both below V16 |
| valve | 0.565519 | 0.540581 | 0.560836 | both below V16 |

### 7. Final Verdict

Neither V27 nor V28 beats V16 on both hmean and pAUC.

```text
V16 remains the frozen CTAF-v1 mainline.
```

For V27 specifically, the interpretation is:

```text
This is a domain-stability diagnostic, not a new best method.
```

For V28 specifically, the interpretation is also diagnostic:

```text
This is a domain-stability diagnostic, not a new best method.
```

Conclusion:

- `use_target_train` does not replace V16
- `source+target normal memory bank` does not replace V16
- these data/target-function changes improve some drift aspects but do not beat
  the frozen mainline on the core discrimination metrics
- the next phase should return to multi-seed stability or genuinely new
  structure-level ideas, not more combinations of `target_train`,
  `loss_view_mode`, or `memory_bank` switches

## 68. V29 Result: Multi-Resolution CTAF

V29 tested the first explicit multi-scale representation change:

```text
multires_7view =
1. logmel_short
2. logmel_base
3. logmel_long
4. delta_base
5. deltadelta_base
6. edge_base
7. enhanced_base
```

The rest of the configuration was frozen to the V16 architecture:

```text
model_type = viewwise_unshared_late_fusion
latent_channels = 64
stem_channels = 8
per_view_channels = 16
dropout_p = 0.0
score_mode = latent_summary_knn
k = 5
training = source normal only
memory bank = source normal only
loss_view_mode = all_views
```

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view,multires_7view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v29_multires_7view_unshared_late_fusion --score_modes latent_summary_knn --latent_knn_k 5
```

Overall comparison:

| experiment | hmean | pAUC | KS |
|---|---:|---:|---:|
| logmel_only | 0.543587 | 0.524586 | 0.4029 |
| full_5view | 0.553511 | 0.547820 | 0.3286 |
| multires_7view | 0.556256 | 0.553233 | 0.3286 |

Comparison against frozen V16:

```text
V16 full_5view / latent_summary_knn:
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886

V29 multires_7view / latent_summary_knn:
mean_hmean = 0.556256   delta = -0.008038
mean_pAUC  = 0.553233   delta = -0.000226
mean_KS    = 0.3286     delta = +0.0400
```

Matched baseline comparison:

```text
logmel_only / latent_summary_knn:
mean_hmean = 0.543587
mean_pAUC  = 0.524586

multires_7view / latent_summary_knn:
mean_hmean = 0.556256   delta = +0.012669
mean_pAUC  = 0.553233   delta = +0.028647
```

Per-machine hmean versus V16:

| machine | V16 | V29 multires_7view | best observation |
|---|---:|---:|---|
| bearing | 0.619938 | 0.639499 | improved over V16 |
| fan | 0.599299 | 0.552164 | regresses badly |
| gearbox | 0.543746 | 0.536318 | slight regression |
| slider | 0.529575 | 0.549576 | improved over V16 |
| toycar | 0.562915 | 0.516215 | clear regression |
| toytrain | 0.529068 | 0.514497 | regression |
| valve | 0.565519 | 0.585521 | improved over V16 |

Audit against the frozen V29 success criteria:

- `multires_7view hmean > 0.564294`: FAIL
- `multires_7view pAUC >= 0.553459`: FAIL, but only by `0.000226`
- `KS not clearly worse than 0.2886`: FAIL
- `at least 4/7 machines not below V16`: FAIL, only `3/7`
- `multires_7view beats matched logmel_only`: PASS

Final verdict:

```text
V29 does not cleanly beat V16.
```

More specifically:

```text
V16 remains the frozen CTAF-v1 mainline.
```

Failure analysis:

- multi-resolution logmel adds real signal and clearly beats matched logmel_only
- the added scales do not beat the frozen V16 mainline on balanced hmean
- pAUC is almost preserved, so the main failure is not raw low-FPR separation
- the decisive problem is worse cross-domain score geometry (`KS = 0.3286`)
- the gains are concentrated in `bearing`, `slider`, and `valve`, while
  `fan`, `toycar`, and `toytrain` regress too much

## 69. V16 Multi-Seed Stability Audit

Before attempting any further structure-level change, we audited whether the
frozen `V16` mainline is actually stable across seeds under the same clean,
label-free protocol.

Frozen V16 definition:

```text
CTAF V16 =
full_5view tensor frontend
+ viewwise_unshared_late_fusion Conv2D AE
+ latent summary vectors(mean/std/max)
+ train-normal latent kNN memory score
+ k=5
+ source-only normal training
```

Frozen point estimate:

```text
mean_hmean = 0.564294
mean_pAUC  = 0.553459
mean_KS    = 0.2886
seed       = 13711
```

Fixed 5-seed panel:

`13711, 13721, 13731, 13741, 13751`

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_late_fusion --seed <SEED> --device cuda:0 --output_dir results/v16_seed_panel/seed_<SEED> --score_modes latent_summary_knn --latent_knn_k 5 --no-progress
```

Per-seed overall comparison:

| seed | logmel hmean | V16 hmean | delta | logmel pAUC | V16 pAUC | delta | V16 KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13711 | 0.543587 | 0.553511 | +0.009924 | 0.524586 | 0.547820 | +0.023234 | 0.3286 |
| 13721 | 0.532118 | 0.550205 | +0.018087 | 0.518571 | 0.544962 | +0.026391 | 0.2943 |
| 13731 | 0.536815 | 0.548404 | +0.011589 | 0.526992 | 0.541654 | +0.014662 | 0.3086 |
| 13741 | 0.543400 | 0.553533 | +0.010133 | 0.539173 | 0.543008 | +0.003835 | 0.2686 |
| 13751 | 0.540003 | 0.542080 | +0.002077 | 0.536767 | 0.534211 | -0.002556 | 0.2743 |

Aggregate stability:

| metric | mean | std | min | max | frozen V16 | delta(mean-baseline) |
|---|---:|---:|---:|---:|---:|---:|
| hmean | 0.549547 | 0.004720 | 0.542080 | 0.553533 | 0.564294 | -0.014747 |
| pAUC | 0.542331 | 0.005097 | 0.534211 | 0.547820 | 0.553459 | -0.011128 |
| KS | 0.294880 | 0.024697 | 0.268600 | 0.328600 | 0.2886 | +0.006280 |

Per-machine stability against frozen V16:

| machine | frozen V16 hmean | seed-panel mean hmean | std | frozen V16 pAUC | seed-panel mean pAUC | std | observation |
|---|---:|---:|---:|---:|---:|---:|---|
| bearing | 0.619938 | 0.636343 | 0.015200 | 0.600526 | 0.603368 | 0.016951 | stable positive contributor |
| fan | 0.599299 | 0.572991 | 0.013326 | 0.576842 | 0.562947 | 0.026383 | repeatable regression vs frozen point |
| gearbox | 0.543746 | 0.542791 | 0.026038 | 0.523158 | 0.524632 | 0.014758 | roughly neutral, moderate variance |
| slider | 0.529575 | 0.522276 | 0.026034 | 0.545263 | 0.528316 | 0.005462 | mild hmean/pAUC erosion |
| toycar | 0.562915 | 0.501151 | 0.036409 | 0.559474 | 0.503263 | 0.009020 | largest instability and biggest drop |
| toytrain | 0.529068 | 0.517280 | 0.012142 | 0.516842 | 0.511895 | 0.009286 | small but consistent drop |
| valve | 0.565519 | 0.553996 | 0.033082 | 0.552105 | 0.561895 | 0.028593 | mixed but still strong |

Key audit conclusions:

- `full_5view` beats matched `logmel_only` on hmean in all `5/5` seeds
- `full_5view` beats matched `logmel_only` on both hmean and pAUC in `4/5`
  seeds
- no seed beats the original frozen V16 point estimate on both hmean and pAUC
- the direct rerun of seed `13711` does **not** reproduce the original frozen
  V16 result:

```text
frozen V16:
hmean = 0.564294
pAUC  = 0.553459
KS    = 0.2886

rerun seed 13711:
hmean = 0.553511
pAUC  = 0.547820
KS    = 0.3286
```

Verdict:

```text
V16 is promising but not yet stable enough to be treated as a fully reliable
frozen launchpad for further structure innovation without acknowledging seed
sensitivity.
```

Operational implication:

- do not treat the original `V16` point estimate as a sufficient proof that a
  new structure must exceed `0.564294 / 0.553459 / 0.2886` on a single seed
- the next decision layer should prioritize multi-seed validation discipline
  around V16 before resuming aggressive structure search

## 70. V30 Result: Temporal Pyramid Inside V16

Under the updated seed-panel evidence rule, the next structure experiment was:

```text
CTAF-v2A =
viewwise_unshared_temporal_pyramid
```

Design:

- keep the V16 per-view unshared stems
- insert a fixed three-branch temporal pyramid block after each view stem
- use lightweight temporal convolutions for short / medium / long receptive
  fields
- keep late fusion, latent summary vectors, train-normal latent kNN, `k=5`,
  source-only training, and all evaluation rules unchanged

### Stage 1: Single-Seed Gate Check (`13711`)

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_temporal_pyramid --seed 13711 --device cuda:0 --output_dir results/v30_temporal_pyramid_seed13711 --score_modes latent_summary_knn --latent_knn_k 5 --no-progress
```

Overall result:

| experiment | hmean | pAUC | KS |
|---|---:|---:|---:|
| logmel_only | 0.549725 | 0.539248 | 0.4143 |
| full_5view | 0.562034 | 0.548045 | 0.2943 |

Stage-1 decision:

- beats matched `logmel_only`: `PASS`
- beats V16 seed-panel mean: `PASS`
- beats original frozen V16 point: `FAIL`

So `V30` was allowed to enter the pre-registered `3-seed` panel, but was not
allowed to become a new mainline on a single seed.

### Stage 2: `3-seed` Panel (`13711, 13721, 13731`)

Per-seed results:

| seed | logmel hmean | full_5view hmean | delta | logmel pAUC | full_5view pAUC | delta | full_5view KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13711 | 0.549725 | 0.562034 | +0.012309 | 0.539248 | 0.548045 | +0.008797 | 0.2943 |
| 13721 | 0.540117 | 0.545995 | +0.005878 | 0.534286 | 0.538947 | +0.004661 | 0.2886 |
| 13731 | 0.546489 | 0.557611 | +0.011122 | 0.529173 | 0.536917 | +0.007744 | 0.3029 |

Seed-panel mean:

```text
V30 3-seed mean:
hmean = 0.555213
pAUC  = 0.541303
KS    = 0.295267
```

Compared with the active V16 seed-panel baseline:

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

Delta:

```text
hmean delta = +0.005666
pAUC  delta = -0.001028
KS    delta = +0.000387
```

Final decision under the updated evaluation system:

- majority matched-baseline gate: `PASS`, `3/3`
- seed-panel mean hmean gate: `PASS`
- seed-panel mean pAUC gate: `FAIL`
- KS stability: essentially tied, but slightly worse

Verdict:

```text
V30 temporal pyramid is a real positive structure signal, but it does not
cleanly beat the V16 seed-panel baseline.
```

Therefore:

```text
V30 should not replace V16.
```

Interpretation:

- the internal temporal pyramid is a meaningful direction because it produces
  reproducible gains over matched `logmel_only`
- the failure is no longer simple single-seed instability
- the remaining gap is low-FPR quality (`pAUC`), not gross domain-shift
  collapse
- under the new seed-panel evidence rule, this is a promising branch but not a
  new mainline

## 71. V31 Result: Residual Temporal Pyramid Inside V16

The next constrained correction was:

```text
CTAF-v2B =
viewwise_unshared_residual_temporal_pyramid
```

Design:

- keep the original `V16` unshared late-fusion main path
- add a temporal-pyramid residual branch after each per-view stem
- use:

```text
h = original stem feature
p = temporal pyramid correction
h_out = h + alpha * p
```

- `alpha` is a single global learnable scalar, initialized to `0.1`
- keep late fusion, latent summary vectors, train-normal latent kNN, `k=5`,
  source-only training, and all evaluation rules unchanged

### Single-Seed Audit (`13711`)

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_residual_temporal_pyramid --seed 13711 --device cuda:0 --output_dir results/v31_residual_temporal_pyramid_seed13711 --score_modes latent_summary_knn --latent_knn_k 5 --no-progress
```

Overall result:

| experiment | hmean | pAUC | KS |
|---|---:|---:|---:|
| logmel_only | 0.542030 | 0.532632 | 0.3829 |
| full_5view | 0.555806 | 0.541654 | 0.2714 |

Comparison targets:

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V30 seed 13711:
hmean = 0.562034
pAUC  = 0.548045
KS    = 0.2943
```

Gate evaluation:

- beats matched `logmel_only`: `PASS`
- `hmean > 0.549547`: `PASS`
- `pAUC >= 0.542331`: `FAIL`, short by `0.000677`
- `KS not clearly worse than 0.294880`: `PASS`, actually better
- stronger than `V30 seed 13711` on both hmean and pAUC: `FAIL`

Verdict:

```text
V31 does not pass the single-seed gate.
```

Therefore:

```text
Do not continue to the 3-seed panel.
Do not continue to V32.
```

Interpretation:

- residual temporal correction does improve score geometry meaningfully
- it still beats matched `logmel_only`
- but it misses the active `V16` seed-panel `pAUC` threshold by a small margin
- that means the correction helps stability more than low-FPR discrimination
- under the current seed-panel evidence rule, this is a useful failure, not a
  promotable successor

## 72. V32 Result: Local Tail Latent Memory Score

To directly attack the confirmed low-FPR bottleneck without changing the
encoder, we tested a fixed local-token score:

```text
score_mode = local_tail_knn_top5
```

Rule:

- extract encoder latent map `Z ∈ R^{C×H×W}`
- tokenize every spatial position into a local latent token
- build a train-normal local token memory bank
- compute token-wise kNN distance to the train-normal local memory
- file score = mean(top `5%` token distances)
- larger score = more anomalous

No per-machine tuning, no percentile search, no fusion, and no reverse score
were allowed.

### Single-Seed Run (`13711`)

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v32_local_tail_knn_seed13711 --score_modes latent_summary_knn,local_tail_knn_top5 --latent_knn_k 5 --no-progress
```

Overall comparison:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.539762 | 0.524737 | 0.4057 |
| logmel_only | local_tail_knn_top5 | 0.543065 | 0.523534 | 0.2971 |
| full_5view | latent_summary_knn | 0.561823 | 0.548647 | 0.2686 |
| full_5view | local_tail_knn_top5 | 0.505107 | 0.503383 | 0.2771 |

Gate evaluation for `full_5view / local_tail_knn_top5`:

- beats matched `logmel_only / local_tail_knn_top5`: `FAIL`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least `2` bottleneck machines with improved `TPR@99n`: `FAIL`, `0/5`

Why it fails:

- local-tail scoring inflates the extreme right tail of normal scores
- anomaly local-token scores rise too, but not enough to move beyond the new
  normal `q95–q99` boundary
- therefore the intended low-FPR tail separation does not appear

Critical result:

```text
V32 does not reveal a useful hidden local anomaly tail under this fixed top-5%
local-token kNN rule.
```

Operational consequence:

```text
Do not continue to a V32 seed panel.
Do not tune the top percentile.
Treat this as a fixed-rule failure analysis.
```

## 73. V33 Result: Normal-Tail Compactness Training

To move low-FPR control from score postprocessing into training, we kept the
frozen `V16` architecture and scoring rule but added a fixed train-normal
tail-compactness regularizer:

```text
loss = reconstruction_loss + lambda_tail * normal_tail_compactness_loss
```

where:

- latent summary is extracted for each normal batch sample
- batch center is computed
- sample distances to the center are computed
- the top `10%` distances are averaged
- `lambda_tail = 0.01`

No sweep, no per-machine setting, and no anomaly labels were used.

### Single-Seed Run (`13711`)

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --loss_view_mode all_views --model_type viewwise_unshared_late_fusion --tail_compact_loss --tail_compact_lambda 0.01 --tail_compact_top_frac 0.10 --seed 13711 --device cuda:0 --output_dir results/v33_tail_compact_seed13711 --score_modes latent_summary_knn --latent_knn_k 5 --no-progress
```

Overall result:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.530561 | 0.522632 | 0.3714 |
| full_5view | latent_summary_knn | 0.531218 | 0.518195 | 0.2800 |

Gate evaluation:

- full_5view beats matched logmel_only: `FAIL`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least `2` bottleneck machines improve `TPR@99n`: `FAIL`, `0/5`
- normal `q99` does not rise overall: `PASS`

Key finding:

```text
V33 successfully compresses the normal right tail, but also compresses anomaly
scores toward the center.
```

So it is:

```text
tail-compressing, but not tail-separating
```

This explains the failure:

- normal `q99` falls
- `KS` improves
- but anomaly median and anomaly `q75` also fall
- bottleneck-machine `TPR@99n` gets worse instead of better
- therefore `pAUC` does not recover

Verdict:

```text
V33 does not pass the single-seed gate.
```

Operational consequence:

```text
Do not continue to V34.
Do not sweep lambda.
Record this as a training-side tail-compactness failure analysis.
```

## 74. V34 Result: Masked Cross-View Prediction

Next we changed the self-supervised task, not the architecture:

```text
ssl_task = masked_cross_view_prediction
loss_view_mode = masked_view_only
```

Rule:

- input is `full_5view`
- one view is masked uniformly at random
- model predicts all views
- loss is computed only on the masked view
- score remains `latent_summary_knn`

No anomaly labels, no per-machine masking rule, no fusion, and no automatic
mask-strategy search were used.

### Single-Seed Run (`13711`)

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --ssl_task masked_cross_view_prediction --loss_view_mode masked_view_only --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v34_masked_cross_view_prediction_seed13711 --score_modes latent_summary_knn --latent_knn_k 5 --no-progress
```

Overall result:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.524520 | 0.518496 | 0.3829 |
| full_5view | latent_summary_knn | 0.539228 | 0.531955 | 0.2457 |

Gate evaluation:

- full_5view beats matched logmel_only: `PASS`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least `2` bottleneck machines improve `TPR@99n`: `FAIL`, `0/5`

Key observation:

```text
masked cross-view prediction improves geometry and beats the matched baseline,
but it still does not produce the needed low-FPR tail separation.
```

The bottleneck pattern remains the same:

- normal `q95/q99` rise on bottleneck machines
- anomaly scores also rise, but not far enough beyond that normal extreme tail
- `TPR@99n` gets worse on `gearbox`, `slider`, `toycar`, `toytrain`, and `fan`

Verdict:

```text
V34 does not pass the single-seed gate.
```

Operational consequence:

```text
Do not continue to a V34 seed panel.
Do not change the mask strategy yet.
Record this as a masked cross-view failure analysis.
```

## 75. V35 Result: Temporal Predictive SSL

Next we switched from reconstruction-style SSL to a fixed temporal prediction
task:

```text
ssl_task = temporal_predictive_ssl
loss_view_mode = future_only
```

Rule:

- later half of the time axis is masked from the input
- target remains the original tensor
- loss is computed only on the later half
- score remains `latent_summary_knn`

No horizon search, no fusion, no per-machine rule, and no anomaly labels were
used.

### Single-Seed Run (`13711`)

Command:

```bash
python -m src.run_experiment --data_root /data1/test/LAES_2025/data/dcase --machines bearing,fan,gearbox,slider,toycar,toytrain,valve --experiments logmel_only,full_5view --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 --per_view_channels 16 --dropout_p 0.0 --ssl_task temporal_predictive_ssl --loss_view_mode future_only --model_type viewwise_unshared_late_fusion --seed 13711 --device cuda:0 --output_dir results/v35_temporal_predictive_ssl_seed13711 --score_modes latent_summary_knn --latent_knn_k 5 --no-progress
```

Overall result:

| experiment | score_mode | hmean | pAUC | KS |
|---|---|---:|---:|---:|
| logmel_only | latent_summary_knn | 0.535139 | 0.524887 | 0.3886 |
| full_5view | latent_summary_knn | 0.528608 | 0.522180 | 0.2743 |

Gate evaluation:

- full_5view beats matched logmel_only: `FAIL`
- `hmean > 0.549547`: `FAIL`
- `pAUC >= 0.542331`: `FAIL`
- `KS not clearly worse than 0.294880`: `PASS`
- at least `2` bottleneck machines improve `TPR@99n`: `FAIL`, `0/5`

Key finding:

```text
Temporal predictive SSL improves geometry, but not low-FPR tail separation.
```

The failure mechanism mirrors earlier failures:

- normal `q95/q99` rise on bottleneck machines
- anomaly scores rise too
- but anomaly tail does not move farther right than the normal extreme tail
- therefore `TPR@99n` stays flat or drops

Verdict:

```text
V35 does not pass the single-seed gate.
```

Operational consequence:

```text
Do not continue to a V35 seed panel.
Do not open V36 from this result.
Record this as a temporal-predictive SSL failure analysis.
```
