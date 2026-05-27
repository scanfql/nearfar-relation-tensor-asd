# CTAF Low-FPR Failure Attribution

Reference scoring source: `V37 seed_13711 / full_5view / latent_summary_knn`.

## gearbox

- output dir: `/data1/test/LAES_2025/new_projects/clean_tensor_asd/results/low_fpr_diagnostic/gearbox`
- missed anomalies: `94`
- normal-tail samples: `4`
- detected anomalies: `6`
- normal q95 / q99: `0.117578 / 0.129624`
- missed anomaly visually distinguishable on 5-view input: `weak`
- missed anomaly mixed with normal-tail in latent space: `yes`
- candidate frontend evidence from modulation/cepstral/transient views: `weak/unclear`

Group diagnostics:

- missed anomaly: count=20, mean_score=0.110715, mean_nn_dist=0.106153, mean_input_diff=1.045067
- normal tail: count=4, mean_score=0.125715, mean_nn_dist=0.122535, mean_input_diff=1.246972
- detected anomaly: count=6, mean_score=0.145582, mean_nn_dist=0.129997, mean_input_diff=1.165100

Attribution: `A. input-limited`

- rationale: Missed anomalies look too similar to normal-tail samples already at the input level.

## slider

- output dir: `/data1/test/LAES_2025/new_projects/clean_tensor_asd/results/low_fpr_diagnostic/slider`
- missed anomalies: `100`
- normal-tail samples: `4`
- detected anomalies: `0`
- normal q95 / q99: `0.122654 / 0.159626`
- missed anomaly visually distinguishable on 5-view input: `weak`
- missed anomaly mixed with normal-tail in latent space: `yes`
- candidate frontend evidence from modulation/cepstral/transient views: `weak/unclear`

Group diagnostics:

- missed anomaly: count=20, mean_score=0.127062, mean_nn_dist=0.118967, mean_input_diff=1.024112
- normal tail: count=4, mean_score=0.137642, mean_nn_dist=0.130454, mean_input_diff=1.204183
- detected anomaly: count=0, mean_score=0.000000, mean_nn_dist=0.000000, mean_input_diff=0.000000

Attribution: `A. input-limited`

- rationale: Missed anomalies look too similar to normal-tail samples already at the input level.

## toycar

- output dir: `/data1/test/LAES_2025/new_projects/clean_tensor_asd/results/low_fpr_diagnostic/toycar`
- missed anomalies: `96`
- normal-tail samples: `4`
- detected anomalies: `4`
- normal q95 / q99: `0.077860 / 0.082011`
- missed anomaly visually distinguishable on 5-view input: `weak`
- missed anomaly mixed with normal-tail in latent space: `yes`
- candidate frontend evidence from modulation/cepstral/transient views: `weak/unclear`

Group diagnostics:

- missed anomaly: count=20, mean_score=0.077969, mean_nn_dist=0.074442, mean_input_diff=0.950893
- normal tail: count=4, mean_score=0.081031, mean_nn_dist=0.077188, mean_input_diff=0.889230
- detected anomaly: count=4, mean_score=0.085749, mean_nn_dist=0.081117, mean_input_diff=0.912368

Attribution: `A. input-limited`

- rationale: Missed anomalies look too similar to normal-tail samples already at the input level.

## toytrain

- output dir: `/data1/test/LAES_2025/new_projects/clean_tensor_asd/results/low_fpr_diagnostic/toytrain`
- missed anomalies: `97`
- normal-tail samples: `4`
- detected anomalies: `3`
- normal q95 / q99: `0.095072 / 0.122015`
- missed anomaly visually distinguishable on 5-view input: `weak`
- missed anomaly mixed with normal-tail in latent space: `yes`
- candidate frontend evidence from modulation/cepstral/transient views: `weak/unclear`

Group diagnostics:

- missed anomaly: count=20, mean_score=0.085777, mean_nn_dist=0.081862, mean_input_diff=0.897727
- normal tail: count=4, mean_score=0.106328, mean_nn_dist=0.101052, mean_input_diff=1.152564
- detected anomaly: count=3, mean_score=0.132519, mean_nn_dist=0.125515, mean_input_diff=1.122960

Attribution: `A. input-limited`

- rationale: Missed anomalies look too similar to normal-tail samples already at the input level.

## fan

- output dir: `/data1/test/LAES_2025/new_projects/clean_tensor_asd/results/low_fpr_diagnostic/fan`
- missed anomalies: `88`
- normal-tail samples: `4`
- detected anomalies: `12`
- normal q95 / q99: `0.229054 / 0.238505`
- missed anomaly visually distinguishable on 5-view input: `weak`
- missed anomaly mixed with normal-tail in latent space: `yes`
- candidate frontend evidence from modulation/cepstral/transient views: `weak/unclear`

Group diagnostics:

- missed anomaly: count=20, mean_score=0.227570, mean_nn_dist=0.221616, mean_input_diff=2.063235
- normal tail: count=4, mean_score=0.237100, mean_nn_dist=0.232164, mean_input_diff=1.935114
- detected anomaly: count=12, mean_score=0.250269, mean_nn_dist=0.246027, mean_input_diff=2.197668

Attribution: `A. input-limited`

- rationale: Missed anomalies look too similar to normal-tail samples already at the input level.

## Final Attribution

- A = input-limited
- B = encoder-limited
- C = scoring-limited
- D = unclear

Next-step suggestion: Prefer modulation / cepstral / transient frontend work.

This diagnostic is offline only. No new model was trained, no score was tuned, and no per-machine method was selected.
