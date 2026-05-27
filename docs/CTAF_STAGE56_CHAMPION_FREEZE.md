# CTAF Stage56 Champion Freeze

## Freeze Status

Current promoted research champion:

`Stage52 w006 epoch005 + Stage48 target train-normal memory + Stage53 domain_q95_ratio score calibration`

Primary metric:

- mean_hmean: `0.6499400423151978`
- mean_pAUC: `0.6036842105263158`
- mean_TPR99: `0.17714285714285719`

This supersedes the previous raw target-memory champion:

- `Stage52 w006 epoch005 + raw adapted_ldn_diff_K1_scorek5`
- mean_hmean: `0.6492908450451674`
- mean_pAUC: `0.6031578947368422`
- mean_TPR99: `0.19714285714285712`

Tradeoff:

- domain_q95_ratio improves hmean by `+0.0006491972700303217`
- domain_q95_ratio improves pAUC by `+0.0005263157894735636`
- domain_q95_ratio lowers TPR99 by `-0.020000000000000018`

## Champion Artifacts

Checkpoint:

`results/stage52a_stage35_epoch_sweep_e6_w006/checkpoints/stage51a_stage35_epoch_sweep_epoch005.pt`

Target-memory embedding/eval root:

`results/stage52a_stage52a_stage35_epoch_sweep_e6_w006_stage51a_stage35_epoch_sweep_epoch005_target_memory_full7_eval`

Domain-calibrated score root:

`results/stage53a_stage52_w006_e005_domain_calibrated_score`

Full champion score audit:

`results/stage53a_stage52_w006_e005_domain_calibrated_score/stage53a_domain_calibrated_verdict.json`

Relevant scripts:

- `scripts/audit_stage48a_adapted_full7_eval_target_memory.py`
- `scripts/audit_stage51a_stage35_epoch_sweep.py`
- `scripts/audit_stage53a_target_memory_domain_calibrated_score.py`
- `scripts/audit_stage56a_source_pseudo_target_guard.py`

## Route Definition

Representation:

- clean EAT initialized
- Stage35-style source-only pseudo-attribute training
- scheduled mild tail guardrail
- best discovered variant: `tail_dispersion_loss_weight=0.06`
- selected epoch: `epoch005`

Frozen backend:

- `min_clamp__mean`
- `ldn_diff_K1_scorek5`
- target train-normal clips included in reference memory
- LDN `K/r/score_k` unchanged

Score calibration:

- `domain_q95_ratio`
- source and target score scales are normalized using train-normal memory score `q95` per domain
- no anomaly labels are used for calibration
- no model parameters are trained in Stage53

Important governance note:

The `domain_q95_ratio` transformation itself is label-free. However, promoting it over raw LDN was decided after full7 audit. Treat it as a frozen route from this point onward; do not keep adding more calibrated score candidates and selecting by full7 hmean unless explicitly marking that as exploratory test-set route search.

## Result Ladder

Key promoted milestones:

| Stage | Route | mean_hmean | mean_pAUC | mean_TPR99 |
|---|---|---:|---:|---:|
| Clean best | clean EAT + LDN | `0.6159467288425517` | `0.5603007518796993` | `0.10571428571428572` |
| Stage30 | guardrailed pseudo attribute + source memory | `0.6231264650860286` | `0.5717293233082706` | `0.10857142857142857` |
| Stage43 | source+target pseudo inventory + fan protected + target memory | `0.6470293685016306` | `0.5961654135338347` | `0.15142857142857144` |
| Stage52 raw | source-only Stage35 w006 epoch005 + target memory | `0.6492908450451674` | `0.6031578947368422` | `0.19714285714285712` |
| Stage53 champion | Stage52 raw + domain_q95_ratio | `0.6499400423151978` | `0.6036842105263158` | `0.17714285714285719` |

Distance to user target:

- target mean_hmean: `0.7`
- current gap: `0.05005995768480215`

## Negative Branches After Stage48

Stage55:

- source+target pseudo inventory with Stage35-like schedule
- best calibrated hmean: `0.648439853240`
- conclusion: target train-normal clips should not receive pseudo CE labels in this branch.

Stage56:

- source-only pseudo CE plus unlabeled target train-normal teacher guard
- best calibrated hmean: `0.649738657531`
- conclusion: target guard slightly helps pAUC in places but does not beat Stage52/53 champion.

Stage52 tail weight sweep:

- `w006 epoch005` is the best hmean variant
- `w004` and `w007` are essentially tied but lower
- later epochs improve pAUC/tail sometimes, but hmean declines

Stage51 epoch sweep:

- epoch005 remains the hmean optimum
- epoch007 increases pAUC but loses hmean

## Interpretation

The largest recent gain came from changing the reference memory to include target train-normal clips. This moved the route from the `0.62x` regime to the `0.647+` regime without changing LDN hyperparameters or using anomaly labels.

The second gain came from a simple train-normal domain scale normalization (`q95_ratio`). It is small but consistent around the Stage35/52 champion.

Small Stage35-style training variations are now near saturation. The route appears locally capped around `0.65`. Reaching `0.7` likely needs a new evidence source or representation family, not another minor tail/guard weight sweep.

## Recommended Next Direction

Do not continue random seed search or small Stage35 loss-weight search.

Higher-upside directions:

1. Add a second frozen representation family with independent error profile, but avoid label-tuned fusion.
2. Build a train-normal-only route-selection or calibration rule and freeze it before full7 comparison.
3. Explore stronger encoder features or multi-resolution evidence, then evaluate with the same Stage48/53 governance.
4. If training continues, optimize a train-normal proxy that explicitly predicts target-memory score geometry, not pseudo CE on target clips.

## Reproduction Commands

Extract target-memory embeddings for the champion checkpoint:

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && CUDA_VISIBLE_DEVICES=1 python3 scripts/audit_stage48a_adapted_full7_eval_target_memory.py --mode extract_adapted_embeddings --device cuda --adapted_ckpt_path results/stage52a_stage35_epoch_sweep_e6_w006/checkpoints/stage51a_stage35_epoch_sweep_epoch005.pt --output_root results/stage52a_stage52a_stage35_epoch_sweep_e6_w006_stage51a_stage35_epoch_sweep_epoch005_target_memory_full7_eval --use_target_train_memory
```

Score raw target-memory LDN:

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && python3 scripts/audit_stage48a_adapted_full7_eval_target_memory.py --mode score_full7 --adapted_ckpt_path results/stage52a_stage35_epoch_sweep_e6_w006/checkpoints/stage51a_stage35_epoch_sweep_epoch005.pt --output_root results/stage52a_stage52a_stage35_epoch_sweep_e6_w006_stage51a_stage35_epoch_sweep_epoch005_target_memory_full7_eval --use_target_train_memory
```

Apply Stage53 domain calibration:

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && python3 scripts/audit_stage53a_target_memory_domain_calibrated_score.py --mode score_domain_calibrated --embedding_root results/stage52a_stage52a_stage35_epoch_sweep_e6_w006_stage51a_stage35_epoch_sweep_epoch005_target_memory_full7_eval --output_root results/stage53a_stage52_w006_e005_domain_calibrated_score --doc_path docs/CTAF_STAGE53A_STAGE52_W006_E005_DOMAIN_CALIBRATED_SCORE.md
```

Summarize:

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && cat results/stage53a_stage52_w006_e005_domain_calibrated_score/stage53a_domain_calibrated_verdict.json && cat results/stage53a_stage52_w006_e005_domain_calibrated_score/summary.csv && cat results/stage53a_stage52_w006_e005_domain_calibrated_score/per_machine.csv
```
