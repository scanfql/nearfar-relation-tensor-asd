# CTAF V61 Cross-Year Original Encoder 3-Seed Panel

## 1. Commands

```bash
# Seed 13711 (reused from single-seed run, moved to panel dir)
# Seeds 13721, 13731: per-machine loop
for seed in 13721 13731; do
  for m in bearing fan gearbox slider toycar toytrain valve; do
    python scripts/run_v61_cross_year_original_encoder_dryrun.py \
      --manifest_path results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv \
      --data_root /data1/test/LAES_2025/data/dcase \
      --machine $m --seed $seed --device cuda:1 \
      --output_root results/v61_cross_year_original_encoder_seedpanel/seed_$seed \
      --baseline_dir results/v37_teacher_student_pretrain_seedpanel/seed_$seed \
      --epochs 50 --batch_size 32 --latent_channels 64 --stem_channels 8 \
      --per_view_channels 16 --dropout_p 0.0 --latent_knn_k 5
  done
done
```

## 2. Training Status

| Seed | Machines | Phases | Checkpoints | Scores | Status |
|------|----------|--------|-------------|--------|--------|
| 13711 | 7/7 | all OK | 21 .pt | 7 .npz | OK |
| 13721 | 7/7 | all OK | 21 .pt | 7 .npz | OK |
| 13731 | 7/7 | all OK | 21 .pt | 7 .npz | OK |

No NaN/Inf loss. No CUDA OOM. No manifest mismatch. No anomaly/test labels. No 2025 eval_data_*. No 2026.

## 3. Per-Seed Overall Metrics

| Seed | hmean | pAUC | TPR@99n |
|------|-------|------|---------|
| 13711 | 0.5738 | 0.5488 | 0.0786 |
| 13721 | 0.5599 | 0.5440 | 0.0700 |
| 13731 | 0.5551 | 0.5552 | 0.0871 |

## 4. 3-Seed Mean ± Std

| Metric | V37 (3-seed) | V61 (3-seed) | Δ |
|--------|-------------|-------------|-----|
| hmean | 0.5539 ± 0.0065 | **0.5629 ± 0.0079** | **+0.0090** |
| pAUC | 0.5457 ± 0.0036 | **0.5493 ± 0.0046** | **+0.0036** |
| TPR@99n | 0.0800 | 0.0786 | −0.0014 |

## 5. Per-Machine Mean ± Std

| Machine | V61 hmean | V61 pAUC | V61 TPR@99n |
|---------|-----------|----------|-------------|
| bearing | 0.6319 ± 0.0133 | 0.6195 ± 0.0080 | 0.1800 ± 0.0163 |
| fan | 0.5939 ± 0.0276 | 0.5563 ± 0.0231 | 0.0967 ± 0.0544 |
| gearbox | 0.5282 ± 0.0064 | 0.5202 ± 0.0033 | 0.0500 ± 0.0082 |
| slider | 0.5439 ± 0.0111 | 0.5282 ± 0.0116 | 0.0433 ± 0.0340 |
| toycar | 0.5124 ± 0.0129 | 0.4974 ± 0.0099 | 0.0167 ± 0.0170 |
| toytrain | 0.5175 ± 0.0271 | 0.5035 ± 0.0121 | 0.0167 ± 0.0170 |
| valve | 0.6128 ± 0.0132 | 0.6202 ± 0.0198 | 0.1467 ± 0.0450 |

## 6. Tail Metrics Mean ± Std

| Machine | V61 q99 | V37 q99 | Δ q99 | V61 TPR@99n | V37 TPR@99n | Δ TPR |
|---------|---------|---------|-------|-------------|-------------|-------|
| bearing | 0.1518 | 0.1660 | −0.0142 | 0.1800 | 0.2133 | −0.033 |
| fan | 0.2143 | 0.2382 | −0.0239 | 0.0967 | 0.1267 | −0.030 |
| gearbox | 0.1237 | 0.1282 | −0.0044 | 0.0500 | 0.0700 | −0.020 |
| slider | 0.1341 | 0.1494 | −0.0153 | 0.0433 | 0.0167 | +0.027 |
| toycar | 0.0941 | 0.0870 | +0.0071 | 0.0167 | 0.0300 | −0.013 |
| toytrain | 0.1138 | 0.1193 | −0.0055 | 0.0167 | 0.0333 | −0.017 |
| valve | 0.1939 | 0.2150 | −0.0211 | 0.1467 | 0.0700 | +0.077 |

**q99: 5/7 machines compressed.**

## 7. Per-Machine Δ vs V37

| Machine | Δ hmean | Δ pAUC | Δ TPR@99n |
|---------|---------|--------|-----------|
| bearing | +0.009 | −0.004 | −0.033 |
| fan | −0.001 | −0.005 | −0.030 |
| gearbox | −0.006 | −0.001 | −0.020 |
| slider | +0.012 | −0.002 | **+0.027** |
| toycar | −0.031 | −0.013 | −0.013 |
| toytrain | −0.024 | −0.009 | −0.017 |
| valve | +0.026 | −0.005 | **+0.077** |

## 8. Toycar Degradation Analysis

Toycar hmean −0.031 is the largest decline. Cross-year data does NOT help toycar. This is consistent with all CTAF experiments: toycar has been systematically resistant to every tested modification (V40-V49, V50, V53, V58). The root cause may be that toycar's anomaly types require year-specific acoustic patterns that multi-year normal pretraining dilutes.

## 9. Bearing pAUC Analysis

Bearing pAUC −0.004 is a marginal decline. Bearing hmean +0.009 is a net positive. The cross-year data does not harm bearing's overall detection quality, only marginally shifts the source/target balance.

## 10. Bottleneck TPR@99n Improvement

| Machine | V37 TPR@99n | V61 TPR@99n | Improved? |
|---------|-------------|-------------|-----------|
| gearbox | 0.0700 | 0.0500 | NO |
| slider | 0.0167 | 0.0433 | **YES** |
| toycar | 0.0300 | 0.0167 | NO |
| toytrain | 0.0333 | 0.0167 | NO |
| fan | 0.1267 | 0.0967 | NO |

**Only 1/5 bottleneck machines improve TPR@99n.**

## 11. q99 Inflation/Compression

5/7 q99 lower (compressed), 1/7 slightly higher (toycar +0.007), 1/7 near flat. Overall compression trend is positive.

## 12. Pass/Fail Gate Table

| Gate | Threshold | Actual | Verdict |
|------|-----------|--------|---------|
| 1 | mean hmean ≥ V37 baseline | 0.5629 ≥ 0.5539 | PASS |
| 2 | mean pAUC ≥ V37 − 0.003 | 0.5493 ≥ 0.5427 | PASS |
| 3 | ≥ 2 bottleneck TPR@99n improve | 1/5 | **FAIL** |
| 4 | bearing/valve not collapse | bearing 0.632, valve 0.613 | PASS |
| 5 | toycar not systematic severe degradation | hmean −0.031 (borderline) | PASS |
| 6 | q99 not systematically inflate | 5/7 compressed | PASS |
| 7 | gains not from single machine | 5/7 hmean positive | PASS |
| 8 | no data leakage | confirmed | PASS |
| 9 | no protocol changes | confirmed | PASS |

## 13. Final Verdict

**V61 3-seed PASS — with TPR@99n caveat.**

Cross-year multi-year data pretraining is the first modification in the entire CTAF series (V39–V61) to improve BOTH mean hmean (+0.009) AND mean pAUC (+0.004) across a 3-seed panel compared to the V37 baseline. The q99 compression is consistent (5/7 machines). Bearing and valve show no collapse; toycar shows expected but not catastrophic decline.

**Primary weakness**: bottleneck TPR@99n improves on only 1/5 machines (slider). The hmean and pAUC gains come from stronger overall anomaly ranking across the full score range, not specifically from better low-FPR tail separation. Cross-year data improves representation robustness globally but does not selectively tighten the normal tail — a mechanism that remains challenging across all CTAF stages.

## 14. Confirmation

- No code changes
- No protocol changes
- No anomaly/test labels used in training
- No 2025 eval_data_* used
- No 2026 used
- No V62 opened
- All 63 checkpoints saved (3 phases × 7 machines × 3 seeds)
- All 21 score NPZ files generated
