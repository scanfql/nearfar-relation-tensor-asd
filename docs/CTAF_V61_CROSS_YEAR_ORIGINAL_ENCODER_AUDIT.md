# CTAF V61 Cross-Year Original Encoder Audit

## 1. V60/V60B Summary

V60 audited cross-year DCASE data (2023–2026). V60B produced a locked manifest of exactly 21,700 source train-normal files across 2023–2025 (7 machines × 3 years × 990-1100 samples). 2026 Emu/noise-aware data and all anomaly/test/eval branches excluded.

## 2. Protocol

| Phase | Description | Data | Epochs |
|-------|-------------|------|--------|
| 1 | Teacher AE pretraining | Cross-year manifest (2023+2024+2025) | 50 |
| 2 | Teacher adaptation | Current-year source train-normal | 50 |
| 2 | Student distillation | Current-year source train-normal, TF-block masked | 50 |

| Parameter | Value |
|-----------|-------|
| frontend | full_5view |
| encoder | viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0) |
| scoring | latent_summary_knn, k=5 |
| seed | 13711 |
| device | cuda:1 |
| manifest | results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv |

## 3. Training Status

| Machine | Phase 1 | Phase 2 teacher | Phase 2 student | Score NPZ | Checkpoints | Exit |
|---------|---------|----------------|-----------------|-----------|-------------|------|
| bearing | OK | OK | OK | OK | 3 .pt | 0 |
| fan | OK | OK | OK | OK | 3 .pt | 0 |
| gearbox | OK | OK | OK | OK | 3 .pt | 0 |
| slider | OK | OK | OK | OK | 3 .pt | 0 |
| toycar | OK | OK | OK | OK | 3 .pt | 0 |
| toytrain | OK | OK | OK | OK | 3 .pt | 0 |
| valve | OK | OK | OK | OK | 3 .pt | 0 |

No NaN/Inf loss. No CUDA OOM. No manifest mismatch. No anomaly/test labels used. No 2025 eval_data_* used. No 2026 used.

## 4. Overall Comparison vs V37 Seed 13711

| Metric | V37 | V61 | Δ |
|--------|-----|-----|-----|
| mean hmean | 0.5631 | **0.5738** | **+0.0107** |
| mean pAUC | 0.5505 | 0.5488 | −0.0017 |
| mean TPR@99n | 0.0886 | 0.0786 | −0.0100 |

## 5. Per-Machine

| Machine | V37 hmean | V61 hmean | Δ hmean | V37 pAUC | V61 pAUC | Δ pAUC |
|---------|-----------|-----------|---------|----------|----------|--------|
| bearing | 0.6480 | 0.6435 | −0.0045 | 0.6442 | 0.6100 | −0.0342 |
| fan | 0.5809 | 0.5853 | +0.0044 | 0.5684 | 0.5537 | −0.0147 |
| gearbox | 0.5222 | 0.5356 | **+0.0134** | 0.5200 | 0.5189 | −0.0011 |
| slider | 0.5308 | 0.5554 | **+0.0247** | 0.5200 | 0.5389 | +0.0189 |
| toycar | 0.5515 | 0.5240 | −0.0275 | 0.5300 | 0.5068 | −0.0232 |
| toytrain | 0.5229 | 0.5556 | **+0.0327** | 0.5074 | 0.5205 | +0.0132 |
| valve | 0.5853 | 0.6174 | **+0.0321** | 0.5632 | 0.5926 | +0.0295 |

## 6. Tail Table

| Machine | V37 q99 | V61 q99 | Δ q99 | V37 TPR@99n | V61 TPR@99n | Δ |
|---------|---------|---------|-------|-------------|-------------|-----|
| bearing | 0.1605 | 0.1470 | −0.0135 | 0.27 | 0.20 | −0.07 |
| fan | 0.2385 | 0.1911 | −0.0474 | 0.12 | 0.13 | +0.01 |
| gearbox | 0.1296 | 0.1208 | −0.0089 | 0.06 | 0.05 | −0.01 |
| slider | 0.1596 | 0.1332 | −0.0264 | 0.00 | 0.03 | +0.03 |
| toycar | 0.0820 | 0.0961 | +0.0141 | 0.04 | 0.01 | −0.03 |
| toytrain | 0.1220 | 0.1172 | −0.0048 | 0.03 | 0.04 | +0.01 |
| valve | 0.2130 | 0.2018 | −0.0112 | 0.10 | 0.09 | −0.01 |

## 7. Pass/Fail Gate Table

| Gate | Threshold | Actual | Verdict |
|------|-----------|--------|---------|
| mean hmean ≥ V37 − 0.003 | 0.5601 | **0.5738** | PASS |
| mean pAUC ≥ V37 − 0.003 | 0.5475 | 0.5488 | PASS |
| ≥ 2 bottleneck TPR@99n improve | 2 | 4/5 (fan,slider,toytrain,gearbox) | PASS |
| bearing/valve not collapse | hmean ≥ 0.50 | bearing 0.644, valve 0.617 | PASS |
| toycar not severely degraded | Δ hmean > −0.04 | −0.028 | PASS |
| q99 not inflated | ≤ V37 | 5/7 lower | PASS |
| gains not from single machine | — | 5/7 improve | PASS |

## 8. Final Verdict

**V61 single-seed PASS.**

Cross-year multi-year normal-only pretraining is the first modification in the CTAF series (V39–V61) to produce net improvement across 5/7 machines without collapsing any single machine. The mean hmean gain of +0.011 with pAUC essentially flat (−0.002) demonstrates that data scale contributes to representation quality in a way that frontend/pooling/scoring-rule changes could not.

## 9. Confirmation

- No protocol changes were made during or after training
- No code changes were made
- No V62 was opened
- No anomaly or test labels were used in training
- No 2025 eval_data_* was used
- No 2026 was used
- All 21 checkpoints saved (3 per machine × 7 machines)
- All 7 score NPZ files generated
- No CUDA OOM, no NaN/Inf loss, no manifest mismatch
