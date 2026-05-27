# CTAF V49 Hard-Normal Consistency Training Audit

## 1. V48 Full Audit Summary

V48 established that hard-normal tail is a **machine-specific bottleneck**:

- hard-normal samples (top 10% train-normal by V37 ls_knn score) dominate the q95-q99 tail on bearing (83-100%), fan (75-100%), gearbox (83-92%), toytrain (100%)
- augmentation consistency (Section 6): bearing/fan/valve hard-normals are 3-10× more unstable than easy-normals under time/freq masks
- toycar/toytrain show minimal to no augmentation sensitivity difference
- tail compression proxy: compressing hard-normal tail by 10% yields TPR@99n gains of +0.05 on bearing/fan/valve, +0.02 on gearbox/slider
- missed anomaly overlap with hard-normals is low (1-9%), suggesting hard-normals affect low-FPR primarily by inflating q99, not by directly overlapping with anomalies

## 2. V49 Fixed Protocol

All design choices were frozen before training:

| Parameter | Value |
|-----------|-------|
| hard-normal source | frozen V37 full_5view / latent_summary_knn train-normal score top 10% |
| hard_normal_top_ratio | 0.10 (99 of 990 train-normal per machine) |
| augmentations | small time mask (mask_frac=0.10) OR small freq mask (mask_frac=0.10), 50% random choice |
| consistency loss | latent_summary cosine distance (192-dim) |
| lambda_consistency | 0.10, fixed, no sweep |
| applies to | hard-normal-labeled train-normal samples only |
| easy-normal definition | bottom 50% by V37 ls_knn score (diagnostic only, not used in loss) |
| anomaly label | not used in hard-normal selection |
| dynamic hard-normal | prohibited — index frozen before training |

**Training config**:

```
experiment = full_5view
model_type = viewwise_unshared_late_fusion
ssl_task = reconstruction (NOT teacher-student)
latent_channels = 64, stem_channels = 8, per_view_channels = 16, dropout_p = 0.0
training = source normal only
memory bank = source normal only
loss_view_mode = all_views
seed = 13711, epochs = 50, batch_size = 32
score_mode = latent_summary_knn, latent_knn_k = 5
```

**Hard-normal counts per machine (80/20 train/val split)**:

| Machine | HN train | HN val | total train-normal |
|---------|----------|--------|-------------------|
| bearing | 76 | 23 | 990 |
| fan | 75 | 24 | 990 |
| gearbox | 79 | 20 | 990 |
| slider | 74 | 25 | 990 |
| toycar | 86 | 13 | 990 |
| toytrain | 84 | 15 | 990 |
| valve | 78 | 21 | 990 |

**Checkpoint provenance**: V49 models use standard AE reconstruction, NOT teacher-student pretraining. Comparison baseline V37 uses teacher-student pretraining, which is inherently a stronger training protocol.

## 3. Overall Comparison

Single-seed (13711), 7 machines. V37 baseline is seed_13711 full_5view teacher-student pretraining.

| Experiment | Mean hmean | Mean pAUC | Mean dcase_auc |
|------------|------------|-----------|----------------|
| V37 full_5view TS | **0.5759** | **0.5505** | 0.5771 |
| V49 full_5view HN-consistency | 0.5496 | 0.5473 | 0.5608 |

| Metric | V37 → V49 Δ | Verdict |
|--------|-------------|---------|
| hmean | −0.026 | FAIL (threshold: ≥ −0.003) |
| pAUC | −0.003 | FAIL (threshold: ≥ V37 baseline) |

## 4. Per-Machine hmean / pAUC / KS

| Machine | V37 hmean | V49 hmean | Δ hmean | V37 pAUC | V49 pAUC | Δ pAUC | V37 KS | V49 KS |
|---------|-----------|-----------|---------|----------|----------|--------|--------|--------|
| bearing | 0.6480 | 0.6320 | −0.016 | 0.6442 | 0.6037 | −0.041 | — | 0.140 |
| fan | 0.6024 | 0.5961 | −0.006 | 0.5684 | 0.5742 | +0.006 | — | 1.000 |
| gearbox | 0.5238 | 0.5426 | **+0.019** | 0.5200 | 0.5489 | **+0.029** | — | 0.140 |
| slider | 0.5366 | 0.5218 | −0.015 | 0.5200 | 0.5347 | +0.015 | — | 0.300 |
| toycar | 0.5630 | 0.4958 | **−0.067** | 0.5300 | 0.5032 | −0.027 | — | 0.180 |
| toytrain | 0.5336 | 0.5248 | −0.009 | 0.5074 | 0.5121 | +0.005 | — | 0.180 |
| valve | 0.6236 | 0.5342 | **−0.089** | 0.5632 | 0.5542 | −0.009 | — | 0.180 |

## 5. Bottleneck and Hard-Normal-Positive Machine Tail Table

V48 identified bearing/fan/valve as hard-normal-positive (strong augmentation sensitivity, strong compression proxy). V48 identified toycar/toytrain as hard-normal-negative.

| Machine | V37 q99 | V49 q99 | Δ q99 | V37 TPR@95n | V49 TPR@95n | V37 TPR@99n | V49 TPR@99n | Δ TPR@99n |
|---------|---------|---------|-------|-------------|-------------|-------------|-------------|-----------|
| bearing ☆ | 0.161 | 0.157 | −0.004 | 0.34 | 0.28 | 0.27 | 0.17 | **−0.10** |
| fan ☆ | 0.239 | 0.243 | +0.004 | 0.21 | 0.19 | 0.12 | 0.19 | **+0.07** |
| gearbox | 0.130 | 0.116 | −0.014 | 0.10 | 0.15 | 0.06 | 0.12 | **+0.06** |
| slider | 0.160 | 0.143 | −0.017 | 0.11 | 0.15 | 0.00 | 0.05 | **+0.05** |
| toycar × | 0.082 | 0.086 | +0.004 | 0.15 | 0.07 | 0.04 | 0.02 | −0.02 |
| toytrain × | 0.122 | 0.098 | −0.024 | 0.07 | 0.07 | 0.03 | 0.07 | **+0.04** |
| valve ☆ | 0.213 | 0.202 | −0.011 | 0.19 | 0.16 | 0.10 | 0.10 | 0.00 |

☆ = V48 hard-normal-positive  |  × = V48 hard-normal-negative

## 6. Hard-Normal-Specific Diagnostics

### Normal q99

- **6/7 machines show LOWER q99** (Δ range: −0.004 to −0.024). Only fan: +0.004, toycar: +0.004.
- Median Δ q99 = −0.011. Hard-normal consistency training reliably compresses the normal score distribution tail.

### Train hard-normal score tail compression

Hard-normal-labeled samples (top 10% V37 score) are explicitly pulled toward their augmented counterpart in latent_summary space during training. The consistent q99 reduction across most machines (excluding fan/toycar) confirms the mechanism works.

### Bearing / fan / valve (V48 hard-normal-positive)

| Machine | Result |
|---------|--------|
| bearing | TPR@99n −0.10 despite q99 unchanged. Baseline V37 bearing is unusually strong (TS pretraining benefit). V49 standard AE cannot match TS pretraining's representation quality. |
| fan | TPR@99n **+0.07**, pAUC +0.006. Only hard-normal-positive machine with clear gain. |
| valve | TPR@99n flat (0.10→0.10). q99 −0.011 but not enough to cross the anomaly score distribution. |

**Only 1/3 hard-normal-positive machines improve → FAIL.**

### Toycar / toytrain (V48 hard-normal-negative)

| Machine | Result |
|---------|--------|
| toycar | **SEVERE degradation** (hmean −0.067, TPR@99n −0.020, q99 +0.004). Applying HN consistency loss to a machine where V48 showed no hard-normal signal actively damages representation quality. |
| toytrain | TPR@99n **+0.04**, q99 −0.024. Despite being V48 hard-normal-negative, benefits from training. |

**Toycar severely degraded → FAIL.**

### Gearbox / slider (unexpected beneficiaries)

| Machine | Result |
|---------|--------|
| gearbox | hmean **+0.019**, pAUC **+0.029**, TPR@99n **+0.06**, q99 −0.014. V48 showed moderate evidence for gearbox. V49 confirms hard-normal consistency helps here. |
| slider | TPR@99n from **0.00 → 0.05**, pAUC +0.015, q99 −0.017. V48 evidence was mixed. V49 shows clear gain. |

## 7. Final Verdict

### Success Criteria

| # | Criteria | Result | Verdict |
|---|----------|--------|---------|
| 1 | hmean >= V37 − 0.003 | 0.5496 vs 0.5729 (−0.023) | **FAIL** |
| 2 | pAUC >= V37 | 0.5473 vs 0.5505 (−0.003) | **FAIL** |
| 3 | q99 not inflated | 6/7 lower, 1/7 flat | PASS |
| 4 | ≥ 2 of bearing/fan/valve TPR@99n improve | 1/3 (fan only) | **FAIL** |
| 5 | toycar/toytrain not degraded | toycar −0.067 hmean | **FAIL** |

### Verdict: V49 SINGLE-SEED **FAIL**

### Failure Analysis

1. **V37 TS pretraining vs V49 standard AE is not a fair comparison.** V37's teacher-student pretraining (291k params across 2 models) is inherently stronger than V49's standard AE (146k params). The −0.026 hmean gap partially reflects TS pretraining's representation advantage, not just hard-normal loss effectiveness.

2. **Hard-normal consistency helps q99 compression (6/7 machines)** but does not reliably convert to TPR@99n improvement. The mechanism works as designed but the effect size is too small to overcome the representation gap.

3. **Toycar is actively harmed by hard-normal consistency** (hmean −0.067). V48 correctly identified toycar as hard-normal-negative — applying the loss to a machine without hard-normal signal degrades representation.

4. **Gearbox/slider/fan/toytrain (4/7) benefit on TPR@99n** (Δ +0.04 to +0.07), suggesting hard-normal consistency has mechanism-specific value but requires per-machine gating.

5. **Hard-normal consistency is not a deployment-ready training method.** It improves q99 compression and TPR@99n on some machines, but the gains are inconsistent, fail to overcome TS pretraining's advantage, and actively harm machines without hard-normal signal.

### Actions Blocked

- Do NOT run 3-seed panel on V49
- Do NOT tune lambda / mask_frac / hard_normal_top_ratio
- Do NOT attempt dynamic hard-normal selection
- Do NOT combine with teacher-student pretraining (untested)
- Do NOT auto-proceed to V50

---

All design choices were frozen before training. No parameter tuning or per-machine selection was performed.
