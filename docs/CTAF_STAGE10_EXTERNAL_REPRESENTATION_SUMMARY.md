# CTAF Stage 10: External Pretrained Representation Summary (V54–V56A)

Stage 10 tested whether an external frozen pretrained audio representation (BEATs) could complement or replace the CTA-F full_5view pipeline. The evidence confirms that BEATs provides machine-level complementary signal but lacks sample-level distillable structure for safe integration into CTAF training.

---

## 1. V54/V55 — Frozen BEATs + kNN

**Hypothesis**: frozen BEATs (AudioSet-pretrained 90M model) with kNN scoring can outperform CTA-F full_5view + V37 TS + latent_summary_knn without any CTAF training.

**Protocol**: raw waveform → BEATs fbank preprocessing → final-layer frame embeddings → time-mean pooling → cosine kNN (k=5) on source train-normal memory.

**Results** (seed 13711, 7 machines):

| Metric | V37 CTAF | V55 BEATs | Δ |
|--------|---------|-----------|-----|
| mean hmean | 0.5631 | **0.5786** | +0.015 |
| mean pAUC | **0.5505** | 0.5388 | −0.012 |
| bearing TPR@99n | **0.27** | 0.07 | −0.20 |
| valve TPR@99n | 0.10 | **0.18** | +0.08 |
| normal q99 inflated | — | 6/7 machines | — |

**Verdict**: frozen BEATs hmean surpasses V37, but pAUC fails, q99 inflates on most machines, and bearing collapses. BEATs provides machine-level complementary signal (strong on valve/gearbox, weak on bearing/fan) but cannot directly replace CTAF.

---

## 2. V56A — BEATs-to-CTAF Distillability Diagnostic

**Purpose**: determine whether BEATs embeddings can be linearly predicted from CTA-F V37 latent_summary at the sample level. If yes, BEATs could serve as a distillation teacher during CTAF training. If no, cross-architecture distillation is not justified.

**Protocol**: no-training audit. Ridge regression (α=1.0, 5-fold CV), residual analysis, neighbor preservation (k=5 cosine), and score correlation (Spearman ρ) on aligned V37 latent_summary (192-dim) and V55 BEATs embeddings (768-dim) for all 7 machines.

**Alignment**: 7/7 machines. 990 train / 200 test each.

### Ridge Regression R²

| Machine | R² mean | R² std |
|---------|---------|--------|
| valve | 0.676 | 0.009 |
| toytrain | 0.639 | 0.008 |
| fan | 0.586 | 0.005 |
| slider | 0.490 | 0.014 |
| toycar | 0.485 | 0.006 |
| bearing | 0.343 | 0.015 |
| gearbox | 0.286 | 0.020 |

6/7 R² ≥ 0.30 (PASS). But bearing R²=0.343 is below the mean−0.10 threshold of 0.401 (FAIL).

### Residual Analysis

| Machine | Residual z vs all |
|---------|-------------------|
| bearing | **+1.35** |
| gearbox | +1.22 |
| toycar | +0.29 |
| slider | +0.28 |
| fan | −0.72 |
| toytrain | −1.02 |
| valve | −1.40 |

bearing has the largest residual, exceeding the z≤1.0 threshold. BEATs information on bearing is structurally harder to predict from CTAF latent_summary than for other machines.

### Neighbor Preservation

| Machine | Recall@5 |
|---------|----------|
| slider | 0.218 |
| fan | 0.155 |
| toytrain | 0.108 |
| bearing | 0.094 |
| gearbox | 0.036 |
| valve | 0.026 |
| toycar | 0.011 |

Mean recall@5 = 0.0925 (FAIL, threshold ≥ 0.10). CTAF and BEATs have systematically different local neighborhood structures—sample-level topology does not transfer.

### Score Correlation

| Machine | Spearman ρ |
|---------|-----------|
| fan | **0.820** |
| gearbox | 0.524 |
| bearing | 0.429 |
| valve | 0.380 |
| slider | 0.378 |
| toytrain | 0.218 |
| toycar | **−0.161** |

6/7 ρ ≥ 0.20 (PASS). But toycar ρ = −0.161—BEATs and CTAF rank anomalies in opposite directions for this machine.

### Success Gates

| # | Gate | Result |
|---|------|--------|
| 1 | ≥4 machines R²≥0.30 | 6/7 PASS |
| 2 | bearing/fan R² ≥ mean−0.10 | FAIL (bearing 0.343 < 0.401) |
| 3 | mean recall@5 ≥ 0.10 | FAIL (0.0925) |
| 4 | ≥4 machines score ρ≥0.20 | 6/7 PASS |
| 5 | bearing/fan/toycar residual z≤1.0 | FAIL (bearing 1.35) |
| 6 | data aligned | 7/7 PASS |

**V56A Verdict: FAIL. BEATs has machine-level complementary signal, but not sample-level distillable structure. Do not train V56.**

---

## 3. Interpretation

1. **BEATs has clear machine-level complementarity with CTAF.** BEATs outperforms CTAF on valve/toytrain/gearbox/slider in hmean. CTAF outperforms BEATs on bearing/fan. This pattern is consistent with V55's per-machine results.

2. **This complementarity does not translate to sample-level distillability.** The Ridge R² for bearing (0.343) is structurally lower than the fleet average. bearing's residual is the highest. Neighbor recall is below threshold fleet-wide. BEATs and CTAF agree on which samples are anomalous at a coarse level (score ρ ≥ 0.2 for 6/7), but disagree on sample-level topology (recall@5 = 0.09).

3. **Direct distillation would risk the same pattern as every previous CTAF modification.** Forcing BEATs structure onto CTAF through distillation would likely improve valve/toytrain at the cost of degrading bearing—where CTAF's hmean=0.648 is the strongest result in the entire CTAF series. This is the same tradeoff that V40, V45, V49, V50, and V53 all exhibited.

4. **BEATs should be treated as external evidence, not as a CTAF training signal.** Its value is in demonstrating that (a) external pretrained audio representations can provide strong global anomaly ranking, but may still require careful low-FPR adaptation, and (b) the specific machines where CTAF struggles (valve, gearbox) are not inherently impossible—they are addressable with stronger representations. This supports the Stage 7 conclusion that the bottleneck is representation learning quality under the current training framework, not merely scoring rules or frontend design.

---

## 4. Final Conclusion

- Do **not** train V56 (BEATs-to-CTAF distillation)
- Do **not** pursue direct BEATs-to-CTAF cross-architecture distillation under the current protocol
- CTAF full_5view frontend remains the main self-developed representation line
- BEATs line is useful as external evidence/reference, not as a replacement or teacher for CTAF frontend in the current protocol

V56A closes the direct BEATs-to-CTAF distillation branch. Further work should preserve CTAF full_5view as the main frontend unless a new preregistered hypothesis provides a safer way to use external pretraining.
