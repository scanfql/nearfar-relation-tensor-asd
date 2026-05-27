# CTAF Stage 9: Non-Contrastive Invariance SSL Summary (V52–V53)

Stage 9 tested whether replacing AE reconstruction with a non-contrastive invariance SSL objective (VICReg) on full_5view could learn a tighter normal manifold and improve low-FPR kNN scoring, without changing the frontend, pooling, or scoring rule.

---

## 1. V52 Preregistration Summary

V52 defined Stage 9 as a new framework-level hypothesis, explicitly distinct from all previous CTAF search directions:

- **Not** frontend replacement (V39–V47)
- **Not** pooling replacement (V50)
- **Not** segment / patch scoring (V42–V45)
- **Not** hard-normal consistency (V48–V49)
- **Not** frozen-representation boundary scoring (V51)

The core hypothesis: non-contrastive invariance SSL (VICReg-style), trained on full_5view normal samples with mild augmentations, can produce a latent_summary space where the normal support manifold is more stable. This should naturally tighten the normal score tail and improve kNN's low-FPR separation — without using negative-pair contrastive learning (SimCLR would risk widening the normal manifold).

Key design choices frozen in V53 before training:

- VICReg only (no BYOL / SimSiam / SimCLR)
- encoder latent_summary for scoring (projector discarded after training)
- fixed loss weights (25/25/1), fixed augmentations (mask_frac=0.05, gain 0.95–1.05)
- single-seed pilot (seed=13711) before any 3-seed panel

---

## 2. V53 Fixed Protocol

| Parameter | Value |
|-----------|-------|
| SSL objective | VICReg |
| encoder | viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0) |
| representation | encoder latent_summary (channel-wise mean + std + max, 192-dim) |
| projector | MLP 192 → 256 → 256, ReLU, dropout=0.0 |
| λ_inv / λ_var / λ_cov | 25 / 25 / 1 |
| augmentations | time_mask (0.05) + freq_mask (0.05) + gain jitter ([0.95, 1.05]) |
| n_augmentations per view | 2 |
| time shift | not used |
| training data | source train-normal only |
| experiment | full_5view only |
| seed | 13711 only |
| scoring | latent_summary_knn, k=5 |

---

## 3. V53 Results

### Overall vs V37 Baseline (seed 13711)

| Metric | V37 baseline | V53 VICReg | Δ |
|--------|-------------|-----------|-----|
| hmean | 0.5631 | 0.4980 | **−0.065** |
| pAUC | 0.5505 | 0.5000 | **−0.050** |
| KS | 0.3143 | 0.3771 | +0.063 |

### Per-Machine hmean / pAUC

| Machine | V53 hmean | V37 hmean |
|---------|-----------|-----------|
| bearing | **0.532** | 0.648 |
| fan | 0.529 | 0.602 |
| gearbox | 0.541 | 0.524 |
| slider | 0.508 | 0.537 |
| toycar | 0.500 | 0.563 |
| toytrain | 0.469 | 0.534 |
| valve | **0.408** | 0.624 |

### Bottleneck Tail — Normal q99 Inflation

| Machine | V37 q99 | V53 q99 | V37 TPR@99n | V53 TPR@99n |
|---------|---------|---------|-------------|-------------|
| bearing | 0.161 | **0.241** | 0.27 | 0.02 |
| fan | 0.239 | **0.338** | 0.12 | 0.04 |
| gearbox | 0.130 | **0.216** | 0.06 | 0.04 |
| slider | 0.160 | **0.216** | 0.00 | 0.02 |
| toycar | 0.082 | **0.160** | 0.04 | 0.00 |
| toytrain | 0.122 | **0.157** | 0.03 | 0.00 |
| valve | 0.213 | **0.264** | 0.10 | 0.00 |

**All 7 machines show normal q99 inflation. TPR@99n improves on zero machines.**

### Success Gates

| # | Gate | Result |
|---|------|--------|
| 1 | hmean >= V37 − 0.003 | FAIL (−0.065) |
| 2 | pAUC >= V37 | FAIL (−0.050) |
| 3 | ≥2 bottleneck TPR@99n improve | FAIL (0/5) |
| 4 | bearing/valve not collapse | FAIL |
| 5 | toycar not degraded | FAIL |
| 6 | q99 not inflated | FAIL |
| 7 | gains not from single machine | FAIL |

**V53 single-seed FAIL. 7/7 gates fail.**

---

## 4. Representation Diagnostics

The model does not show trivial collapse to constant features:

- latent_summary feature std: 0.20–0.48 (active representation)
- projector output std: 1.03–1.16 (normal spread)
- VICReg variance loss: 0.004–0.011 (non-zero activity)
- VICReg covariance loss: near 0 to ~1.6 (some decorrelation)

**However, non-collapse alone is insufficient.** The learned invariance appears to smooth normal and anomaly structure together rather than selectively tightening the normal manifold. Both normal q99 and anomaly median rise — the invariance objective compresses everything, not just normal variation. The signal that kNN relied on at low FPR is lost.

---

## 5. Final Conclusion

**V53 is a scientific negative under the fixed Stage-9 protocol.**

Blocked actions:

- Do **not** run 3-seed panel
- Do **not** tune VICReg weights (25/25/1)
- Do **not** tune augmentations (mask_frac, gain range)
- Do **not** try BYOL / SimSiam inside this stage
- Do **not** auto-proceed to V54
- Stage 9 non-contrastive invariance SSL branch is **closed**

V53 closes the fixed non-contrastive invariance SSL branch. Further improvement requires a genuinely new hypothesis or external representation source, not another local modification of the current CTAF pipeline.
