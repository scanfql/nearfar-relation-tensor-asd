# CTAF V56A BEATs-to-CTAF Distillability Diagnostic

## 1. V55 Summary

V55 tested frozen BEATs + kNN as an external representation:
- hmean 0.5786 > V37 0.5631 (PASS)
- pAUC 0.5388 < V37 0.5505 (FAIL)
- 6/7 machines normal q99 inflation
- bearing collapse (TPR@99n 0.27→0.07), valve breakout (0.10→0.18)
- BEATs has machine-level complementary signal but cannot directly replace CTAF

## 2. V56A Motivation

Determine whether BEATs embeddings are linearly predictable from CTA-F
V37 latent_summary at the sample level. If yes, BEATs may serve as a
distillation teacher. If no, BEATs is only machine-level complementary,
and cross-architecture distillation is not justified.

Fixed: Ridge α=1.0, CV=5-fold, k=5.
No training. No BEATs fine-tuning. No CTAF training.

## 3. Alignment Validity

Train/test sample counts must match exactly between V37 NPZ and V55 NPZ.

| Machine | CTAF train | BEATs train | CTAF test | BEATs test | Aligned? |
|---------|-----------|-------------|----------|------------|----------|
| bearing    | 990 | 990 | 200 | 200 | YES |
| fan        | 990 | 990 | 200 | 200 | YES |
| gearbox    | 990 | 990 | 200 | 200 | YES |
| slider     | 990 | 990 | 200 | 200 | YES |
| toycar     | 990 | 990 | 200 | 200 | YES |
| toytrain   | 990 | 990 | 200 | 200 | YES |
| valve      | 990 | 990 | 200 | 200 | YES |

## 4. Ridge Regression Distillability

CTA-F latent_summary (192-dim) → BEATs embedding (768-dim), Ridge α=1.0, 5-fold CV.

| Machine | n_train | R² mean | R² std |
|---------|--------|---------|--------|
| bearing    | 990 | 0.3432 | 0.0148 |
| fan        | 990 | 0.5860 | 0.0045 |
| gearbox    | 990 | 0.2858 | 0.0196 |
| slider     | 990 | 0.4899 | 0.0144 |
| toycar     | 990 | 0.4847 | 0.0056 |
| toytrain   | 990 | 0.6392 | 0.0083 |
| valve      | 990 | 0.6759 | 0.0085 |

## 5. Residual Analysis

Per-sample L2 residual norm from Ridge fit on full train set.

| Machine | Residual mean | Residual std | z vs all machines |
|---------|--------------|-------------|-------------------|
| bearing    | 17.1490 | 3.1985 | 1.3494 |
| fan        | 13.4827 | 3.7642 | -0.7161 |
| gearbox    | 16.9247 | 5.1855 | 1.223 |
| slider     | 15.2445 | 2.9401 | 0.2764 |
| toycar     | 15.2716 | 2.9472 | 0.2917 |
| toytrain   | 12.9351 | 2.0601 | -1.0246 |
| valve      | 12.2692 | 2.1123 | -1.3998 |

## 6. Neighbor Preservation

k=5, cosine distance on L2-normalized train-normal vectors.

| Machine | Recall@5 | Jaccard@5 |
|---------|----------|-----------|
| bearing    | 0.0941 | 0.0560 |
| fan        | 0.1552 | 0.1027 |
| gearbox    | 0.0364 | 0.0220 |
| slider     | 0.2176 | 0.1398 |
| toycar     | 0.0113 | 0.0063 |
| toytrain   | 0.1075 | 0.0674 |
| valve      | 0.0255 | 0.0143 |

## 7. Score Relation

Spearman ρ and Pearson r between V37 CTA-F test scores and V55 BEATs test scores.

| Machine | Spearman ρ | Pearson r | n_test |
|---------|-----------|----------|--------|
| bearing    | 0.4291 | 0.5008 | 200 |
| fan        | 0.8197 | 0.9365 | 200 |
| gearbox    | 0.5237 | 0.5393 | 200 |
| slider     | 0.3776 | 0.4267 | 200 |
| toycar     | -0.1614 | -0.2015 | 200 |
| toytrain   | 0.2183 | 0.4128 | 200 |
| valve      | 0.3796 | 0.4578 | 200 |

## 8. Final Verdict

| Gate | Condition | Result |
|------|-----------|--------|
| 1 | ≥4 machines R²≥0.30 | 6/7 PASS |
| 2 | bearing/fan R² ≥ mean-0.10 (0.401) | bearing=0.343 fan=0.586 FAIL |
| 3 | mean recall@5 ≥ 0.10 | 0.0925 FAIL |
| 4 | ≥4 machines score ρ≥0.20 | 6/7 PASS |
| 5 | bearing/fan/toycar residual z≤1.0 | bearing=1.3494 fan=-0.7161 toycar=0.2917 FAIL |
| 6 | data aligned | PASS |

**V56A fail: do not train V56. BEATs has machine-level complementary signal, but not sample-level distillable structure.**

---

No training performed. No BEATs fine-tuning. No CTAF encoder modification. 
No score fusion. No per-machine selection.
