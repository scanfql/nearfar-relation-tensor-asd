# CTAF V58 Within-Machine Pseudo-Anomaly Audit

## 1. Stage 12 Prereg Summary

V58 trains full_5view encoder with binary BCE classification: clean vs
same-machine local TF patch replacement pseudo-anomaly. Classifier head
discarded at inference. Scoring: latent_summary_knn (k=5).

## 2. Fixed Protocol

- patch_time_frac = 0.05
- patch_freq_frac = 0.05
- n_patches = 1, all 5 views
- donor = same-machine train-normal
- binary head = 192 → 64 → 1 (discarded at inference)
- loss = BCE, lr=0.001, epochs=50, bs=32
- seed = 13711
- scoring = latent_summary_knn, k=5

## 3. Overall Metrics

V37 baseline: hmean=0.5631, pAUC=0.5505
V58: hmean=0.5026, pAUC=0.5083
Δ hmean = -0.0605, Δ pAUC = -0.0421

## 4. Per-Machine

| Machine | hmean | pAUC |
|---------|-------|------|
| bearing    | 0.5149 | 0.5474 |
| fan        | 0.5053 | 0.5163 |
| gearbox    | 0.5053 | 0.5163 |
| slider     | 0.4984 | 0.4953 |
| toycar     | 0.5000 | 0.5000 |
| toytrain   | 0.4939 | 0.4821 |
| valve      | 0.5003 | 0.5011 |

## 5. Tail Table

| Machine | q95 | q99 | TPR@99n |
|---------|-----|-----|---------|
| bearing    | 0.1600 | 0.1736 | 0.0900 |
| fan        | 0.2946 | 0.3062 | 0.0400 |
| gearbox    | 0.1722 | 0.1846 | 0.0600 |
| slider     | 0.2167 | 0.2377 | 0.0100 |
| toycar     | 0.1047 | 0.1072 | 0.0400 |
| toytrain   | 0.2243 | 0.2648 | 0.0000 |
| valve      | 0.1116 | 0.1175 | 0.0400 |

## 6. Pass/Fail Gates

| Gate | Threshold | Actual | Verdict |
|------|-----------|--------|---------|
| 1 | hmean >= V37 - 0.003 | 0.5026 vs 0.5601 | FAIL |
| 2 | pAUC >= V37 | 0.5083 vs 0.5505 | FAIL |
| 3 | ≥2 bottleneck TPR@99n improve | 1/5 | FAIL |
| 4 | bearing/valve not collapse | bearing=OK valve=OK | PASS |
| 5 | toycar not degraded | OK | PASS |

**V58 single-seed FAIL.**

---

No protocol changes. No sweep. No per-machine selection. No V59.
