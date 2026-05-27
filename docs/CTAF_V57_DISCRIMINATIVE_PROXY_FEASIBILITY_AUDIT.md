# CTAF V57 Discriminative Proxy Feasibility Audit

## 1. Stage 11 Prereg Summary

V57 tests whether frozen V37 full_5view latent_summary (192-dim) contains
sufficient machine-type discriminative structure for AAM-softmax to have
room to improve intra-class normal manifold compactness.

No CTAF encoder training. No anomaly labels. No V58.

## 2. Protocol

- Feature: V37 frozen latent_summary, 192-dim, 7 machines × 990 train-normal
- Label: machine type (7 classes)
- D1: Linear classification, 5-fold CV, StandardScaler
- D2: AAM-softmax, s=30, m=0.2, 30 epochs, lr=0.01, 5-fold CV
- D3-D6: Geometry diagnostics on full train set using AAM weight vectors

## 3. D1 — Linear Classification Accuracy

5-fold CV macro accuracy: **0.9999**

| Machine | Per-class accuracy |
|---------|-------------------|
| bearing    | 1.0000 |
| fan        | 1.0000 |
| gearbox    | 1.0000 |
| slider     | 1.0000 |
| toycar     | 1.0000 |
| toytrain   | 1.0000 |
| valve      | 0.9990 |

Confusion matrix (rows=true, cols=pred):
```
         bear   fan  gear  slid  toyc  toyt  valv
bear    990     0     0     0     0     0     0
fan       0   990     0     0     0     0     0
gear      0     0   990     0     0     0     0
slid      0     0     0   990     0     0     0
toyc      0     0     0     0   990     0     0
toyt      0     0     0     0     0   990     0
valv      0     0     0     0     1     0   989
```

## 4. D2 — AAM-Softmax Classification Accuracy

5-fold CV macro accuracy: **0.9364**

| Machine | Per-class accuracy |
|---------|-------------------|
| bearing    | 1.0000 |
| fan        | 0.7657 |
| gearbox    | 1.0000 |
| slider     | 0.8576 |
| toycar     | 1.0000 |
| toytrain   | 1.0000 |
| valve      | 0.9313 |

Confusion matrix:
```
         bear   fan  gear  slid  toyc  toyt  valv
bear    990     0     0     0     0     0     0
fan     202   758     7    21     1     1     0
gear      0     0   990     0     0     0     0
slid     68    56     0   849     0     6    11
toyc      0     0     0     0   990     0     0
toyt      0     0     0     0     0   990     0
valv      4    10    36     5     0    13   922
```

## 5. D3 — Intra-Class Cosine Spread

| Machine | Mean spread | Std | Q95 | Q99 |
|---------|-------------|-----|-----|-----|
| bearing    | 0.0101 | 0.0065 | 0.0243 | 0.0317 |
| fan        | 0.0121 | 0.0059 | 0.0240 | 0.0284 |
| gearbox    | 0.0050 | 0.0030 | 0.0111 | 0.0183 |
| slider     | 0.0202 | 0.0109 | 0.0450 | 0.0598 |
| toycar     | 0.0026 | 0.0007 | 0.0039 | 0.0049 |
| toytrain   | 0.0100 | 0.0188 | 0.0209 | 0.1290 |
| valve      | 0.0531 | 0.0242 | 0.1035 | 0.1252 |

## 6. D4 — Inter-Class Angular Margin (AAM weights)

| Machine | Nearest class | Min angle (rad) | Min angle (deg) |
|---------|---------------|-----------------|-----------------|
| bearing    | toycar     | 1.4285 | 81.8 |
| fan        | bearing    | 1.5420 | 88.3 |
| gearbox    | bearing    | 1.5242 | 87.3 |
| slider     | toytrain   | 1.4496 | 83.1 |
| toycar     | bearing    | 1.4285 | 81.8 |
| toytrain   | slider     | 1.4496 | 83.1 |
| valve      | bearing    | 1.4749 | 84.5 |

## 7. D5 — Nearest-Class Centroid Margin

| Machine | Nearest class | Cosine distance | Angle (rad) |
|---------|---------------|-----------------|-------------|
| bearing    | fan        | 0.0147 | 0.1717 |
| fan        | bearing    | 0.0147 | 0.1717 |
| gearbox    | fan        | 0.0178 | 0.1892 |
| slider     | fan        | 0.0206 | 0.2032 |
| toycar     | bearing    | 0.0902 | 0.4281 |
| toytrain   | fan        | 0.0713 | 0.3798 |
| valve      | slider     | 0.0259 | 0.2279 |

## 8. D6 — Compactness Potential

| Machine | Potential (mean spread) |
|---------|------------------------|
| bearing    | 0.0101 |
| fan        | 0.0121 |
| gearbox    | 0.0050 |
| slider     | 0.0202 |
| toycar     | 0.0026 |
| toytrain   | 0.0100 |
| valve      | 0.0531 |

## 9. Pass/Fail Gate Table

| Gate | Condition | Result |
|------|-----------|--------|
| 1 | D1 linear macro acc ≥ 0.90 | 0.9999 PASS |
| 2 | D2 AAM macro acc ≥ 0.90 | 0.9364 PASS |
| 3 | No collapsed classes (spread > 0) | PASS |
| 4 | ≥2 machines with angle margin ≥ 0.05 rad | 7/7 PASS |
| 5 | ≥3 machines with compactness potential > 0.10 | 0/7 FAIL |
| 6 | D1/D2 accuracy consistent (Δ < 0.10) | Δ=0.0635 PASS |

**V57 fail: do not train V58. Frozen V37 features have insufficient room for AAM-softmax to improve intra-class compactness.**

---

No CTAF encoder training. No anomaly labels. No checkpoint modification. 
AAM parameters (s=30, m=0.2) fixed. No sweep. No per-machine selection.
