# CTAF V41 Transient Signal Preservation Diagnostic

## 1. Purpose

Diagnose where V39 Stage-0 input-level separation signal is lost through the V40 `ViewwiseUnsharedLateFusionAutoencoder` pipeline.

- Experiment: `transient_3view`
- Score mode: `latent_summary_knn`
- Normal-tail band: q0.95 -- q0.99

robust_d = (median(group) - median(normal_tail)) / (0.5*(IQR(group)+IQR(normal_tail)) + 1e-8)

## 2. Per-Machine Per-Stage Separation

### gearbox

n_missed=98  n_detected=2  n_normal_tail=4

| Stage | robust_d_missed | robust_d_detected | d_tail | d_missed | d_detected | Signal? |
|-------|-----------------|-------------------|--------|----------|------------|---------|
| input        | -0.7717 | 3.5406 | 282.2526 | 274.9437 | 869.6900 | weak |
| stem         | -0.6322 | 3.5249 | 237.1364 | 229.3305 | 739.3126 | preserved |
| enc2         | -0.2484 | 3.4160 | 188.5133 | 187.7637 | 554.4799 | preserved |
| fused_pre    | -0.2484 | 3.4160 | 188.5133 | 187.7637 | 554.4799 | LOST |
| latent_map   | -0.6162 | 3.4218 | 116.1270 | 112.9244 | 353.5332 | LOST |
| latent_summary | 1.8486 | 2.5418 | 10.4457 | 12.8097 | 46.5320 | preserved |

### slider

n_missed=99  n_detected=1  n_normal_tail=4

| Stage | robust_d_missed | robust_d_detected | d_tail | d_missed | d_detected | Signal? |
|-------|-----------------|-------------------|--------|----------|------------|---------|
| input        | -0.2250 | 25.1107 | 296.7146 | 300.9484 | 661.0227 | weak |
| stem         | -0.2186 | 20.4544 | 258.9496 | 256.6928 | 546.3363 | LOST |
| enc2         | 0.0027 | 23.3589 | 205.2616 | 210.9552 | 449.0483 | preserved |
| fused_pre    | 0.0027 | 23.3589 | 205.2616 | 210.9552 | 449.0483 | preserved |
| latent_map   | -0.2435 | 18.8804 | 121.7748 | 120.6917 | 259.8506 | LOST |
| latent_summary | 1.4425 | 31.4093 | 11.7553 | 13.8504 | 41.6037 | preserved |

### toycar

n_missed=100  n_detected=0  n_normal_tail=4

| Stage | robust_d_missed | robust_d_detected | d_tail | d_missed | d_detected | Signal? |
|-------|-----------------|-------------------|--------|----------|------------|---------|
| input        | 1.6746 | 0.0000 | 212.1940 | 303.5714 | 0.0000 | baseline |
| stem         | 1.4913 | 0.0000 | 187.7867 | 262.9142 | 0.0000 | decaying |
| enc2         | 1.7697 | 0.0000 | 141.0456 | 197.3381 | 0.0000 | preserved |
| fused_pre    | 1.7697 | 0.0000 | 141.0456 | 197.3381 | 0.0000 | preserved |
| latent_map   | 1.6349 | 0.0000 | 87.2089 | 122.7737 | 0.0000 | preserved |
| latent_summary | -0.2186 | 0.0000 | 8.5943 | 8.7295 | 0.0000 | LOST |

### toytrain

n_missed=92  n_detected=8  n_normal_tail=4

| Stage | robust_d_missed | robust_d_detected | d_tail | d_missed | d_detected | Signal? |
|-------|-----------------|-------------------|--------|----------|------------|---------|
| input        | 0.0185 | 1.7841 | 398.5059 | 406.9574 | 534.8875 | weak |
| stem         | -0.0819 | 1.6661 | 356.8675 | 355.0160 | 477.4987 | LOST |
| enc2         | 0.0176 | 1.9822 | 266.6919 | 271.9465 | 359.9948 | preserved |
| fused_pre    | 0.0176 | 1.9822 | 266.6919 | 271.9465 | 359.9948 | preserved |
| latent_map   | 0.1375 | 1.9859 | 160.7387 | 166.1976 | 220.0719 | preserved |
| latent_summary | 1.4296 | 2.1084 | 11.8915 | 16.9790 | 18.1291 | preserved |

### fan

n_missed=94  n_detected=6  n_normal_tail=4

| Stage | robust_d_missed | robust_d_detected | d_tail | d_missed | d_detected | Signal? |
|-------|-----------------|-------------------|--------|----------|------------|---------|
| input        | 2.3034 | 2.1251 | 193.1773 | 380.7953 | 274.0952 | baseline |
| stem         | 2.2791 | 2.2024 | 174.5142 | 362.2051 | 256.5880 | preserved |
| enc2         | 2.4788 | 2.3557 | 128.4179 | 257.4601 | 185.3359 | preserved |
| fused_pre    | 2.4788 | 2.3557 | 128.4179 | 257.4601 | 185.3359 | preserved |
| latent_map   | 2.7674 | 2.0827 | 79.3137 | 139.3131 | 112.4528 | preserved |
| latent_summary | 3.9147 | 1.3789 | 3.4565 | 10.8726 | 6.8925 | preserved |

## 3. Cross-Machine Stage Summary

Mean |robust_d_missed| across 5 bottleneck machines, per stage.

| Stage | Mean |rd_missed| | Min |rd_missed| | Max |rd_missed| | Signal retention |
|-------|-----------------|-----------------|-----------------|-------------------|
| input        | 0.9986 | 0.0185 | 2.3034 | strong |
| stem         | 0.9406 | 0.0819 | 2.2791 | strong |
| enc2         | 0.9034 | 0.0027 | 2.4788 | strong |
| fused_pre    | 0.9034 | 0.0027 | 2.4788 | strong |
| latent_map   | 1.0799 | 0.1375 | 2.7674 | strong |
| latent_summary | 1.7708 | 0.2186 | 3.9147 | strong |

## 4. Signal Drop-off Analysis

- **Primary signal loss stage**: `input -> stem` (drop = 0.0580)
- Input |robust_d|: 0.9986
- Stem |robust_d|: 0.9406
- Enc2 |robust_d|: 0.9034
- Latent map |robust_d|: 1.0799
- Latent summary |robust_d|: 1.7708

### Per-machine drop pattern

- **gearbox**: input=0.7717 -> summary=1.8486 (retention=239.5%)
- **slider**: input=0.2250 -> summary=1.4425 (retention=641.1%)
- **toycar**: input=1.6746 -> summary=0.2186 (retention=13.1%)
- **toytrain**: input=0.0185 -> summary=1.4296 (retention=7737.5%)
- **fan**: input=2.3034 -> summary=3.9147 (retention=170.0%)

## 5. Toytrain: Why Signal Survives

- Input signal: 0.0185
- Stem output: 0.0819
- Latent summary: 1.4296

## 6. Failure Machines: Where Signal Dies

- **gearbox**: input=0.7717 -> stem=0.6322 -> summary=1.8486. Primary loss at: `scoring`.
- **slider**: input=0.2250 -> stem=0.2186 -> summary=1.4425. Primary loss at: `enc2`.
- **toycar**: input=1.6746 -> stem=1.4913 -> summary=0.2186. Primary loss at: `latent_summary aggregation`.
- **fan**: input=2.3034 -> stem=2.2791 -> summary=3.9147. Primary loss at: `scoring`.

## 7. Next-Step Guidance

- **Recommendation**: Transient-specific per-view stem.
- The current per-view stem (enc1: 1->8, stride=2) is shared across all three views but is too aggressive for transient features.
- Consider a transient-specific stem with smaller stride or dilated temporal convolutions to preserve transient events.

---

Models were loaded from saved V40 checkpoints trained under the fixed V40 protocol. No retraining was performed. No frontend parameters were tuned. No per-machine selection was applied.
