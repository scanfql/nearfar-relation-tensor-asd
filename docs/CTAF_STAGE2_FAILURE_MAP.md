# CTAF Stage-2 Failure Map

Generated: 2026-05-15
Role: Audit document, not a version plan.
Constraint: No V35 proposal. No failure packaging. Only cause-and-constraint mapping.

---

## 0. Stage-2 Entry Point: V16 Seed-Panel Baseline

### What V16 Is

```
full_5view tensor frontend
+ viewwise_unshared_late_fusion Conv2D AE
  (stem_channels=8, per_view_channels=16, latent_channels=64)
+ latent summary vectors (mean/std/max pooled from encoder spatial map)
+ train-normal latent kNN memory score, k=5
+ source-only normal training, 50 epochs, Adam, seed 13711
```

### Seed-Panel Baseline (5 seeds: 13711, 13721, 13731, 13741, 13751)

| metric | mean | std | min | max |
|---|---|---|---|---|
| hmean | 0.549547 | 0.004720 | 0.542080 | 0.553533 |
| pAUC | 0.542331 | 0.005097 | 0.534211 | 0.547820 |
| KS | 0.294880 | 0.024697 | 0.268600 | 0.328600 |

### What Is Confirmed Before Stage-2 Begins

- full_5view beats matched logmel_only on hmean in 5/5 seeds.
- full_5view beats matched logmel_only on both hmean and pAUC in 4/5 seeds.
- The CTAF direction is real, not a single-seed accident.
- **But:** The original frozen single-seed point (hmean=0.564, pAUC=0.553, seed 13711) is not reproducible on rerun. The seed-panel mean is the only valid operational baseline.

### Key Machines: Who Carries V16 and Who Drags It

| role | machines | evidence |
|---|---|---|
| strong positives | bearing, fan, valve | hmean > 0.56, pAUC > 0.55 |
| marginal | gearbox, toytrain | hmean ~0.53-0.54, pAUC ~0.51-0.52 |
| weak | slider, toycar | high variance, unstable across seeds |

### What Remains Unsolved Entering Stage-2

V16 proves the tensor frontend + unshared late fusion is a real signal. But:

- The absolute performance is modest (hmean ~0.55).
- The per-machine spread is wide.
- At low FPR, several machines are barely above random.
- There is no evidence that this architecture will scale further without structural change.

---

## 1. V30: Viewwise Unshared Temporal Pyramid

### a. What failure mode does it try to solve?

**Failure mode:** The V16 encoder treats each time frame independently at the deepest spatial bottleneck. Temporal structure across frames is only captured implicitly by 2D convolutions over (F, T). Multi-scale temporal dynamics—short glitches, medium rhythm changes, long operating-point drift—are not explicitly modeled.

**Hypothesis:** Inserting a three-branch temporal pyramid (short/medium/long) after each per-view stem, before late fusion, will create a multi-scale temporal representation that separates transient anomalies from stable normal patterns better.

### b. What did it actually improve?

| metric | V16 seed-panel mean | V30 3-seed mean | delta |
|---|---|---|---|
| hmean | 0.549547 | **0.555213** | **+0.005666** |
| pAUC | 0.542331 | 0.541303 | -0.001028 |
| KS | 0.294880 | 0.295267 | +0.000387 |

- hmean improves reproducibly across all 3/3 seeds.
- Every seed beats its matched logmel_only on both hmean and pAUC.
- The temporal pyramid is not noise—it is a real, directionally consistent signal.

### c. Where did it fail?

- **pAUC does not clear the V16 seed-panel mean.** It is short by 0.001028. Under the freeze rule, this is a failure—not a near-miss to be waved through.
- **KS is effectively tied with V16.** The temporal pyramid improves mid-rank discrimination but does not improve domain-shift alignment.
- **The per-machine pattern is unbalanced:**
  - bearing improves on both axes (+0.024 pAUC, +0.023 hmean in single-seed).
  - toycar and toytrain lose pAUC (-0.055 and -0.019 in single-seed).
  - gearbox loses pAUC (-0.012).
- The temporal pyramid helps machines that already work well. It does not rescue the bottleneck machines.

### d. What constraint does it leave for next steps?

> **Temporal multi-scale modeling is a valid signal direction, but it does not solve low-FPR. Any future temporal architecture must also explicitly target the anomaly tail separation problem—not just aggregate ranking quality.**

---

## 2. V31: Viewwise Unshared Residual Temporal Pyramid

### a. What failure mode does it try to solve?

**Failure mode:** V30 added a temporal pyramid as a replacement path, discarding the original V16 stem output. This is a hard architectural switch—the model loses the proven V16 signal path entirely.

**Hypothesis:** Keep the V16 stem output as the main path. Add the temporal pyramid as a residual correction with a small learnable scalar alpha (init=0.1). This should preserve V30's hmean gain while recovering V16-level pAUC and possibly improving domain geometry.

```
h_out = h_v16_stem + alpha * temporal_pyramid(h_v16_stem)
```

### b. What did it actually improve?

| metric | V16 seed-panel mean | V31 seed 13711 | delta |
|---|---|---|---|
| hmean | 0.549547 | **0.555806** | **+0.006259** |
| pAUC | 0.542331 | 0.541654 | -0.000677 |
| KS | 0.294880 | **0.2714** | **-0.023480** |

- **KS improves clearly.** This is the best domain-shift alignment of any V16-family variant in Stage-2.
- hmean still beats the seed-panel mean.
- The residual design does what it was supposed to do on the geometry axis: source-target normal score distributions become better aligned.

### c. Where did it fail?

- **pAUC is short by 0.000677.** Extremely close to the gate, but the rule is the rule.
- **Versus its direct sibling V30 (same seed 13711):**

| metric | V30 seed 13711 | V31 seed 13711 | delta |
|---|---|---|---|
| hmean | 0.562034 | 0.555806 | -0.006228 |
| pAUC | 0.548045 | 0.541654 | -0.006391 |
| KS | 0.2943 | 0.2714 | -0.0229 |

The residual correction trades raw discrimination for better geometry. The trade is real: V31 is more stable but less discriminative than V30.

- **Per-machine tail analysis reveals the core failure:**
  - slider V31: normal q99 = 0.1663, anomaly median = 0.1152, TPR@99n = **0.00**
  - gearbox V31: normal q99 = 0.1280, anomaly median = 0.1010, TPR@99n = 0.04
  - toycar V31: normal q99 = 0.0914, anomaly median = 0.0760, TPR@99n = 0.03

  Better KS means normals are better aligned. But anomalies are still not crossing the normal extreme-tail boundary.

### d. What constraint does it leave for next steps?

> **Score geometry improvement alone is not enough. Better KS does not guarantee better pAUC. The residual correction proves that domain alignment and anomaly tail separation are different problems. Any future architecture must address tail separation directly—it cannot be achieved as a side effect of better normal alignment.**

---

## 3. V32: Local Tail kNN (score_mode = local_tail_knn_top5)

### a. What failure mode does it try to solve?

**Failure mode:** V16/V30/V31 all score each file with a single latent summary vector (mean/std/max pooled over the entire encoder spatial map). This global pooling may average away local anomaly evidence. If an anomaly manifests as a spatially localized pattern in the latent map, global pooling will dilute it.

**Hypothesis:** Tokenize the encoder latent map at every spatial position. Build a train-normal local memory bank. File score = mean of the top 5% most anomalous token distances. This should capture local anomaly evidence that global pooling loses.

### b. What did it actually improve?

```
full_5view / local_tail_knn_top5:
hmean = 0.505107
pAUC  = 0.503383
KS    = 0.2771
```

**Nothing.** The score collapses relative to latent_summary_knn.

### c. Where did it fail?

- **Does not beat matched logmel_only.** Matched logmel_only / local_tail_knn_top5 has hmean=0.543, pAUC=0.524—both higher than full_5view under the same score rule.
- **The collapse is not subtle.** hmean drops from 0.562 (latent_summary_knn) to 0.505. pAUC drops from 0.549 to 0.503.
- **Only gearbox and toytrain show marginal relief.** Bearing, fan, slider, toycar, and valve all lose substantial low-FPR quality.
- **The bottleneck-machine tail table exposes the mechanism:**

| machine | score_mode | normal q99 | anomaly median | TPR@99n |
|---|---|---|---|---|
| gearbox | latent_summary_knn | 0.1351 | 0.0938 | 0.06 |
| gearbox | local_tail_knn_top5 | **0.4316** | 0.4142 | **0.01** |
| slider | latent_summary_knn | 0.1503 | 0.1099 | 0.03 |
| slider | local_tail_knn_top5 | **0.4759** | 0.4416 | **0.01** |
| toycar | latent_summary_knn | 0.0796 | 0.0662 | 0.04 |
| toycar | local_tail_knn_top5 | **0.3946** | 0.3852 | **0.01** |

**The local-tail score inflates the normal q99 threshold by 3-5x, and anomaly scores rise with it but never break away.** Local latent tokens are highly sensitive to normal variability. The top-5% rule picks up that normal noise, not anomaly tails.

### d. What constraint does it leave for next steps?

> **The current encoder latent map does not contain directly exploitable local tail signal under a fixed top-5% rule. Score-side postprocessing that targets the tail without changing the representation cannot work—the local token distribution is dominated by normal variability, not anomaly structure. Tail separation must be created during representation learning, not recovered after the fact.**

---

## 4. V33: Tail Compact Training Loss

### a. What failure mode does it try to solve?

**Failure mode:** V32 proved that score-side tail postprocessing fails because the latent representation itself does not separate normal tails from anomaly tails. The next logical move is to move tail control into training.

**Hypothesis:** Add a tail-compactness regularizer during training:

```
loss = MSE_reconstruction + lambda_tail * tail_compactness_loss
```

where tail_compactness_loss = mean distance of the top 10% train-normal latent summary vectors from batch center. This should compress the normal tail during representation learning, creating headroom for anomalies at test time.

### b. What did it actually improve?

```
full_5view / latent_summary_knn:
hmean = 0.531218
pAUC  = 0.518195
KS    = 0.2800
```

**Normal q99 is reduced on every bottleneck machine:**

| machine | V16 normal q99 | V33 normal q99 | change |
|---|---|---|---|
| gearbox | 0.1245 | 0.1068 | -14% |
| slider | 0.1319 | 0.0985 | -25% |
| toycar | 0.0757 | 0.0491 | -35% |
| toytrain | 0.1122 | 0.0763 | -32% |
| fan | 0.2367 | 0.1324 | -44% |

The tail-compactness loss does exactly what it is designed to do on the normal side.

### c. Where did it fail?

- **Does not beat matched logmel_only.** pAUC is lower than matched logmel_only.
- **Does not beat the V16 seed-panel mean gate.** hmean=0.531 < 0.550. pAUC=0.518 < 0.542.
- **The mechanism of failure is precise:**

| machine | V16 TPR@99n | V33 TPR@99n |
|---|---|---|
| gearbox | 0.06 | **0.01** |
| slider | 0.09 | **0.01** |
| toycar | 0.10 | **0.01** |
| toytrain | 0.05 | **0.00** |
| fan | 0.13 | 0.10 |

**The compactness loss compresses the normal tail AND the anomaly tail together.** Anomaly median and q75 both move downward with the normal tail. The relative separation does not improve because the loss is:

```
tail-compressing, but not tail-separating
```

It pulls everything toward the center. Normals get tighter. Anomalies also get tighter. The gap does not widen.

### d. What constraint does it leave for next steps?

> **A training-side regularizer that only compresses the normal tail will also compress the anomaly tail unless there is an explicit separation mechanism. Tail compactness without separation is not useful for low-FPR. Any future training objective must create a force that pushes anomalies away from the normal tail, not just a force that pulls normals toward the center.**

---

## 5. V34: Masked Cross-View Prediction

### a. What failure mode does it try to solve?

**Failure mode:** V16 and all subsequent variants train with standard autoencoder reconstruction: input = target = full tensor. The model learns identity-style mapping. This may produce latent representations that are good at copying but bad at detecting structural anomalies that preserve per-view statistics while breaking cross-view consistency.

**Hypothesis:** Switch to a self-supervised task: randomly mask one view, predict all views, compute loss only on the masked view. The model must learn stable cross-view relations of normal machine sound. Anomalies that disrupt these relations should produce larger prediction errors → larger latent distances.

### b. What did it actually improve?

```
full_5view / latent_summary_knn:
hmean = 0.539228
pAUC  = 0.531955
KS    = 0.2457
```

- Beats matched logmel_only: hmean delta = +0.015, pAUC delta = +0.013.
- **KS improves to 0.2457**, the best in Stage-2. Cross-view prediction produces the most domain-invariant latent representation tested so far.

### c. Where did it fail?

- **Does not pass the seed-panel mean gate.** hmean=0.539 < 0.550. pAUC=0.532 < 0.542.
- **Does not achieve candidate status** (requires hmean > 0.564, pAUC >= 0.553).
- **Bottleneck-machine TPR@99n gets worse on every audited machine:**

| machine | V16 TPR@99n | V34 TPR@99n |
|---|---|---|
| gearbox | 0.06 | **0.03** |
| slider | 0.09 | **0.04** |
| toycar | 0.10 | **0.01** |
| toytrain | 0.05 | **0.02** |
| fan | 0.13 | **0.00** |

- **The normal tail also rises.** V34 normal q99 is higher than V16 on every bottleneck machine. The cross-view SSL objective makes the latent space more domain-invariant but also inflates the normal extreme tail. As with V32, normal_q99 rises faster than anomaly occupancy beyond it.

### d. What constraint does it leave for next steps?

> **Cross-view prediction is a valid self-supervised direction for domain invariance (best KS in Stage-2), but it does not improve low-FPR tail separation. The tradeoff pattern is now repeated across three different intervention types (score-side, training-loss, SSL task): domain alignment and tail separation are independent problems. Improving one does not imply improving the other. Any future method that claims to solve low-FPR must demonstrate tail separation directly, not infer it from KS or hmean.**

---

## 6. Stage-2 Failure Summary

### What Was Tested

| version | intervention type | target failure mode | hmean vs V16 panel | pAUC vs V16 panel | KS vs V16 panel | gate result |
|---|---|---|---|---|---|---|
| V30 | architecture (temporal pyramid) | no explicit multi-scale time | +0.006 | -0.001 | tie | FAIL (pAUC) |
| V31 | architecture (residual pyramid) | V30 replaces V16 path entirely | +0.006 | -0.001 | **-0.023** | FAIL (pAUC) |
| V32 | score postprocessing (local tail kNN) | global pooling dilutes local anomalies | -0.044 | -0.039 | -0.018 | FAIL (all) |
| V33 | training loss (tail compactness) | normal tail too wide | -0.018 | -0.024 | -0.015 | FAIL (all) |
| V34 | training task (masked cross-view) | identity AE ignores cross-view structure | -0.010 | -0.010 | **-0.049** | FAIL (hmean, pAUC) |

### What Survived

1. **Temporal multi-scale modeling is a real signal.** V30 improves hmean reproducibly across seeds. The idea is not dead—it just does not solve pAUC.

2. **Residual correction improves domain geometry.** V31 achieves the best KS in the V16 family. Domain alignment can be engineered without losing all discrimination.

3. **Cross-view SSL improves domain invariance.** V34 achieves the best KS overall. Self-supervised pretext tasks are a valid direction for drift reduction.

### What Is Dead (Do Not Revisit)

1. **Score-side local-tail postprocessing.** V32 proves the current latent map has no local tail signal. Do not tune top-K percentiles, do not try local-window variants, do not fuse local + global scores.

2. **Training-side tail compression without separation.** V33 proves that pulling normals inward pulls anomalies inward equally. Do not try different lambda values, different top fractions, or different distance metrics for the same compactness idea.

3. **Masked cross-view prediction for the current architecture.** V34 gives the best KS but loses too much hmean and pAUC. Do not try different mask probabilities or mask strategies on this AE backbone.

---

## 7. The Single Unsolved Bottleneck

After five structural interventions spanning architecture, scoring, training loss, and self-supervised task design, exactly one bottleneck remains unsolved:

```
LOW-FPR EXTREME-TAIL SEPARATION
```

### Definition

A CTAF method solves low-FPR extreme-tail separation if and only if, on the bottleneck machines (gearbox, slider, toycar, toytrain, fan):

```
anomaly scores occupy the region beyond normal_q99
at a rate meaningfully higher than the V16 seed-panel baseline
```

Quantitatively: TPR@99n must improve on at least 2-3 bottleneck machines without collapsing on the strong machines (bearing, valve).

### Why Every Stage-2 Attempt Failed at This

| version | why it failed at tail separation |
|---|---|
| V30 | improves mid-rank, does not touch the extreme tail |
| V31 | aligns normals better, anomalies stay below normal_q99 |
| V32 | inflates normal_q99 faster than anomaly scores rise |
| V33 | compresses normal and anomaly tails equally |
| V34 | inflates normal_q99 via cross-view sensitivity, anomalies do not break away |

### What Is Not the Bottleneck

The following are NOT the bottleneck and should not be targeted by future work:

- **More representation power.** V12 proved wider encoders do not help.
- **Better score normalization.** V8/V9/V11 proved score whitening/robust-z/block-scaling do not help beyond V7.
- **Denoising corruption.** V6/V10 proved tf_mask hurts the latent-summary mainline.
- **More training data.** V25 proved target-domain normal data improves KS but hurts discrimination.
- **Different reconstruction targets.** V26 proved logmel-only reconstruction is weaker than all-view reconstruction.
- **More views.** V29 proved multi-resolution views do not beat V16.

### The Only Valid Direction

The Stage-2 evidence forces a single constraint on any future V35+:

> **A new method is only worth testing if its design hypothesis explicitly addresses why it will push anomaly scores beyond normal_q99 on the bottleneck machines—gear box, slider, toycar, toytrain, fan—without inflating normal_q99 at the same rate.**

Corollary: **Any method that can only demonstrate hmean improvement or KS improvement, without tail-separation evidence, is repeating a Stage-2 failure pattern and should not be pursued.**

---

## Appendix: Bottleneck Machine State (V16 baseline)

For reference in any future tail-separation work:

| machine | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---|---|---|---|---|---|
| gearbox | 0.1174 | 0.1245 | 0.0959 | 0.1057 | 0.09 | 0.06 |
| slider | 0.1239 | 0.1319 | 0.1084 | 0.1155 | 0.15 | 0.09 |
| toycar | 0.0711 | 0.0757 | 0.0646 | 0.0685 | 0.17 | 0.10 |
| toytrain | 0.0906 | 0.1122 | 0.0694 | 0.0762 | 0.09 | 0.05 |
| fan | 0.2271 | 0.2367 | 0.1704 | 0.2253 | 0.23 | 0.13 |

These are the numbers any future method must move.
