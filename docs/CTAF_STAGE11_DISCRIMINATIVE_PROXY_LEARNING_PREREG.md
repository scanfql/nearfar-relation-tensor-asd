# CTAF Stage 11 / V57 — Discriminative Proxy Learning Preregistration

## 1. Motivation

CTAF V39–V56A has exhaustively tested unsupervised training objectives and frozen-representation scoring within the current pipeline:

| Stage | Training Objective | Result |
|-------|--------------------|--------|
| V16/V37 | AE reconstruction / teacher-student distillation | current best, plateaued |
| V48–V49 | reconstruction + hard-normal consistency loss | mechanism positive, method negative |
| V52–V53 | VICReg non-contrastive invariance SSL | 7/7 gates fail |
| V51 | frozen representation + OC-SVM learned boundary | q99 inflation, below kNN |
| V54–V55 | frozen BEATs external representation | hmean positive, pAUC/q99 fail |
| V56A | BEATs-to-CTAF distillation diagnosis | sample-level distillability fail |

Across all stages, **every unsupervised objective—whether AE reconstruction, consistency, or SSL—converges to the same tradeoff**: some machines improve, others degrade, and low-FPR tail separation never reliably exceeds V37 baseline.

**No CTAF experiment has ever used a supervised discriminative training objective.** This is the single largest unexplored dimension in the pipeline. Standard anomaly detection literature makes a clear distinction between unsupervised (AE/SSL) and discriminative (proxy classification) training paradigms, and the CTAF protocol has never crossed that boundary—even with a label-free-compatible proxy.

V57 therefore does **not** continue:

- unsupervised representation learning search (any variant)
- scoring-rule replacement on frozen representations
- frontend or pooling replacement
- external frozen representation evaluation
- hard-normal or augment-based consistency

---

## 2. Hypothesis

**Core hypothesis:**

> A discriminative proxy classification objective (AAM-softmax on machine-type labels) trained on full_5view normal samples can learn a latent_summary space where normal intra-class compactness naturally tightens the normal manifold and improves low-FPR tail separation, without requiring anomaly labels or changing the kNN scoring rule.

Translation into design intent:

The model is trained to classify **which machine type** each normal sample belongs to. The AAM-softmax (Additive Angular Margin) loss explicitly enforces intra-class compactness—normal samples from the same machine are pulled toward a tight class center. Since anomaly samples (at test time) are from the same machine type but with different acoustic characteristics, they are expected to fall further from the trained class center. The encoder's latent_summary is then scored via standard kNN on source train-normal memory—the same scoring protocol as V37—without using the classification head at inference.

This hypothesis reverses the logic of all previous CTAF experiments. Instead of asking "how do we suppress the normal tail?" or "how do we tighten the normal support?", it asks: **"can a discriminative training signal that is already available (machine type) provide structural regularization that unsupervised objectives cannot?"**

---

## 3. Why Machine-Type Labeling Does Not Violate Label-Free Protocol

The DCASE/LAES Task 2 protocol requires that anomaly labels are not used in training, model selection, or scoring design. Machine-type labels are not anomaly labels:

- Every audio file in the DCASE dataset has a known machine type (gearbox, fan, etc.)
- Machine type is a **domain identifier**, not an anomaly indicator
- Train-normal and test-anomaly samples share the same machine type
- The classification head is discarded at inference—it never sees test samples

This is analogous to using domain/section labels as conditioning variables, which is standard in one-class and unsupervised anomaly detection literature.

---

## 4. Candidate Method

**Primary candidate:** `full_5view + AAM-softmax machine-type proxy classification`

Design sketch (implementation not authorized in this step):

```
Input:         full_5view tensor [B, 5, F, T]
               ↓
Encoder:       viewwise_unshared_late_fusion (frozen architecture)
               ↓
Representation: global latent_summary (192-dim)
               ↓
Classifier:    linear layer 192 → num_machines (7)
               ↓
Loss:          AAM-softmax (s=30, m=0.2, scale and margin fixed, no sweep)
               ↓
Training data: source train-normal only (labeled by machine type)
               ↓
Anomaly label: not used
               ↓
Inference:     discard classifier head
               latent_summary → kNN (k=5) on source train-normal memory
```

**Why AAM-softmax over standard cross-entropy:**

- Standard CE encourages correct classification but does not explicitly minimize intra-class distances
- AAM-softmax adds an angular margin that forces same-class samples to be within a tight angular sector from the class weight vector
- The margin directly creates intra-class compactness, which is precisely the mechanism hypothesized to tighten the normal manifold
- Margin (m=0.2) and scale (s=30) are standard values from the face recognition literature where intra-class compactness is the primary goal

**Why 7-class classification over section or domain proxy:**

| Proxy label | Classes | Samples/class | Risk |
|-------------|---------|---------------|------|
| machine type | 7 | ~990 | none—largest stable grouping |
| section | ~70 | ~10–20 | overfitting, near sample-identity |
| machine × domain | 14 | ~495 | domain label leaks at test time |

Machine type is the only proxy with both sufficient per-class samples and no test-time information leak.

---

## 5. Two-Step Validation Protocol

### V57: No-Training Feasibility Check

Before any training, run a diagnostic on **frozen V37 full_5view latent_summary** (no encoder training). All features are already available in V37 score NPZ files.

**Required diagnostics:**

| # | Diagnostic | What it measures | Minimum report format |
|---|-----------|-----------------|----------------------|
| D1 | Machine-type linear classification accuracy | Whether frozen V37 features already contain linearly separable machine-type information | per-machine held-out accuracy, 5-fold CV |
| D2 | Machine-type AAM-softmax classification accuracy | Same as D1 but with angular margin—measures how much margin is available | same as D1, with AAM parameters fixed (s=30, m=0.2) |
| D3 | Per-class intra-class cosine spread | Mean pairwise cosine distance within each machine class—measures how tight each normal cluster already is | per-class mean ± std |
| D4 | Inter-class angular margin | Minimum angular distance between class weight vectors after training AAM-softmax—measures separability headroom | per-class min angular separation to nearest other class |
| D5 | Per-class nearest-class margin | For each class, the distance to the nearest other class's centroid—measures whether classes are well-isolated | per-class value |
| D6 | Intra-class compactness potential | Difference between current cosine spread and minimum reachable spread (all samples at centroid)—measures how much room there is for AAM-softmax to tighten | per-class: current_spread − 0 (normalized) |

**V57 pass conditions (all must hold):**

1. D1 linear classification accuracy ≥ 0.90 across all 7 machine types
2. D2 AAM-softmax accuracy ≥ 0.90 across all 7 machine types
3. D3 per-class intra-class cosine spread is non-zero (representation is not collapsed) and shows meaningful variation across machines
4. D4 inter-class angular margin shows at least 2 machines with margin ≥ 0.05 radians from their nearest neighbor
5. D6 at least 3/7 machines show intra-class spread > 0.10 (cosine distance units), indicating room for AAM-softmax to tighten the manifold
6. D1 and D2 accuracy values are consistent (no catastrophic degradation from AAM margin)

**Critical gate**: classification accuracy ≥ 0.90 is necessary but **not sufficient**. V57 must also demonstrate that intra-class compactness has room for improvement (D6), and that inter-class boundaries are well-defined (D4). If accuracy is 0.98 but intra-class spread is already near 0.01, AAM-softmax training adds nothing—the features are already maximally compact and further margin has no room to operate.

**No training of the CTAF encoder in V57.**

### V58: Single-Seed Pilot (only if V57 passes ALL gates above)

If V57 confirms both separability AND compactness potential, train full_5view encoder from scratch with AAM-softmax loss:

- seed = 13711 only
- epochs = 50, batch_size = 32
- no anomaly labels in training
- scoring = latent_summary_knn (k=5) on source train-normal memory
- no score fusion, no per-machine selection

---

## 6. Fixed Training Protocol Draft (for V58)

| Parameter | Value |
|-----------|-------|
| frontend | full_5view only |
| encoder | viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0) |
| representation | global latent_summary (192-dim) |
| classifier | linear 192 → 7 |
| loss | AAM-softmax, s=30, m=0.2 |
| training data | source train-normal only |
| proxy labels | machine type (7 classes) |
| anomaly labels | not used |
| epochs | 50, batch_size=32 |
| seed | 13711 |
| scoring after training | latent_summary_knn, k=5 |
| score fusion | prohibited |
| per-machine selection | prohibited |

---

## 7. Difference from Previous Stages

### vs V37 Teacher-Student

V37 is unsupervised teacher-student distillation (student predicts teacher's local latent). Stage 11 is supervised proxy classification (encoder predicts machine type). The training signal in Stage 11 comes from an external label that was always available but never used. The scoring protocol is identical (latent_summary + kNN).

### vs V52–V53 VICReg SSL

VICReg is unsupervised SSL (augmented-view invariance). Stage 11 is supervised proxy classification. VICReg failed because invariance SSL smooths normal and anomaly structure together. AAM-softmax with margin creates the opposite—sharp intra-class boundaries that may separate anomalies naturally.

### vs V49 Hard-Normal Consistency

V49 added a consistency loss on a heuristically-selected hard-normal subset. Stage 11 adds no auxiliary loss—the entire objective is AAM-softmax on all train-normal. The distinction between "hard" and "easy" normal is learned implicitly through the angular margin.

### vs V51 OC-SVM

V51 replaced kNN scoring with a learned boundary on frozen features. Stage 11 changes the training objective while keeping kNN scoring unchanged.

### vs V54–V55 BEATs

BEATs introduced external frozen representations. Stage 11 keeps CTAF's own encoder and introduces a new training signal (machine-type labels) that was always present in the dataset.

---

## 8. Risk: Necessary but Not Sufficient

AAM-softmax learns machine-type discrimination: the loss drives same-machine samples toward a compact class center and separates different-machine clusters. The ASD evaluation task requires **within-machine normal-vs-anomaly separation**—distinguishing normal gearbox samples from anomalous gearbox samples, not distinguishing gearbox from fan.

These two objectives are structurally different:

- **Proxy task**: between-class separation (gearbox vs fan)
- **ASD task**: within-class separation (normal gearbox vs anomalous gearbox)

High machine-type classification accuracy is **necessary but not sufficient** for low-FPR anomaly detection. It is possible for an encoder to achieve near-perfect machine-type classification while producing latent_summary features that are no better—or even worse—than V37 at normal-vs-anomaly separation. This could happen if:

1. The encoder over-specializes to inter-machine differences and loses within-machine anomaly-sensitive features
2. The AAM-softmax angular margin pushes anomalous samples (same machine, different acoustics) further from the class center—which helps—but also pushes normal samples so close together that the kNN tail loses resolution
3. The margin is too aggressive for certain machine types where intra-class normal variation is naturally high

**Therefore V58 is only justified if V57 shows not only separability (D1, D2) but also meaningful intra-class compactness potential (D6) with room for the margin to operate without collapsing within-class structure.**

If V57 accuracy is high but intra-class spread is near zero, the V37 representation is already at the compactness limit—AAM-softmax training cannot improve it further and may only add overfitting risk.

---

## 9. Success Gates (for future V58 single-seed pilot)

If V57 passes and V58 proceeds:

| # | Gate | Threshold |
|---|------|-----------|
| 1 | mean hmean >= V37 baseline − 0.003 | must hold |
| 2 | mean pAUC >= V37 baseline | must hold |
| 3 | at least 2 bottleneck machines improve TPR@99n | must hold |
| 4 | bearing / valve do not collapse | must hold |
| 5 | toycar does not clearly degrade | must hold |
| 6 | normal q99 does not clearly inflate | must hold |
| 7 | gains are not carried by only one machine | must hold |

Reference baseline: V37 full_5view latent_summary_knn (seed 13711: hmean=0.5631, pAUC=0.5505).

---

## 10. Failure Criteria

Stage 11 first implementation is recorded as negative if:

- hmean is clearly below V37 baseline
- pAUC is below V37 baseline
- normal q99 increases
- bearing / valve collapse
- toycar degrades
- only gearbox / slider improve while main machines worsen
- AAM-softmax margin (m) requires sweep to be useful
- classification accuracy on train-normal is near chance level (encoder failed to learn discriminative features)

---

## 11. Forbidden Actions

The following are **strictly prohibited** under this preregistration:

- Do NOT open V58 during preregistration. V58 is authorized only after this preregistration is approved AND V57 feasibility check passes.
- Do NOT implement training code in this step
- Do NOT train any model
- Do NOT change full_5view frontend
- Do NOT change global latent_summary pooling
- Do NOT use section or domain labels as proxy
- Do NOT sweep AAM-softmax scale or margin
- Do NOT test alternative classification losses (CE, center loss, triplet)
- Do NOT do score fusion
- Do NOT do per-machine selection
- Do NOT use anomaly labels in training
- Do NOT auto-propose V59

---

## 12. Human Decisions Required Before Implementation

Before V57/V58 implementation begins:

| # | Decision | Recommendation |
|---|----------|---------------|
| 1 | Proxy label | machine type (7 classes) |
| 2 | AAM-softmax scale (s) | 30 (standard value from literature) |
| 3 | AAM-softmax margin (m) | 0.2 (standard value) |
| 4 | Encoder architecture | reuse viewwise_unshared_late_fusion |
| 5 | V57 feasibility gate threshold | classification accuracy ≥ 0.90 on frozen V37 features |
| 6 | Whether V57 should also report intra-class cosine spread vs V37 AE features | recommended |

---

## 13. Final Sentence

Stage 11 is a new framework-level hypothesis: discriminative proxy supervision, not unsupervised representation learning. It must not be treated as a continuation of V39–V56A AE / SSL / consistency / scoring-rule search. No code implementation or training is authorized until this preregistration is reviewed and approved.
