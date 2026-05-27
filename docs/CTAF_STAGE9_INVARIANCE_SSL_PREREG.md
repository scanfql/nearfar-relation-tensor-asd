# CTAF Stage 9 / V52 — Non-Contrastive Invariance SSL Preregistration

## 1. Motivation

CTAF V39–V51 has already converged on the following evidence:

| Stage | Scope | Evidence | Verdict |
|-------|-------|----------|---------|
| Stage 5 | mechanism frontends | 3× 3-view frontends cannot replace full_5view | negative |
| Stage 5 | segment scoring | helps weak frontends, degrades full_5view | non-transferable |
| Stage 6 | hard-normal training | compresses q99, unstable TPR@99n | mechanism positive, method negative |
| Stage 7 | patch pooling | 3/5 bottleneck gain but bearing/valve collapse | negative |
| Stage 8 | OC-SVM learned boundary | q99 inflation, hmean/pAUC below kNN | negative |

**V51 is the decisive evidence.** After testing the last remaining frozen-representation scoring paradigm (learned one-class boundary), the conclusion is:

> **kNN on V37 full_5view latent_summary is near-optimal scoring for this representation space. Further low-FPR improvement requires a new representation learning or training framework, not another scoring rule.**

V52 therefore does **not** continue:

- frontend replacement
- pooling replacement
- segment / patch scoring
- hard-normal consistency search
- OC-SVM / density / kNN scoring variants
- any frozen latent_summary scorer

---

## 2. Hypothesis

**Core hypothesis:**

> A non-contrastive invariance-based SSL objective trained on full_5view normal samples can learn a latent_summary space with a more stable normal support manifold and better low-FPR tail separation than AE reconstruction can provide.

Translation into design intent:

The model should **not** be trained to distinguish different normal samples from each other (as SimCLR does). Instead, it should learn a representation that is **stable under normal perturbations** — the same normal sample, subjected to two different small acoustic augmentations, should map to nearly the same latent summary vector. This encourages the normal manifold to be compact and smooth, which naturally tightens the normal score tail and makes it easier for kNN to detect off-manifold anomalies at low FPR.

The target outcome: *normal support tighter, not normal instances pushed apart.*

---

## 3. Why Not SimCLR-Style Contrastive Learning

SimCLR-style instance discrimination uses other normal samples as negatives:

- for each anchor normal sample, all other normal samples in the batch serve as negatives
- the loss pushes the anchor representation away from all negative representations
- this can **widen the normal manifold** rather than tighten it
- for one-class anomaly detection, a wider normal manifold may raise normal q99
- higher normal q99 directly degrades TPR@99n

Therefore the first Stage 9 implementation explicitly avoids negative-pair contrastive learning.

Non-contrastive alternatives (VICReg, BYOL, SimSiam) do not use explicit negatives. They learn invariance via architectural or loss-design mechanisms that avoid pushing normal samples apart.

---

## 4. Candidate Method

**Primary candidate:** `full_5view + VICReg-style non-contrastive SSL`

Design sketch (implementation not authorized in this step):

```
Input:   full_5view tensor [B, 5, F, T]
         ↓
         2 augmented views per sample (v1, v2)
         ↓
Encoder: viewwise_unshared_late_fusion (shared weights)
         ↓
Representation: global latent_summary (channel-wise mean + std + max, 192-dim)
         ↓
Projector: small MLP (e.g., 192 → 256 → 256)
         ↓
VICReg loss on projector output:
  L = λ_inv * invariance_loss(v1_proj, v2_proj)
    + λ_var  * variance_loss   (across batch dim)
    + λ_cov  * covariance_loss (across feature dim)

No negative pairs.
No anomaly labels.
No per-machine tuning.
```

**VICReg loss components:**

- **Invariance loss**: MSE between projected representations of two augmented views of the same sample. Encourages the representation to be stable under augmentation.
- **Variance loss**: Hinge loss that prevents each feature dimension from collapsing to a constant. Maintains representation diversity.
- **Covariance loss**: Penalizes off-diagonal terms of the feature covariance matrix. Decorrelates feature dimensions to prevent informational collapse.

After training, scoring uses the **encoder's latent_summary** (not the projector output) with standard latent_summary_knn (k=5).

---

## 5. Fixed Training Protocol Draft

If implemented after approval:

| Parameter | Value |
|-----------|-------|
| data | source train-normal only |
| anomaly labels | not used in training |
| test normal | not used in training |
| frontend | full_5view only |
| pooling | global latent_summary only |
| encoder | viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0) |
| projector | MLP, to be specified before implementation |
| scoring after training | latent_summary_knn, k=5 |
| score fusion | prohibited |
| per-machine selection | prohibited |
| seeds | 13711, 13721, 13731 |
| validation strategy | single-seed pilot (seed=13711) before 3-seed panel |

---

## 6. Augmentation Policy

Augmentations must be mild enough to preserve the acoustic identity of normal samples, and strong enough to provide useful invariance signal.

**Candidate augmentations** (to be finalized before implementation):

| Augmentation | Description |
|---|---|
| small time mask | zero out a short contiguous time segment |
| small frequency mask | zero out a narrow frequency band |
| mild time shift | roll the spectrogram by a few frames |
| mild gain jitter | multiply by a small random gain factor |

**Explicitly prohibited:**

- aggressive augmentation (large masks, heavy noise injection)
- augmentation sweep for hyperparameter search
- label-informed augmentation (e.g., stronger augment for hard-normal)
- per-machine augmentation selection

**Parameters requiring human confirmation before implementation:**

| Parameter | Description | Status |
|-----------|-------------|--------|
| `time_mask_frac` | Maximum fraction of time axis to mask | **TBD** |
| `freq_mask_frac` | Maximum fraction of frequency axis to mask | **TBD** |
| `time_shift_max` | Maximum frames to shift | **TBD** |
| `gain_jitter_range` | Gain multiplier range (e.g., 0.8–1.2) | **TBD** |
| `n_augmentations_per_view` | Number of augmentations to compose | **TBD** |

---

## 7. Difference from Previous Stages

### vs V37 Teacher-Student

V37 is teacher-student local distillation: the student predicts the teacher's local latent features from TF-block-masked input. Stage 9 is invariance SSL: the encoder learns to produce the same representation for two differently-augmented views of the same sample. No teacher model; no local distillation target.

### vs V49 Hard-Normal Consistency

V49 applied consistency loss only to a frozen hard-normal subset (top 10% by kNN score). Stage 9 applies invariance learning to **all** train-normal samples. The mechanism is different: V49 tried to manually compress an already-identified tail; Stage 9 learns a representation where the tail is naturally tight.

### vs V51 OC-SVM

V51 replaced only the scoring boundary on a frozen V37 representation. Stage 9 changes the representation learning objective itself. V51's negative result is the direct motivation for this shift.

### vs V50 Patch Summary

V50 changed the pooling aggregation (global → 2×2 patch). Stage 9 keeps global latent_summary unchanged and only changes the training objective.

---

## 8. Success Gates (for future V53 single-seed pilot)

If V52 preregistration is approved and V53 single-seed training proceeds:

| # | Gate | Threshold |
|---|------|-----------|
| 1 | mean hmean >= V37 baseline − 0.003 | must hold |
| 2 | mean pAUC >= V37 baseline | must hold |
| 3 | at least 2 bottleneck machines improve TPR@99n | must hold |
| 4 | bearing / valve do not collapse | must hold |
| 5 | toycar does not clearly degrade | must hold |
| 6 | normal q99 does not clearly inflate | must hold |
| 7 | gains are not carried by only one machine | must hold |

Reference baseline: V37 full_5view latent_summary_knn (3-seed panel mean hmean ≈ 0.554, pAUC ≈ 0.546).

---

## 9. Failure Criteria

Stage 9 first implementation is recorded as negative if:

- hmean is clearly below V37 baseline
- pAUC is below V37 baseline
- normal q99 increases
- bearing / valve collapse
- toycar degrades
- only gearbox / slider improve while main machines worsen
- useful results require augmentation sweep
- useful results require score fusion
- the invariance objective causes representation collapse (constant features)

---

## 10. Forbidden Actions

The following are **strictly prohibited** under this preregistration:

- Do NOT open V53 during preregistration. V53 implementation is authorized only after this preregistration is manually approved.
- Do NOT implement training code in this step
- Do NOT train any model
- Do NOT change full_5view frontend
- Do NOT change global latent_summary pooling
- Do NOT implement SimCLR negative-pair contrastive learning
- Do NOT do score fusion
- Do NOT do per-machine selection
- Do NOT continue OC-SVM / density / kNN scoring variant search
- Do NOT continue patch / segment pooling search
- Do NOT auto-propose V54
- Do NOT run augmentation sweep to select parameters

---

## 11. Human Decisions Required Before Implementation

Before any V53 implementation begins, the following must be manually confirmed:

| # | Decision | Description |
|---|----------|-------------|
| 1 | SSL objective family | VICReg vs BYOL vs SimSiam (recommended: VICReg for simplicity and explicit variance/covariance control) |
| 2 | Augmentation parameters | time_mask_frac, freq_mask_frac, time_shift_max, gain_jitter_range |
| 3 | Encoder architecture | whether to reuse viewwise_unshared_late_fusion exactly or allow minor adaptations |
| 4 | Representation for scoring | whether to score from encoder latent_summary or projector output |
| 5 | Loss weights | λ_inv, λ_var, λ_cov |
| 6 | Projector architecture | MLP depth, hidden dim |
| 7 | Single-seed pilot command | exact CLI for V53 training |
| 8 | Augmentation sanity check | whether to run a no-training check before V53 training |

---

## 12. Final Sentence

Stage 9 is a new framework-level hypothesis. It must not be treated as a continuation of V39–V51 scorer / pooling / frontend search. No code implementation or training is authorized until this preregistration is reviewed and approved.

---

## 13. V53 Pilot Frozen Definitions

If V52 preregistration is approved and V53 single-seed pilot proceeds, the following are fixed before training. None of these are sweepable inside V53.

### 1. SSL Objective

- **VICReg only**
- No BYOL / SimSiam / SimCLR in V53

### 2. Representation for Scoring

- Use **encoder latent_summary** (channel-wise mean + std + max, 192-dim)
- Do **not** use projector output for scoring
- The projector exists only to provide a loss target; it is discarded after training

### 3. Projector Architecture

| Parameter | Value |
|-----------|-------|
| structure | MLP 192 → 256 → 256 |
| activation | ReLU |
| dropout | 0.0 |
| architecture sweep | prohibited |

### 4. VICReg Loss Weights

| Weight | Value |
|--------|-------|
| λ_inv | 25 |
| λ_var | 25 |
| λ_cov | 1 |
| weight sweep | prohibited |

### 5. Augmentation Policy

| Parameter | Value |
|-----------|-------|
| augmentations | small time mask + small frequency mask + mild gain jitter |
| time_mask_frac | 0.05 |
| freq_mask_frac | 0.05 |
| gain_jitter_range | [0.95, 1.05] |
| n_augmentations_per_view | 2 |
| time shift | not used in V53 |
| augmentation sweep | prohibited |

### 6. Encoder Architecture

| Parameter | Value |
|-----------|-------|
| architecture | viewwise_unshared_late_fusion |
| latent_channels | 64 |
| stem_channels | 8 |
| per_view_channels | 16 |
| dropout_p | 0.0 |
| architecture change | prohibited |

### 7. V53 Single-Seed Pilot Scope

| Parameter | Value |
|-----------|-------|
| seed | 13711 only |
| experiment | full_5view only |
| training data | source train-normal only |
| scoring | latent_summary_knn, k=5 |
| score fusion | prohibited |
| per-machine selection | prohibited |

### 8. If V53 Single-Seed Fails

- Do **not** tune augmentations
- Do **not** tune VICReg weights
- Do **not** attempt BYOL / SimSiam inside V53
- Write failure analysis only
- Do **not** auto-proceed to V54
