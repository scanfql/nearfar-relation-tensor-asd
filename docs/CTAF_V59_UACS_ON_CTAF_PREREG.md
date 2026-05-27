# CTAF V59 — CTAF-UACS Machine-Level Backend Calibration Selection Preregistration

## 1. Motivation

CTAF V39–V58 has exhaustively tested training-side modifications. The accumulated evidence is unambiguous:

| Dimension | Stages | Verdict |
|-----------|--------|---------|
| Frontend replacement | V39–V47 | full_5view cannot be replaced by single-mechanism 3-view frontends |
| Pooling replacement | V42–V45, V50 | global latent_summary is the empirical optimum |
| Scoring rule replacement | V51 | kNN is near-optimal scoring for the V37 representation |
| Training objective | V48–V49, V52–V53, V57–V58 | none (hard-normal, VICReg, AAM, pseudo-anomaly) exceed V37 AE/TS |
| External representation | V54–V56A | BEATs provides complementary signal but cannot be safely distilled in |

Every modification to the training pipeline has converged to the same pattern: local improvement on some machines, degradation on others, and no reliable low-FPR gain.

**V59 represents a category shift.** It does not modify the frontend, encoder, latent summary, or training objective. It asks: **can the anomaly scores already produced by the strongest CTAF configuration (V37 full_5view latent_summary_knn) be better utilized through a label-free backend calibration and selection framework?**

CTAF and UACS are not competing methods. They operate at different pipeline stages:

- **CTAF**: frontend + representation + scoring (produces raw anomaly scores)
- **UACS**: backend calibration + candidate selection (takes raw scores and selects the most reliable calibration)

V59 tests whether UACS-style label-free backend selection, applied to frozen V37 CTAF scores, can improve low-FPR tail separation without any retraining or representation change.

---

## 2. Key Principle

### Monotonic normalization alone cannot change AUC/pAUC

If the only operation is a monotonic transform of the raw anomaly scores (z-score, rank, min-max), then the ranking of samples is preserved. AUC, pAUC, and hmean are ranking-based metrics and cannot change under monotonic transforms.

**Therefore V59 is not testing simple z-score normalization.** It is testing **candidate selection**: given a pool of backend candidates (different K, different calibration references, different domain treatments), UACS uses label-free distributional evidence to select the most reliable candidate per machine.

### When UACS can change AUC/pAUC/hmean

UACS can change ranking-based metrics only when:

1. **The candidate pool contains non-rank-preserving operations.** V59 does not vary k. Ranking changes, if any, can only come from source/target-aware domain-specific calibration or `min(source_score, target_score)`-style non-monotonic candidate construction. These are not monotonic transforms of a single score vector.

2. **The selection is label-free.** UACS chooses candidates based on distributional properties of the score histograms (alignment, tail balance, calibration stability), not on AUC/pAUC labels. This is essential for evaluation integrity.

3. **Fallback prevents negative transfer.** The raw CTAF score must always be available as a fallback candidate. If no candidate shows strong label-free evidence of superiority, UACS falls back to the raw score.

4. **Machine-level calibration is properly scoped.** Fleet-level normalization would collapse the distinct score distributions of bearing, fan, gearbox, etc. Machine-level calibration preserves each machine's specific tail structure.

---

## 3. Score Distribution Assumption

### Train scores vs test scores: same metric, different distributions

CTAF V37 NPZ files contain both `train_scores` and `test_scores`. These are computed in the same latent_summary space using the same KNN distance metric (L2-normalized k-NN, k=5):

- **`train_scores`**: leave-one-out KNN distances of train-normal samples to the train-normal memory bank. This is the **normal-reference distribution** — it represents how "normal" samples look to the KNN backend under in-distribution conditions.

- **`test_scores`**: KNN distances of test samples (normal + anomaly, source + target) to the train-normal memory bank. This is the **evaluation distribution** — it contains domain shift (source→target) and anomaly-induced tail behavior.

**Key assumption**: train_scores and test_scores share the same distance metric and embedding space, so they are commensurable. However, they describe two different phenomena:

- train_scores describe the within-normal variation under training conditions
- test_scores describe the response of the backend to both normal and anomalous samples under deployment conditions, including domain shift

**Therefore train_scores can be used as a reference distribution** for estimating the normal score mean/std, median/MAD, q95/q99 thresholds, and tail stability. But they cannot be assumed to represent the full test-time normal score distribution, because test-time source→target domain shift may widen or shift the normal score distribution.

---

## 4. Calibration Level

### Machine-level only. No fleet-level.

**Frozen for V59:**

- Calibration and selection operate at the **machine level** — one reference distribution per machine, one set of candidates per machine, one selection decision per machine.
- **Fleet-level calibration is prohibited.**

**Rationale:**

- CTAF has separate train-normal memory banks per machine (990 samples each)
- Different machines have vastly different normal score distributions: bearing q99≈0.16, fan q99≈0.24, toycar q99≈0.08
- Fleet-level z-score would collapse these differences, polluting bearing/fan/toycar with valve/gearbox score scales
- Machine-level calibration preserves each machine's specific tail structure, which is the property UACS exploits for selection

### Source/target within-machine candidates

If source/target domain metadata is available for train-normal samples, **within-machine source/target-aware candidates** can be constructed (separate reference distribution per domain within each machine). If not available, source/target-specific candidates are marked as unavailable but do not invalidate the experiment.

---

## 5. Source/Target min Rule

### min() is a candidate, not the default

`min(source_score, target_score)` is a conservative ensemble rule: a sample is scored anomalous only if both domain backends agree. It reduces false positives at the cost of potentially suppressing anomaly scores for samples that are well-detected in only one domain.

**Frozen for V59:**

- `min(source_score, target_score)` is **one candidate** in the pool (C4)
- It is **not** the default or the only scoring rule
- Whether to use it is decided by UACS label-free selection per machine
- The raw CTAF score (single-domain, no min) is always available as a fallback candidate (C0/C5)

---

## 6. Candidate Pool Draft

V59 first version uses only CTAF V37 score candidates. No BEATs, no V58, no transient_3view, no patch scoring, no score fusion across experiments.

| ID | Description | Requires | Rank-preserving? | Can change AUC/pAUC? |
|----|-------------|----------|-----------------|---------------------|
| **C0** | Raw V37 full_5view latent_summary_knn (k=5) | nothing | baseline | — |
| **C1** | Machine-level z-score calibration using source train-score mean/std | train_scores | yes | **no** — calibration diagnostic only |
| **C2** | Machine-level robust calibration using source train-score median/MAD | train_scores | yes | **no** — calibration diagnostic only |
| **C3** | Machine-level source+target pooled calibration (separate ref per domain within machine) | source/target metadata for train-normal | no (if domain-split) | **yes** (if domain-split) |
| **C4** | Source/target min calibrated score (min of calibrated source, calibrated target) | source/target metadata for train-normal AND test | no | **yes** |
| **C5** | Fallback: raw CTAF score (identical to C0) | nothing | baseline | — |

**Critical clarification:** C1 and C2 are **calibration diagnostics**. They apply the same monotonic transform to the same score vector, so sample ranking is unchanged. They are retained to report calibration quality (q99 stability, tail balance, domain alignment) but are not expected to improve AUC/pAUC. Only C3 (domain-specific calibration) and C4 (non-monotonic min combine) can affect ranking, and both require source/target metadata.

**Important constraints:**

- C3 and C4 are only included if source/target domain metadata exists for both train-normal and test samples. If metadata is missing or unreliable, these candidates are marked **unavailable** — no synthetic labels are created.
- C5 is strictly identical to C0. It exists to ensure the fallback path is visible in the candidate pool and cannot be accidentally removed.
- All candidates are derived from the same frozen V37 score NPZ files. No new score computation, no model inference.

---

## 7. UACS Selection Evidence

UACS selects candidates based on label-free properties of the score distribution. The evidence must not use anomaly labels, test labels, AUC, or pAUC.

### Evidence dimensions

| Evidence | Description | What it measures |
|----------|-------------|-----------------|
| **Alignment** | Source-target normal score distribution gap (KS statistic, median gap, IQR gap) | Domain invariance of the scoring backend |
| **Upper-tail stability** | Anomaly score tail behavior relative to normal tail (tail balance, tail inflation) | Whether calibration inflates or compresses the low-FPR tail |
| **KNN smoothness** | Whether the K value provides sufficient neighborhood resolution without over-smoothing | Backend granularity |
| **Calibration prior** | Prior preference for z-score calibration over raw (small positive weight) | Calibration regularization |
| **Fallback rule** | If no candidate exceeds a minimum quality threshold, fall back to C0/C5 | Safety net |

### Selection formula (design draft, parameters TBD)

```
UACS_score(candidate) = -alignment_penalty
                        + α * tail_balance
                        + smoothness_prior(k)
                        + calib_prior(candidate)
                        - inflation_penalty(if tail inflation detected)
                        - collapse_penalty(if domain collapse detected)
```

**Explicitly forbidden in selection:**

- No AUC or pAUC as selection evidence
- No anomaly label access
- No per-machine oracle (comparing candidates against ground truth)
- No post-hoc selection based on test-set metrics

---

## 8. Baseline

**Primary baseline:** V37 full_5view latent_summary_knn (seed 13711)

| Metric | Value |
|--------|-------|
| hmean | 0.5631 |
| pAUC | 0.5505 |

V59 compares the UACS-selected candidate against this baseline. The raw CTAF score (C0) must also be reported as a reference point.

If V59 passes single-seed, a 3-seed panel (seeds 13711, 13721, 13731) extension is pre-registered as a valid follow-up but is not part of V59 itself.

---

## 9. Success Gates

V59 is positive only if all of the following hold:

| # | Gate | Threshold |
|---|------|-----------|
| 1 | mean hmean >= V37 baseline | 0.5631 |
| 2 | mean pAUC >= V37 baseline | 0.5505 |
| 3 | at least 2 bottleneck machines improve TPR@99n | must hold |
| 4 | bearing / valve do not collapse | must hold |
| 5 | toycar does not clearly degrade | must hold |
| 6 | normal q99 does not clearly inflate | must hold |
| 7 | gains are not carried by only one machine | must hold |
| 8 | selected candidate is not identical to C0 on all machines, unless metrics still improve through valid UACS selection | selected candidate differs from C0 on ≥1 machine |

---

## 10. Failure Criteria

V59 fails if any of the following occur:

- hmean is below V37 baseline
- pAUC is below V37 baseline
- normal q99 inflates on any machine beyond 20% of baseline
- bearing / valve collapse
- toycar degrades
- only one machine improves
- selected candidate depends on unavailable source/target metadata that was not present in the original V37 output
- candidate selection requires anomaly labels or post-hoc AUC optimization
- improvement is purely from monotonic score scaling with no ranking change
- raw CTAF fallback (C0) is beaten only by a candidate that UACS does not select (oracle regret)

---

## 11. Forbidden Actions

The following are **strictly prohibited** under this preregistration:

- Do NOT implement code in this step
- Do NOT run any experiment
- Do NOT train any encoder
- Do NOT modify the full_5view frontend
- Do NOT modify the latent_summary pooling
- Do NOT introduce BEATs or any external representation
- Do NOT do score fusion across experiments
- Do NOT add V58/V55/patch/transient scores to the candidate pool
- Do NOT perform per-machine oracle selection (using test labels)
- Do NOT use anomaly labels in any form during candidate selection
- Do NOT use test labels to tune calibration
- Do NOT expand the candidate pool beyond C0–C5 in V59
- Do NOT sweep UACS evidence weights
- Do NOT auto-proceed to the next experiment version

---

## 12. Human Decisions Required Before Implementation

| # | Decision | Recommendation |
|---|----------|---------------|
| 1 | Confirm V37 NPZ key names | Check: `train_scores`, `test_scores`, `d_test`, `y_test` |
| 2 | Confirm source/target domain metadata exists in V37 NPZ | Check: `d_test` array contains "source"/"target" strings |
| 3 | Confirm target train-normal reference is available | V37 uses source-only memory bank; target train-normal may not exist |
| 4 | Decide whether C3/C4 are available | If no target train-normal → C3/C4 unavailable (not an experiment failure) |
| 5 | UACS evidence formula weights | `α` for tail_balance, prior weights, penalty thresholds — to be frozen before audit |
| 6 | Fallback rule threshold | Minimum UACS quality score below which C0 is used |
| 7 | Whether to run score-file schema audit first | Recommended — confirm all NPZ keys and shapes before any selection logic |

---

## 13. Final Sentence

V59 is a backend calibration-selection hypothesis on top of the fixed CTAF full_5view V37 representation. It must not be treated as a continuation of frontend, pooling, scoring-rule, BEATs, AAM, or pseudo-anomaly search. No implementation or experiment is authorized until this preregistration is reviewed and approved.
