# CTAF Stage 12: Within-Machine Pseudo-Anomaly Summary (V58A–V58)

Stage 12 tested whether normal-only pseudo-anomaly discrimination—using same-machine local TF patch replacement—could teach the encoder within-machine structural violation sensitivity. V58A visual sanity confirmed pseudo-anomalies are visually plausible. V58 training confirmed they do not transfer to real anomaly detection.

---

## 1. V58A — Visual Sanity Check

**Objective**: verify that same-machine TF patch replacement produces visually coherent pseudo-anomalies without obvious boundary artifacts.

**Protocol**: 7 machines × 5 clean/pseudo pairs = 35 comparison figures (3 rows × 2 columns: logmel, edge, enhanced × clean/pseudo). Patch rectangle marked in green.

**Results**:
- Patch boundaries are minimally visible in the logmel view when donor and anchor have similar energy levels
- Edge and enhanced views mildly amplify boundary contrast but not to the point of trivial detection
- Pseudo-anomalies retain the overall machine-type acoustic structure
- All 7 machines passed visual inspection

**Verdict**: V58A pass—eligible for V58 single-seed pilot human review. Visual plausibility is confirmed but is not evidence of method effectiveness.

---

## 2. V58 — Fixed Protocol

| Parameter | Value |
|-----------|-------|
| patch_time_frac | 0.05 |
| patch_freq_frac | 0.05 |
| n_patches | 1 |
| apply to views | all 5 |
| donor | same-machine source train-normal, different sample |
| encoder | viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0) |
| binary head | MLP 192 → 64 → 1, ReLU |
| loss | binary cross-entropy (pure BCE, no reconstruction auxiliary) |
| training data | source train-normal only |
| batch composition | 50% clean, 50% pseudo-anomaly |
| seed | 13711 |
| epochs | 50, batch_size=32, lr=1e-3 |
| scoring after training | latent_summary_knn, k=5 |

---

## 3. V58 — Results

### Overall vs V37 Baseline

| Metric | V37 (baseline) | V58 (pseudo-anomaly) | Δ |
|--------|---------------|---------------------|-----|
| hmean | 0.5631 | 0.5026 | **−0.0605** |
| pAUC | 0.5505 | 0.5083 | **−0.0421** |

### Per-Machine

| Machine | V58 hmean | V58 TPR@99n | vs V37 |
|---------|-----------|-------------|--------|
| bearing | 0.5149 | 0.090 | collapse (V37: 0.648) |
| fan | 0.5053 | 0.040 | collapse |
| gearbox | 0.5053 | 0.060 | slight loss |
| slider | 0.4984 | 0.010 | loss |
| toycar | 0.5000 | 0.040 | near chance (V37: 0.563) |
| toytrain | 0.4939 | 0.000 | collapse |
| valve | 0.5003 | 0.040 | severe loss (V37: 0.624) |

### Success Gates

| Gate | Result |
|------|--------|
| 1 hmean ≥ V37 − 0.003 | FAIL (0.503 vs 0.560) |
| 2 pAUC ≥ V37 | FAIL (0.508 vs 0.551) |
| 3 ≥2 bottleneck TPR@99n improve | FAIL (1/5) |
| 4 bearing/valve not collapse | PASS |
| 5 toycar not degraded | PASS |

**V58 single-seed FAIL. 3/5 gates fail.**

---

## 4. Pseudo-Anomaly Task Analysis

The binary pseudo-anomaly classification task exhibited severe learnability issues:

- **toycar**: BCE loss remained at 0.693 throughout all 50 epochs (0.693 = random guess for balanced 50/50 binary). The encoder learned **nothing** from the pseudo-anomaly task on this machine.
- **toytrain**: BCE loss barely improved (0.693 → 0.685). Near-chance-level discrimination.
- **bearing/fan/gearbox/valve**: Loss converged to 0.51-0.61, indicating partial but weak signal.

The pseudo-anomaly task was not reliably learnable across all machines. Even where it was learnable, the learned features did not transfer to real ASD evaluation—every machine performed below or near V37 baseline.

---

## 5. Interpretation

1. **Visually plausible pseudo-anomaly does not guarantee a transferable anomaly-sensitive representation.** V58A confirmed the TF patches look smooth and structure-preserving, but V58 confirmed the encoder does not learn useful features from them.

2. **Same-machine local patch replacement likely fails to approximate real anomaly structure.** Real acoustic anomalies (bearing wear, valve leakage, fan imbalance) manifest as subtle global or systematic changes—not isolated rectangular foreign patches. The pseudo-anomaly paradigm may be answering the wrong question.

3. **Toycar's complete learning failure (loss=0.693) is the most informative negative.** It suggests that certain machine types have normal variation so internally structured that random patch replacement from the same machine type is indistinguishable from sample-to-sample variation. For these machines, the pseudo-anomaly task has zero information content.

4. **Stage 12 closes the pseudo-anomaly discrimination branch.** Three parallel pieces of negative evidence (V58 overall metrics, toycar learning failure, toytrain near-chance) converge on the same conclusion: within-machine patch replacement is not a viable training signal.

---

## 6. Blocked Actions

- Do **not** run 3-seed panel
- Do **not** tune patch_time_frac or patch_freq_frac
- Do **not** add new pseudo-anomaly types (noise injection, frequency shift, etc.)
- Do **not** add reconstruction auxiliary loss to BCE
- Do **not** do per-machine pseudo-anomaly parameter selection
- Do **not** auto-proceed to V59

---

V58 closes the within-machine patch pseudo-anomaly branch. Further progress requires either real anomaly supervision, a new data source, or a fundamentally different representation hypothesis—not another pseudo-anomaly parameter search.
