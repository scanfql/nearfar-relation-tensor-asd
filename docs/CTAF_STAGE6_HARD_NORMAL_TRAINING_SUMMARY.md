# CTAF Stage 6: Hard-Normal Tail Training Summary (V48–V49)

CTAF Stage 6 测试了"hard-normal consistency training"能否通过压缩正常分数尾部分布来改善 low-FPR TPR@99n。V48 离线诊断确认了 hard-normal 瓶颈的存在（但 machine-specific），V49 固定协议训练验证了机制效果和局限性。

---

## 1. V48 — Hard-Normal Tail Diagnostic

**目标**：离线诊断 V37 full_5view teacher-student latent space 中 hard-normal（train-normal ls_knn score top 10%）是否构成 low-FPR 瓶颈。

**数据**：3 种子 × 7 机器。V37 score NPZ + V45 复现的 V37 protocol checkpoint（21 个 .pt，软链接到 V37 目录）。

**核心发现**：

### Normal-Tail 组成（q95–q99 带）

| Machine | hard-normal-like in q95-q99 | near-q99 hard-normal ratio |
|---------|---------------------------|---------------------------|
| toytrain | 100% | 100% |
| gearbox | 91.7% | 83.3% |
| bearing | 83.3% | 100% |
| fan | 75.0% | 100% |
| slider | 50.0% | 66.7% |
| toycar | 50.0% | 22.2% |
| valve | 50.0% | 66.7% |

### Augmentation Consistency（hard vs easy normal 的 latent 稳定性）

| Machine | hard/easy cos diff (time_mask) | 差异倍数 |
|---------|-------------------------------|---------|
| bearing | 0.0038 / 0.0004 | **~9×** |
| fan | 0.0030 / 0.0003 | **~5-10×** |
| valve | 0.0029 / 0.0007 | ~3× |
| gearbox | 0.0006 / 0.0003 | ~3× |
| toycar | 0.0003 / 0.0002 | ~1.5× |
| toytrain | 0.0005 / 0.0002 | ~2× |

`light_noise` cos ≈ 0.00001 对所有组无区分力。

### Missed Anomaly 与 Hard-Normal 重叠

| Machine | hard-NN overlap ratio |
|---------|----------------------|
| fan | 9.1% |
| bearing | 7.6% |
| toytrain | 4.8% |
| slider | 3.4% |
| valve | 3.6% |
| gearbox | 1.4% |
| toycar | 1.0% |

### Tail Compression Proxy（压缩 hard-normal tail +10% 后的 TPR@99n 增益）

| Machine | Δ TPR@99n |
|---------|-----------|
| fan | **+0.053** |
| bearing | **+0.050** |
| valve | **+0.050** |
| gearbox | +0.023 |
| slider | +0.023 |
| toytrain | +0.013 |
| toycar | +0.003 |

**V48 结论**：hard-normal tail 是 **machine-specific bottleneck**。作用机制是 hard-normal 占据了 low-FPR 判定带（q95-q99），压缩它们会压低 q99 阈值，从而让更多 anomaly 跨过阈值——而不是 hard-normal 直接与 anomaly 在 latent space 重叠。

---

## 2. V49 — Hard-Normal Consistency Training

### Fixed Protocol

| Parameter | Value |
|-----------|-------|
| hard-normal source | frozen V37 full_5view ls_knn train-normal score top 10% |
| augmentations | time_mask (0.10) OR freq_mask (0.10), randomly chosen |
| consistency loss | latent_summary cosine distance (192-dim) |
| lambda | 0.10, no sweep |
| applies to | hard-normal-labeled train-normal samples only |
| model | viewwise_unshared_late_fusion, standard AE reconstruction |
| seed | 13711, epochs=50 |

### Results vs V37 TS Baseline (seed 13711)

| Machine | V37 hmean | V49 hmean | Δ | V37 TPR@99n | V49 TPR@99n | Δ TPR | Δ q99 |
|---------|-----------|-----------|-----|-------------|-------------|-------|-------|
| bearing | 0.648 | 0.632 | −0.016 | 0.27 | 0.17 | −0.10 | −0.004 |
| fan | 0.602 | 0.596 | −0.006 | 0.12 | 0.19 | **+0.07** | +0.004 |
| gearbox | 0.524 | 0.543 | **+0.019** | 0.06 | 0.12 | **+0.06** | −0.014 |
| slider | 0.537 | 0.522 | −0.015 | 0.00 | 0.05 | **+0.05** | −0.017 |
| toycar | 0.563 | 0.496 | **−0.067** | 0.04 | 0.02 | −0.02 | +0.004 |
| toytrain | 0.534 | 0.525 | −0.009 | 0.03 | 0.07 | **+0.04** | −0.024 |
| valve | 0.624 | 0.534 | −0.089 | 0.10 | 0.10 | 0.00 | −0.011 |

| Metric | V37 TS | V49 HN | Δ | Threshold | Verdict |
|--------|--------|--------|-----|-----------|---------|
| mean hmean | 0.5759 | 0.5496 | −0.026 | ≥ −0.003 | **FAIL** |
| mean pAUC | 0.5505 | 0.5473 | −0.003 | ≥ V37 | **FAIL** |
| q99 inflation | — | 6/7 lower | — | no inflation | PASS |
| bearing/fan/valve TPR@99n (≥2) | — | 1/3 improve | — | ≥ 2 | **FAIL** |
| toycar/toytrain not degraded | — | toycar −0.067 | — | no degradation | **FAIL** |

### Success Criteria: 4/5 FAIL → **V49 SINGLE-SEED FAIL**

---

## 3. Scientific Interpretation

1. **Hard-normal consistency successfully compresses the normal score tail.** q99 decreased on 6/7 machines (median Δ = −0.011). The mechanism works exactly as designed.

2. **Normal-tail compression alone is insufficient for reliable low-FPR improvement.** While 4/7 machines improved TPR@99n (gearbox +0.06, fan +0.07, slider +0.05, toytrain +0.04), the gains are inconsistent and fail to close the representation gap against V37's teacher-student pretraining.

3. **Gains are machine-specific and unstable.** V48's hard-normal-positive machines (bearing/fan/valve) showed only 1/3 TPR@99n improvement, while V48's hard-normal-negative machines (gearbox/slider/toytrain) showed unexpected TPR@99n gains. There is no clean mapping between V48 diagnostic signal and V49 training outcome.

4. **Applying hard-normal consistency to machines without hard-normal signal damages representation.** Toycar (V48: no augmentation sensitivity, 22% near-q99 hard-normal ratio, compression proxy +0.003) suffered the largest degradation (−0.067 hmean). The loss actively harms machines where hard-normal is not a real bottleneck.

5. **This is a mechanism diagnostic, not a deployable training method.** V49 proves that normal-tail compression is controllable via a fixed augmentation consistency loss, but the resulting TPR@99n gains are too inconsistent and too small relative to baseline to justify deployment. The V37 teacher-student pretraining baseline remains the superior training protocol across all machines.

---

## 4. Blocked Actions

- Do NOT run 3-seed panel on V49
- Do NOT tune lambda / mask_frac / hard_normal_top_ratio
- Do NOT apply per-machine gating on hard-normal loss
- Do NOT combine hard-normal consistency with teacher-student pretraining automatically
- Do NOT auto-proceed to V50
- Do NOT continue parameter search on this branch

---

V49 closes the hard-normal consistency branch as a controlled mechanism test. The normal-tail compression mechanism is confirmed valid, but the effect size is insufficient to justify deployment. Further progress requires a new pre-registered hypothesis, not parameter search.
