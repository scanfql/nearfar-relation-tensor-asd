# CTAF Stage 5: Mechanism Frontend Summary (V39–V47)

CTAF 项目 Stage 5：对三种 mechanism-specific 3-view 候选前端进行系统性评估。从 input-level prescreen 到 full training validation，最终结论是单独机制前端不能替代 full_5view 主线。

---

## 1. V39 — Frontend Prescreen (Stage-0, No Training)

**目标**：用 pre-registered 四门系统筛选三个候选 3-view 前端 (modulation, cepstral, transient) 的 input-level 异常分离信号。

**方法**：对 5 台 bottleneck 机器 (gearbox, slider, toycar, toytrain, fan)，提取各前端张量，计算 robust_d（IQR 分母 Cohen's d）和冗余度（Pearson r vs 5 baseline views）。无模型训练。

| 前端 | Views | mean robust_d_missed | max_r | Stage-0 |
|------|-------|---------------------|-------|---------|
| transient_3view | logmel + spectral_flux + transient_energy | 0.738 | **0.054** | PASS |
| modulation_3view | logmel + slow + fast envelope | 3.638 | 0.969 | FAIL — Gate C (redundancy) |
| cepstral_3view | logmel + low-Q MFCC + periodicity | NaN | 0.609 | FAIL — implementation invalid |

**判定**：
- `transient_3view` 进入 V40：低冗余、5/5 Gate A 通过
- `modulation_3view` 被排除：modulation_low 与 logmel r=0.969，几乎是平滑版 logmel
- `cepstral_3view` 标记为 implementation-invalid（MFCC upsampling 产生 NaN），不是科学失败

---

## 2. V40 — Transient Training Validation (Single-Seed, Seed=13711)

**配置**：`viewwise_unshared_late_fusion`, lc=64, sc=8, pvc=16, epochs=50, source normal only

训练 transient_3view、logmel_only、full_5view 三项实验在 5 台 bottleneck 机器。

**5 bottleneck 机器均值**：

| Experiment | Views | Mean hmean | Mean pAUC |
|------------|-------|------------|-----------|
| logmel_only | 1 | 0.5394 | 0.5168 |
| full_5view | 5 | 0.5353 | 0.5353 |
| transient_3view | 3 | 0.5441 | 0.5213 |

**Success Criteria**：

| # | Criteria | Result | Verdict |
|---|----------|--------|---------|
| 1 | transient hmean > logmel_only | 0.5441 > 0.5394 (+0.005) | PASS |
| 2 | transient hmean > full_5view | 0.5441 > 0.5353 (+0.009) | PASS |
| 3 | dcase_auc > V16 baseline (0.5495) | 0.5592 (+0.010) | PASS |
| 4 | pAUC >= V16 baseline (0.5423) | 0.5213 (−0.021) | **FAIL** |
| 5 | KS not significantly worse | 0.3320 (+0.037) | PASS |
| 6 | >= 2 bottleneck TPR@99n improve | 1/5 (only toytrain) | **FAIL** |

**判定**：V40 SINGLE-SEED FAIL。transient_3view 的输入层信号未稳定转化为 low-FPR tail separation。主要失败机制是 normal-tail inflation（训练后 q95/q99 大幅上升）。

---

## 3. V41 — Signal Preservation Diagnostic (No Retraining)

**目标**：追踪 V40 训练后 transient 信号穿过 encoder 6 个阶段的变化。

通过 `encode_staged()` 返回输入层、stem、enc2、fused_pre、latent_map、latent_summary 的中间表示，计算每层 robust_d。

**各阶段平均 |robust_d_missed|**：

| Stage | Mean |rd| |
|-------|----------|
| Input (raw tensor) | 0.999 |
| Stem (enc1) | 0.941 |
| Enc2 | 0.903 |
| Fused multi-view | 0.903 |
| Latent map (fusion_enc) | 1.080 |
| Latent summary (pooled) | **1.771** |

**核心发现**：信号未在 encoder 中衰减，反而被放大（|rd| 0.999 → 1.771）。但放大后的信号不能转化为 low-FPR tail separation（TPR@99n/pAUC 未改善）。

**机器差异**：
- toytrain：无信号变有信号（rd 0.02 → 1.43），唯一成功模式
- toycar：信号被 encoder 反转（rd 1.67 → −0.22）
- gearbox/slider/fan：信号放大但 scoring 无法捕获

**结论**：问题不在 frontend/stem/encoder 丢失信号，而在 **global latent_summary pooling 不能将局部表示转化为 TPR@99n**。

---

## 4. V42–V44 — Segment-Level Scoring (No Retraining / 3-Seed)

**目标**：验证 segment-level local kNN 替代 file-level global latent_summary 能否恢复 transient 的 low-FPR 信号。

**V42 (单种子 5 bottleneck)**：

| Score mode | hmean | pAUC | TPR@99n improved |
|------------|-------|------|------------------|
| latent_summary_knn | 0.4986 | 0.4958 | baseline |
| enc2_segment_knn_top5 | **0.5056** | **0.5187** | 2/5 |
| latent_map_segment_knn_top5 | **0.5075** | **0.5233** | 3/5 |

**V43 (全 7 机)**：两种 segment score 均 PASS，bearing/valve 无退化。

**V44 (3-Seed Panel)**：

| 指标 | latent_map_segment | vs ls_knn |
|------|--------------------|-----------|
| hmean | 0.5033 ± 0.0019 | +0.006 |
| pAUC | 0.5103 ± 0.0058 | +0.017 |
| TPR@99n | 5/5 machines improve | — |

**结论**：`transient_3view + latent_map_segment_knn_top5` 是稳定的弱前端补偿方案。但 absolute hmean/pAUC 仍低于 full_5view 主线。

---

## 5. V45 — Transfer Audit to full_5view (No Retraining)

**目标**：测试 segment scoring 能否迁移到 stronger full_5view 表示。

在 V16 full_5view AE 和 V37 full_5view TS 上测试：

| 来源 | ls_knn hmean | segment hmean | 判定 |
|------|-------------|--------------|------|
| V16 full_5view AE | **0.5447** | 0.5298 (−0.015) | FAIL |
| V37 full_5view TS | **0.5502** | 0.5196 (−0.030) | FAIL |

**结论**：segment scoring **不能迁移到 full_5view**。full_5view 的 5-view 互补性使 global pooling 已足够好，segment scoring 反而破坏。segment scoring 是弱前端的补偿机制，不是通用替换。

---

## 6. V46 — Cepstral Frontend Repair + Stage-0 Rerun (No Training)

**目标**：修复 cepstral_3view 在 V39 中的 NaN implementation failure，重跑 Stage-0 prescreen。

**修复内容**：`_upsample_freq()` 使用 NaN-safe interp1d (fill_value="extrapolate" + 边界裁剪)，`_compute_periodicity_from_mfcc()` 对全零帧加 epsilon 保护。

**Stage-0 结果**：

| Gate | 结果 |
|------|------|
| A (robust_d_missed >= 0.30) | 5/5 PASS |
| B (>= 2 machines pass A) | PASS |
| C (max_r <= 0.85) | max_r=0.6089 PASS |
| D (robust_d_detected > 0) | 4/5 PASS |
| Stage-0 | **PASS** |

| Machine | robust_d_missed |
|---------|-----------------|
| gearbox | 3.376 |
| slider | 2.110 |
| toycar | **5.905** |
| toytrain | 0.832 |
| fan | 2.429 |

mean robust_d_missed = **2.930**（transient_3view 的 4 倍）

**cepstral_low_quefrency** r=0.609（与 logmel/delta 族有中等相关性，但低于 0.85 阈值）、**cepstral_periodicity** r=0.095（完全独立）。

**结论**：cepstral_3view 从 implementation-invalid 升级为 Stage-0 cleared。输入层信号很强（尤其 toycar rd=5.90，transient 上 toycar 是最差机器）。进入 V47 训练验证。

---

## 7. V47 — Cepstral Training Validation (Single-Seed, Seed=13711)

**配置**：与 V40 完全一致。训练 cepstral_3view、logmel_only、full_5view。

**全 7 机均值**：

| Experiment | Views | Mean hmean | Mean pAUC | Mean KS |
|------------|-------|------------|-----------|---------|
| logmel_only | 1 | 0.5436 | 0.5246 | 0.4029 |
| full_5view | 5 | **0.5535** | **0.5478** | 0.3286 |
| cepstral_3view | 3 | 0.5070 | 0.5137 | 0.2629 |

**5 bottleneck 机器**：

| Machine | logmel_only hmean | cepstral_3view hmean | Delta |
|---------|-------------------|---------------------|-------|
| gearbox | 0.5562 | 0.5186 | −0.038 |
| slider | 0.5208 | 0.4903 | −0.030 |
| toycar | 0.4998 | 0.4954 | −0.004 |
| toytrain | 0.5224 | 0.5264 | +0.004 |
| fan | 0.6027 | 0.5721 | −0.031 |

0/5 bottleneck machines 的 TPR@99n 改善。

**Success Criteria**：

| # | Criteria | Result | Verdict |
|---|----------|--------|---------|
| 1 | cepstral hmean > logmel_only | 0.5070 < 0.5436 (−0.037) | **FAIL** |
| 2 | cepstral hmean > full_5view | 0.5070 < 0.5535 (−0.046) | **FAIL** |
| 3 | hmean > V16 baseline (0.5495) | 0.5070 | **FAIL** |
| 4 | pAUC >= V16 baseline (0.5423) | 0.5137 | **FAIL** |
| 5 | KS not worse than V16 (0.2949) | 0.2629 | PASS |
| 6 | >= 2 bottleneck TPR@99n improve | 0/5 | **FAIL** |
| 7 | vs logmel_only machines improved | 0/7 (summary.csv) | **FAIL** |

**V47 Verdict：SINGLE-SEED FAIL. 5/6 criteria fail.**

**关键观察**：
- cepstral_3view 的 3-view tensor 表现**低于 V=1 的 logmel_only**（全 7 机 hmean 0.507 vs 0.544）
- toycar cepstral input rd=5.90 但训练后 hmean 仅 0.4954（vs logmel 0.4998）；最强输入信号完全未转化
- cepstral_3view 的 KS=0.2629 低于 logmel_only=0.4029，说明 domain invariance 更好，但分离度不足使这点无意义

**判定**：cepstral_3view 是 **scientific negative**（非 implementation failure）。输入层信号最强的前端 (rd=2.93)，训练后表现最差 (hmean=0.507)。这构成"input-level separability is insufficient"的最强证据。

---

## 8. Final Conclusion

### 事实陈述

1. **full_5view + latent_summary_knn 仍是 CTAF 当前强表示主线**（V=5, hmean=0.554, pAUC=0.548）。

2. **三种 mechanism-specific 3-view 前端均不能替代 full_5view**：
   - `modulation_3view`：输入信号最强 (rd=3.64) 但高冗余 (r=0.969)，Stage-0 就失败
   - `transient_3view`：Stage-0 通过，训练后 hmean 勉强 > logmel (0.544 vs 0.539)，但 pAUC/TPR@99n 失败；segment scoring 可补偿但 absolute 值仍低于主线
   - `cepstral_3view`：Stage-0 最强信号 (rd=2.93)，训练后反而**低于** logmel_only (0.507 vs 0.544)，构成 input signal ≠ training success 的最强反例

3. **input-level separability alone is insufficient**：两个前端（transient, cepstral）都在 Stage-0 表现出强异常分离，但训练后均无法稳定转化为 low-FPR tail separation。这是跨前端、跨评分模式的稳健结论。

4. **segment scoring 是弱前端补偿机制，不是通用方案**：V42–V44 证明它对 transient_3view 有效，但 V45 证明它不能迁移到 full_5view。

### 明确禁止的后继操作

- 不继续 modulation/transient/cepstral 参数搜索
- 不做 6-view 版本
- 不做 frontend fusion
- 不做 per-machine frontend selection
- 不做 score fusion
- 不自动开启新版本

### 建议的下一步方向

下一步不由本文档规定，但可考虑的方向包括：
- 回到 full_5view 主线，在表示层面（而非输入层面）寻找 low-FPR tail 改善
- 探索非 AE 框架（如 normalizing flow、contrastive learning）
- 评估是否 DCASE Task 2 的 low-FPR 要求本质超出了当前 AE + kNN 框架的能力边界

---

**所有结论基于 pre-registered gates 和固定协议。无 post-hoc 参数选择或 per-machine tuning。**
