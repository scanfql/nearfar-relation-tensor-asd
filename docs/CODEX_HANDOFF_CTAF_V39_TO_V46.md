# CODEX_HANDOFF_CTAF_V39_TO_V46.md

CTAF 项目交接文档：V39 至 V46 证据链与执行指令

---

## 1. Project Context

| 项 | 值 |
|---|-----|
| 项目路径 | `/data1/test/LAES_2025/new_projects/clean_tensor_asd` |
| 数据路径 | `/data1/test/LAES_2025/data/dcase` |
| Python 环境 | `source ~/miniconda3/etc/profile.d/conda.sh && conda activate scanf` |
| GPU | 2× RTX 3090 (cuda:0, cuda:1) |
| 项目目标 | CTAF 强前端实验：构建 clean label-free 异常声检测前端 |
| 核心瓶颈 | **low-FPR tail separation**（TPR@99n、pAUC），而非普通 hmean 排名 |
| 协议约束 | label-free: 只用 source-domain normal 训练，anomaly label 仅用于最终 evaluation |

---

## 2. Pre-V39 Frozen Background (简要继承)

以下为 V16–V38 已冻结结论，不展开讨论：

- **V16**：`full_5view + viewwise_unshared_late_fusion + latent_summary_knn` 是当前主线有效基线。5-seed panel mean hmean≈0.55, pAUC≈0.542。
- **V30–V35**：temporal pyramid / SSL 预训练 / tail compactness loss 系列实验，未解决 low-FPR tail separation。
- **V36**：`teacher_student_top5` 主分数失败，但 `latent_summary_knn` 诊断表现正。
- **V37**：teacher-student pretraining 有 representation gain，但 TPR@99n 未改善。
- **V38**：density scoring audit 失败 — 说明不是简单把 kNN 换成 density 就能修复 tail。

**冻结结论**：继续调 AE 架构 / SSL 任务 / density 评分规则的边际收益很低。下一步应该做**机制型前端归因**而非评分规则替换。

---

## 3. Work Since V39 (已参与实验 V39–V45)

### V39 — Frontend Prescreen (Stage-0, no training)

**脚本**：`scripts/diagnose_v39_frontend_prescreen.py`

评估三个候选 3-view 前端在 5 台 bottleneck 机器上的 input-level 异常分离信号：

| 前端 | Input robust_d | Redundancy (max r) | Stage-0 |
|------|---------------|-------------------|---------|
| `transient_3view` (logmel + spectral_flux + transient_energy) | mean 0.738 | 0.054 | **PASS** |
| `modulation_3view` (logmel + slow + fast) | mean 3.638 | 0.969 | FAIL — modulation_low 与 logmel r=0.969 |
| `cepstral_3view` (logmel + low-Q MFCC + periodicity) | all NaN | 0.609 | FAIL — **implementation invalid** (NaN) |

**关键代码依赖**：
- `src/features/logmel.py`：新增 `compute_mfcc_from_logmel()` 函数
- `src/features/tensor_builder.py`：已支持 `transient_3view` (logmel + spectral_flux + transient_energy)
- `src/run_experiment.py`：已注册 `transient_3view` 为合法 experiment

**结论**：只有 `transient_3view` 进入 V40。`modulation_3view` 因冗余度被排除。`cepstral_3view` 因 NaN 问题应视为 **implementation/preprocessing invalid**，不应作为科学失败结论。

### V40 — Transient Training Validation (single-seed, seed=13711)

**脚本**：`src/run_experiment.py` (已扩展支持 transient_3view)

训练 `transient_3view` 在 5 台 bottleneck 机器上。固定配置：`viewwise_unshared_late_fusion`, lc=64, sc=8, pvc=16, dp=0.0, epochs=50。

| 标准 | 结果 | 判定 |
|------|------|------|
| hmean > logmel_only | 0.5441 > 0.5394 | PASS |
| hmean > full_5view | 0.5441 > 0.5353 | PASS |
| dcase_auc > V16 baseline (0.5495) | 0.5592 | PASS |
| pAUC >= V16 baseline (0.5423) | 0.5213 | **FAIL** |
| >= 2/5 TPR@99n improve | 1/5 (only toytrain) | **FAIL** |

**失败机制**：normal-tail inflation — transient_3view 训练后 normal 分数的 q95/q99 大幅上升，而 anomaly 分数未充分超越新阈值。

**结论**：transient 有机制特异性信号（toytrain 改善显著），但 AE + latent_summary 管道不能稳定保留 low-FPR tail separation。

### V41 — Signal Preservation Diagnostic (no retraining)

**脚本**：`scripts/diagnose_v41_signal_preservation.py`

使用 V40 已训练模型，通过 `encode_staged()` 追踪 robust_d 穿过 6 个 encoder 阶段的变化：

| 阶段 | 平均 |robust_d_missed| |
|------|-----------------------------|
| Input (raw tensor) | 0.999 |
| Stem (enc1) | 0.941 |
| Enc2 | 0.903 |
| Latent map | 1.080 |
| Latent summary | **1.771** |

**核心发现**：信号**没有在 encoder 中衰减**，反而在 latent_summary 被**放大**（1.8×）。各机器模式不同：
- **toycar**：input rd=1.67 → summary rd=-0.22（encoder 反转了信号）
- **gearbox/slider/fan**：信号放大但 scoring 无法捕获
- **toytrain**：无信号变有信号（rd 0.02→1.43）

**模型改动**：`src/models/conv_ae.py` 中 `ViewwiseUnsharedLateFusionAutoencoder` 新增 `encode_staged()` 方法，返回 6 层中间表示。

**结论**：问题不在 frontend/stem/encoder 丢失信号，而在 **global latent_summary pooling 无法将局部表示转化为 low-FPR tail separation**。

### V42 — Segment-Level Scoring Audit (no retraining)

**脚本**：`scripts/diagnose_v42_segment_scoring.py`

在 transient_3view 上测试 segment-level local kNN 替代 file-level latent_summary_knn：

| Score mode | hmean | pAUC | TPR@99n improved | q99 |
|------------|-------|------|------------------|-----|
| latent_summary_knn | 0.4986 | 0.4958 | baseline | ~1.0 |
| enc2_segment_knn_top5 | **0.5056** | **0.5187** | 2/5 | ~0.39 |
| latent_map_segment_knn_top5 | **0.5075** | **0.5233** | 3/5 | ~0.55 |

**结论**：segment scoring 一致优于 latent_summary_knn，证实 V41 假设 — 问题在 global pooling。

### V43 — Full-Machine Validation (no retraining)

**脚本**：`scripts/diagnose_v43_full_machine.py`

扩展到全部 7 台机器（补了 bearing/valve 的 V40 checkpoint），两种 segment score 均 PASS：
- bearing/valve 无灾难退化
- 全机均值一致优于 ls_knn

### V44 — 3-Seed Panel (seed=13711, 13721, 13731)

**脚本**：`scripts/train_and_score_v44_seedpanel.py`

transient_3view + latent_map_segment_knn_top5 跨 3 种子：

| 指标 | 均值 ± std | vs ls_knn |
|------|-----------|-----------|
| hmean | 0.5033 ± 0.0019 | +0.006 |
| pAUC | 0.5103 ± 0.0058 | +0.017 |
| TPR@99n | 5/5 machines improve | — |

**结论**：稳定的 transient 分支 candidate。但 absolute hmean/pAUC 仍低于 full_5view 主线。

### V45 — Transfer Audit to full_5view (no retraining)

**脚本**：`scripts/diagnose_v45_segment_transfer.py`

测试 segment scoring 能否迁移到 stronger full_5view 表示：

| 来源 | 模式 | hmean | pAUC |
|------|------|-------|------|
| V37 full_5view TS | ls_knn | **0.5502** | **0.5409** |
| V37 full_5view TS | latent_map_segment | 0.5196 | 0.5033 |
| V16 full_5view AE | ls_knn | **0.5447** | **0.5404** |
| V16 full_5view AE | latent_map_segment | 0.5298 | 0.5074 |

**结论**：segment scoring **不能迁移到 full_5view**。full_5view 的 5-view 互补性使 global pooling 已足够好，segment scoring 反而破坏。

---

## 4. Current Scientific Conclusion

**必须明确**：

1. `full_5view + latent_summary_knn` 仍是**强表示主线**。不替换。
2. `transient_3view + latent_map_segment_knn_top5` 是有价值的**机制诊断 / 弱前端补偿分支**，不是 mainline replacement。
3. Segment scoring 只在弱前端（transient_3view）上有效，不应继续迁移到 full_5view。
4. cepstral_3view 在 V39 中因 NaN/implementation 失败，属于 **preprocessing invalid**，不是科学失败，需要修复后重测。

**禁止**：
- 不继续调 segment k/top_percent
- 不做 enc2/latent_map score fusion
- 不做 per-machine layer/score selection
- 不把 transient_3view 结果包装成 mainline replacement

---

## 5. Current Next Task: V46 — Cepstral Frontend Implementation Sanity + Stage-0 Rerun

### 原因

V39 中 `cepstral_3view` 失败是 NaN / MFCC upsampling 问题（implementation/preprocessing invalid）。需要先修复 cepstral feature extraction，再重新做 no-training Stage-0 prescreen。

### Candidate Frontend

```
cepstral_3view:
  - logmel_base
  - cepstral_low_quefrency   (MFCC[1:13] upsampled to [128, T])
  - cepstral_periodicity      (autocorrelation from MFCC[20:], upsampled to [128, T])
```

### 分析机器

`gearbox, slider, toycar, toytrain, fan`

### 参考 Score 来源

`results/v37_teacher_student_pretrain_seedpanel/seed_13711`
experiment: `full_5view`, score_mode: `latent_summary_knn`

### 固定 Stage-0 Gates

| Gate | Condition | Threshold |
|------|-----------|-----------|
| A | robust_d_missed >= pass_robust_d | >= 0.30 |
| B | >= min_pass_machines pass Gate A | >= 2 |
| C | max_redundancy_r <= threshold | <= 0.85 |
| D | robust_d_detected > 0 | > 0 |

### Normal-tail quantiles

q_low = 0.95, q_high = 0.99

### 输出

```
results/v46_cepstral_frontend_prescreen/
  summary.csv
  per_machine.csv
  redundancy.csv

docs/CTAF_V46_CEPSTRAL_FRONTEND_PRESCREEN.md
```

### 报告必须包含

1. cepstral NaN 修复说明（修复了什么、怎么修的）
2. 修复后是否仍存在 NaN
3. input-level separation 表（per machine, per frontend）
4. robust_d_missed_vs_tail
5. robust_d_detected_vs_tail
6. redundancy / Pearson correlation（每个新 view 与 5 baseline views 的 max |r|）
7. 是否通过 Stage-0
8. **如果失败**，必须写明：_"Cepstral frontend does not provide enough input-level separation under the fixed gate. Recorded as a scientific negative, not an implementation failure."_

### 实施参考

可直接复用 `scripts/diagnose_v39_frontend_prescreen.py` 的架构：
- `InputNormalizer`：train-set per-view mean/std
- `compute_robust_d()`：IQR 分母 Cohen's d
- 冗余度：Pearson r vs 5 baseline views (logmel, delta, delta-delta, edge, enhanced)
- 四门系统
- 报告生成模式

需要修复的 cepstral 问题：
- `_upsample_freq()` 使用 `scipy.interpolate.interp1d`，当 MFCC 某些帧全为零时产生 NaN
- 修复方向：NaN-safe interpolation（fill_value 设为边界值而非 NaN）；或在 upsampling 前对全零帧做小量填充

---

## 6. Explicitly Forbidden

Codex **绝对不能**做以下任何操作：

- 不开 V47（不主动开启新版本）
- 不训练任何模型（no 5-epoch, no 50-epoch, no training of any kind before V46 result is available）
- 不调 quefrency range 来追结果
- 不调 cepstral 前端参数
- 不调 k / top_percent
- 不做 score fusion / frontend fusion
- 不做 per-machine selection
- 不把 transient_3view 结果包装成 mainline replacement
- 不继续 full_5view segment scoring
- 不自动提出新方向

---

## 7. Files Codex Should Read First

### V39–V45 必读文档

```
docs/CTAF_V39_FRONTEND_PRESCREEN.md
docs/CTAF_V40_TRANSIENT_FRONTEND_AUDIT.md
docs/CTAF_V41_TRANSIENT_SIGNAL_PRESERVATION_DIAGNOSTIC.md
docs/CTAF_V42_TRANSIENT_SEGMENT_SCORING_AUDIT.md
docs/CTAF_V43_TRANSIENT_SEGMENT_FULL_MACHINE_AUDIT.md
docs/CTAF_V44_TRANSIENT_SEGMENT_SEED_PANEL.md
docs/CTAF_V45_SEGMENT_SCORING_TRANSFER_AUDIT.md
```

### 可选背景文档

```
docs/CTAF_V37_TEACHER_STUDENT_PRETRAIN_SEED_PANEL.md
docs/CTAF_V38_DENSITY_SCORING_AUDIT.md
docs/CTAF_LOW_FPR_FAILURE_ATTRIBUTION.md
EXPERIMENT_INDEX.md
```

### 关键代码文件

```
src/run_experiment.py                          # 主实验 runner（已支持 transient_3view）
src/models/conv_ae.py                          # 模型（含 encode_staged()）
src/features/tensor_builder.py                 # 张量构建（含 transient_3view）
src/features/logmel.py                         # logmel + compute_mfcc_from_logmel()
src/evaluation/scorer.py                       # 评分函数（含 local_tail_knn）
src/evaluation/metrics.py                      # 指标计算（compute_metrics）
src/data/dataset.py                            # 数据扫描与切分
scripts/diagnose_v39_frontend_prescreen.py     # V39 Stage-0 prescreen（可复用）
scripts/diagnose_v41_signal_preservation.py    # 信号保存诊断（可参考 staged encoding）
```

### 现有结果目录

```
results/v37_teacher_student_pretrain_seedpanel/seed_13711/   # V37 分数 NPZ（参考源）
results/v40_transient_frontend_seed13711/                     # V40 训练结果 + 模型
results/v44_transient_segment_seedpanel/                      # V44 3-seed panel
results/v45_segment_scoring_transfer_audit/                   # V45 迁移审计
```

---

## 8. Final Handoff Sentence

**Codex should now implement/fix only V46 cepstral frontend Stage-0 prescreen under the fixed gate. Do not start training or new research directions until V46 prescreen result is available.**
