# CTAF Stage 7: Full-5view Local Optimum Summary (V39–V50)

CTAF 项目 V39–V50 对 full_5view + AE + kNN 框架进行了系统性探索。所有方向均指向同一结论：当前配置已经是经验局部最优，继续在这个框架内小修小改不会突破 low-FPR 瓶颈。

---

## 1. Stage 5 — Mechanism 3-View Frontends (V39–V47)

**测试内容**：三个候选 3-view mechanism 前端 (modulation, transient, cepstral)，从 V39 Stage-0 input-level prescreen 到 V40/V47 完整训练验证。

| 前端 | Stage-0 rd | 冗余度 | 训练 hmean vs logmel | 训练判定 |
|------|-----------|--------|---------------------|---------|
| modulation_3view | 3.64 | 0.969 | 未进入训练 | Stage-0 redundancy fail |
| transient_3view | 0.74 | 0.054 | 0.544 vs 0.539 (+0.005) | pAUC/TPR@99n fail |
| cepstral_3view | 2.93 | 0.609 | 0.507 vs 0.544 (−0.037) | 全指标 fail |

**补充诊断（V41–V44）**：
- V41：transient_3view 的信号未在 encoder 中衰减，反而被放大（|rd| 0.99→1.77）
- V42–V44：segment-level scoring 能补偿 transient 的弱前端性质，3-seed panel 稳定
- V45：segment scoring 不能迁移到 full_5view —— 它是弱前端的补偿，不是通用方案

**结论**：**input-level separability alone is insufficient.** 两个前端（transient, cepstral）在 Stage-0 有强异常分离信号，训练后均无法转化为稳定的 low-FPR tail separation。三个单独机制前端均不能替代 full_5view。

---

## 2. Stage 6 — Hard-Normal Tail (V48–V49)

**测试内容**：基于 V37 full_5view teacher-student representation 的 hard-normal consistency training。

**V48 离线诊断**：
- hard-normal（V37 ls_knn 分数 top 10% train-normal）占据 normal q95–q99 尾带（toytrain 100%, gearbox 92%, bearing 83%）
- augmentation consistency：bearing/fan/valve 的 hard-normal 对 time/freq mask 的 cosine 敏感度是 easy-normal 的 3–10 倍
- missed anomaly 与 hard-normal 的 latent space 重叠很低（1–9%）——hard-normal 的作用是压低 q99 而非直接与 anomaly 混淆

**V49 训练验证**（固定 V37 hard-normal index + time/freq mask + cosine consistency, λ=0.10）：

| 标准 | 阈值 | 实际 | 判定 |
|------|------|------|------|
| mean hmean ≥ V37 − 0.003 | 0.5729 | 0.5496 | FAIL |
| mean pAUC ≥ V37 | 0.5505 | 0.5473 | FAIL |
| q99 not inflated | — | 6/7 lower | PASS |
| ≥2 of bearing/fan/valve TPR improve | 2 | 1 (fan) | FAIL |
| toycar/toytrain not degraded | — | toycar −0.067 | FAIL |

**结论**：
- hard-normal consistency 成功压缩 normal tail（q99 6/7 下降），机制验证为正
- 但无法转化为稳定 TPR@99n 提升：gains are machine-specific and unstable
- toycar 在 V48 诊断为 hard-normal-negative，V49 证实强加 loss 会显著退化（−0.067 hmean）
- **normal-tail compression alone is insufficient**

---

## 3. V50 — Patch Spatial Summary

**测试内容**：在 V37 latent_map 上用 2×2 patch pooling 替代 global pooling，测试粗空间结构是否保留 tail 信号。No-training audit。

| 指标 | global summary | patch_2x2 | Δ |
|------|---------------|-----------|-----|
| mean hmean | **0.5539** | 0.5136 | −0.040 |
| mean pAUC | **0.5457** | 0.5157 | −0.030 |
| bottleneck TPR@99n improved | — | 3/5 | marginal |
| bearing TPR@99n | 0.213 | 0.040 | −0.173 |
| valve TPR@99n | 0.070 | 0.000 | −0.070 |

**结论**：patch_2x2 **negative**。patch_2x2 的 score scale 明显变大，但相对 tail-width 没有明显 inflation（V50 q99 inflation check PASS）；真正失败原因不是单纯 q99 膨胀，而是整体 hmean/pAUC 显著下降和 bearing/valve collapse。

---

## 4. Final Conclusion

### 核心陈述

**`full_5view + global latent_summary_knn` 是当前 AE + kNN 框架下的经验局部最优。**

它的优势来自两个相互强化的属性：

1. **多视图互补**：logmel, delta, delta-delta, edge, enhanced 五个视图提供频域、时域、结构的三层信息。V45 证明 segment scoring 不能迁移到 full_5view 正是因为去掉了视图间交互。

2. **全局统计池化**：channel-wise mean + std + max 将整个 latent map 的信息压缩到 192-dim 向量，这种全局聚合在 full_5view 上意外地稳健——V50 证明任何空间割裂都会破坏这种稳健性。

### 已测试方向及其结局

| 方向 | 版本 | 结果 | 根因 |
|------|------|------|------|
| 换 mechanism 前端 | V39–V47 | 全负 | input signal ≠ training success |
| 段级评分替代池化 | V42–V45 | 部分正但不迁移 | 适合弱前端，破坏 full_5view 多视图 |
| hard-normal 训练 | V48–V49 | 机制正、训练负 | 压缩 tail 但效果不跨机器 |
| 空间 patch 替代池化 | V50 | 负 | 割裂空间结构破坏全局稳健性 |

### 为什么不该继续在这个框架内搜索

四个方向都显现同一个模式：**能帮助一部分机器（通常是 weak bottleneck 如 gearbox/slider/toytrain），但同时伤害另一部分（通常是 rich source machines 如 bearing/valve）。** 这不是参数可调的 tradeoff——这是当前 full_5view + AE + kNN 协议下表现出的 structural limitation / empirical constraint：

- AE 重建损失不区分难易样本
- kNN scoring 在紧邻 low-FPR 区域缺乏分辨率
- 任何只改前端的操作只能 relabel 异常信号，不能创造新的判断能力

---

## 5. Blocked Actions

- Do NOT open V51
- Do NOT tune patch grid size (3×3, 4×4, etc.)
- Do NOT attempt score fusion between global and patch summary
- Do NOT attempt per-machine pooling selection
- Do NOT continue hard-normal lambda/top_ratio/mask_frac search
- Do NOT continue 3-view mechanism frontend search
- Do NOT auto-proceed to new experiment version

---

## 6. Future Direction

如果需要在 low-FPR tail separation 上继续取得突破，必须提出新的框架级假设，而不是继续在当前 AE + kNN 框架内部做局部修改。可能的新框架方向包括但不限于：

- **替代训练目标**：contrastive learning、normalizing flow、energy-based model，直接建模正常分布的密度边界而非重建
- **替代评分机制**：learned anomaly scoring head、one-class classification、deep SVDD，取代 kNN 统计
- **替代特征表示**：预训练大模型表示 (e.g., CLAP/HuBERT/WavLM fine-tuned on ASD)，用外部知识补充局部 acoustic pattern

以上方向均超出了 Stage 5–7 的 mandate，需在新的预注册假设下启动独立实验线。

以上不是立即执行任务；若开启新方向，必须先形成新的预注册假设。

---

CTAF V39–V50 实验线结束。现有证据支持：在当前 full_5view + AE + kNN 协议下，global latent_summary_knn 是经验局部最优。后续若继续突破 low-FPR tail separation，应另起新的框架级预注册假设，而不是继续在当前管线内做局部调参。
