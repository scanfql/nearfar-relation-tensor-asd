# CODEX HANDOFF STAGE21

## 1. 当前 confirmed best route

当前最强、已冻结、可作为后续一切比较基线的路线是：

- `EAT min_clamp__mean + ldn_diff_K1_scorek5`

对应 full7 核心指标：

- `mean_hmean = 0.6159467288425517`
- `mean_pAUC = 0.5603007518796993`
- `mean_TPR@99n = 0.10571428571428572`

这是当前唯一被确认同时满足以下条件的路线：

- 超过 `EAT min_clamp__mean / baseline_knn_k5`
- 超过旧 best `F_P1_mean_std_A1_065 = 0.607175`
- `improved_machine_count_vs_baseline = 5`
- `strong_machine_severe_collapse = false`

结论：当前项目主线不是新 backbone，也不是轻量训练修补，而是 **frozen clean EAT + protocol-level local-density recalibration**。

## 2. 已停止路线

以下路线已停止，不应继续作为主线推进：

- external backbone 第一轮：
  - `ATST`：one-machine 有信号，但 full7 融合失败
  - `CED`：gearbox negative
  - `HTSAT`：gearbox negative
  - `PaSST`：gearbox strong positive，但 full7 negative
- fan 单机种 LoRA / adaptation 系列
- robust density 变体
- light pooling 变体

这些路线的共同结论是：局部或单机种偶有正信号，但不能稳定超过当前 frozen 主线。

## 3. Stage19 训练线结论

Stage19 训练线整体应视为 **stopped**。

- `Stage19B/C`：纯 masked reconstruction 路线失败。训练能稳定，但 adapted 评估不过 gate。
- `Stage19E/F`：anchor-LoRA 训练健康，但 `adapted_ldn_diff_K1_scorek5` 未超过 Stage18 fan LDN 基线。
- `Stage19H/I`：anchor + pairwise geometry LoRA 仍未过 gate。已确认 `adapted_baseline_knn_k5 = 0.5778836808346788`、`adapted_ldn_diff_K1_scorek5 = 0.5501054732664112`，仍低于 Stage18 fan LDN `0.5527191284766243`。

结论：**不要继续 fan LoRA / adaptation 训练线**。训练稳定不等于 backend 结果成立。

## 4. Stage20A robust-rho 结论

`Stage20A` 已确认失败。

- best 仍然是 `ldn_diff_K1_scorek5`
- robust-rho variants 未超过当前 best

结论：不要继续 robust rho / rho 变体搜索。

## 5. Stage20B light-pooling 结论

`Stage20B` 已确认失败。

- best 仍然是 `min_clamp__mean + ldn_diff_K1_scorek5`
- `mean_std / q90 / mean_q90 / top10norm` 均未超过当前 best

结论：不要继续轻量 pooling 变体搜索。

## 6. Stage21 结论

`Stage21A/B/C/D` 说明：数据层提升是 justified 的，且已完成 prior-year official DCASE normal-only 路线冻结与真实文件审计。

- `Stage21A`：确认可安全使用官方 prior-year DCASE normal-only 数据；首选 `DCASE 2022 Additional Training`
- `Stage21B`：冻结第一条高风险训练路线：prior-year official DCASE normal-only DAP，backend 固定仍为 `min_clamp__mean + ldn_diff_K1_scorek5`
- `Stage21C`：真实文件级审计通过，`allowed_train_normal_count = 21000`，`fan_allowed_count = 3000`，`gearbox_allowed_count = 3000`，`ready_for_stage21d = true`
- `Stage21D`：global normal-only DAP 训练脚手架健康，训练/验证 loss 持续下降，best checkpoint 已产出
- `Stage21E fan`：失败。`adapted_baseline_knn_k5 = 0.5257849196744345`，`adapted_ldn_diff_K1_scorek5 = 0.5320087601167995`，均低于 Stage18 fan baseline 与 fan LDN。该路线在第一关 `fan` 就未通过，不应继续视为 promotable route。

结论：**Stage21 证明“更多 prior-year 正常数据”本身不是充分条件**。如果后续还要继续冲分，优先方向应转向训练目标层升级，而不是继续堆相似数据或继续 fan 小修补。

## 7. Forbidden actions

以下行为应视为禁止：

- 不要调 `LDN` 的 `K / r / score_k`
- 不要加 per-machine selector
- 不要加 fusion 只是为了 chase score
- 不要继续 Stage19 checkpoints
- 不要把已失败的 fan 训练线扩展到 gearbox / full7
- 不要重新大范围扫描全部 docs / results
- 不要用 test labels 做 checkpoint selection
- 不要用 anomaly labels 做训练

## 8. 新对话启动 prompt

你正在接手 `LAES_2025/new_projects/clean_tensor_asd`。当前 confirmed best route 是 `EAT min_clamp__mean + ldn_diff_K1_scorek5`，`mean_hmean = 0.6159467288425517`。Stage19 fan adaptation 线已停止，Stage20A robust-rho 和 Stage20B light-pooling 已失败，Stage21 prior-year DCASE normal-only DAP 也在 `fan` gate 失败。不要再调 LDN 参数，不要继续 fan LoRA/adaptation，不要加 selector 或 chase-score fusion，也不要重新扫描全量历史结果。请从“训练目标层升级是否能更好对齐 frozen backend 几何”这个问题出发，先做一个最小的失败分析 / contingency 设计，而不是重开旧路线。
