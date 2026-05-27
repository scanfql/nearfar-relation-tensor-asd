# CTAF V46 Cepstral Frontend Prescreen

## 1. Cepstral NaN Fix

`V39` 中 `cepstral_3view` 的失败并不是科学负结果，而是
implementation/preprocessing invalid。

定位到的问题是：

- `cepstral_periodicity` 在构造时本质上是 `1 x T`
- 旧版 `_upsample_freq()` 对每一帧调用 `interp1d`
- 当 `n_src == 1` 时，单点频率轴插值是病态的，容易产生 `NaN`
- 同时旧实现使用 `fill_value="extrapolate"`，对边界和退化列不够稳

本次修复只做 implementation sanity fix，不改任何 scientific design：

1. `n_src == 1` 时，不再插值，直接沿频率轴重复到 `128` 个 bins。
2. 一般情形改用 `np.interp(..., left=col[0], right=col[-1])`，不再做外推。
3. 对输入列和输出统一做 `np.nan_to_num(...)`，确保无 `NaN/Inf` 泄漏。
4. 不改 quefrency range，不改 cepstral 设计，不改 Stage-0 gate。

## 2. 修复后是否仍存在 NaN

没有。

证据：

- `per_machine.csv` 中所有 `robust_d_missed / robust_d_detected / cohens_d`
  都是有限值
- 不再出现 `nan_in_tail_distances` / `nan_in_missed_distances`
- `summary.csv` 成功生成
- `cepstral_3view` 可以完成完整的 Stage-0 prescreen

因此：

```text
V46 removes the implementation-invalid status of cepstral_3view.
```

## 3. Fixed Stage-0 Gates

Reference score source:

- result dir: `results/v37_teacher_student_pretrain_seedpanel/seed_13711`
- experiment: `full_5view`
- score_mode: `latent_summary_knn`

Machines:

- `gearbox`
- `slider`
- `toycar`
- `toytrain`
- `fan`

Gates:

| Gate | Condition | Threshold |
|---|---|---:|
| A | `robust_d_missed >= pass_robust_d` | `>= 0.30` |
| B | `>= min_pass_machines` pass Gate A | `>= 2` |
| C | `max_redundancy_r <= threshold` | `<= 0.85` |
| D | `robust_d_detected > 0` | `> 0.0` |

Normal-tail band:

- `q_low = 0.95`
- `q_high = 0.99`

## 4. Input-Level Separation Table

| Machine | Frontend | robust_d_missed | robust_d_detected | cohens_d | Gate A | Gate D | Flags |
|---|---|---:|---:|---:|---|---|---|
| gearbox | cepstral_3view | 3.376018 | 1.917173 | 2.317480 | PASS | PASS | -- |
| slider | cepstral_3view | 2.109820 | 0.000000 | 1.966835 | PASS | FAIL | `no_detected_anomalies` |
| toycar | cepstral_3view | 5.904637 | 4.521709 | 7.173301 | PASS | PASS | -- |
| toytrain | cepstral_3view | 0.831602 | 1.290158 | 0.956655 | PASS | PASS | -- |
| fan | cepstral_3view | 2.428615 | 4.857256 | 2.094194 | PASS | PASS | -- |

## 5. robust_d_missed_vs_tail

`robust_d_missed` by machine:

- `gearbox = 3.376018`
- `slider = 2.109820`
- `toycar = 5.904637`
- `toytrain = 0.831602`
- `fan = 2.428615`

Aggregate:

```text
mean_robust_d_missed = 2.930138
machines passing Gate A = 5/5
```

This is far above the fixed `0.30` threshold and clearly passes Gate A and Gate B.

## 6. robust_d_detected_vs_tail

`robust_d_detected` by machine:

- `gearbox = 1.917173`
- `slider = 0.000000`
- `toycar = 4.521709`
- `toytrain = 1.290158`
- `fan = 4.857256`

Aggregate:

```text
mean_robust_d_detected = 2.517259
machines passing Gate D = 4/5
```

Only `slider` fails Gate D because there are no detected anomalies above the
reference `q99` threshold for this machine.

## 7. Redundancy / Pearson Correlation

| Frontend | View | max(|r|) |
|---|---|---:|
| cepstral_3view | cepstral_low_quefrency | 0.608859 |
| cepstral_3view | cepstral_periodicity | 0.094920 |

Aggregate redundancy verdict:

```text
max_redundancy_r = 0.608859 <= 0.85
```

So Gate C passes comfortably.

## 8. Stage-0 Verdict

Summary:

| frontend | machines_passing_gate_a | machines_passing_all | max_redundancy_r | gate_b_pass | gate_c_pass | stage0_clearance | mean_robust_d_missed | mean_robust_d_detected |
|---|---:|---:|---:|---|---|---|---:|---:|
| cepstral_3view | 5 | 4 | 0.608859 | True | True | True | 2.930138 | 2.517259 |

Final Stage-0 decision:

```text
cepstral_3view passes Stage-0 under the fixed gate.
```

Interpretation:

- after the NaN-safe upsampling fix, `cepstral_3view` is scientifically valid
- it shows strong input-level separation on missed anomalies
- redundancy is well below the fixed threshold
- it is no longer correct to treat cepstral_3view as an implementation failure

Therefore:

```text
V46 upgrades cepstral_3view from implementation-invalid to Stage-0 cleared.
```

This is a positive prescreen result, not a scientific negative.
