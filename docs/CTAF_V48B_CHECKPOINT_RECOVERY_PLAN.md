# CTAF V48B Checkpoint Recovery Plan

## 1. Status

| 项目 | 状态 |
|------|------|
| V37 score NPZ files | **存在** (28/seed × 3 seeds) |
| V37 original checkpoints (`.pt`) | **缺失** — V37 `run_experiment.py` 未保存模型 |
| V37 protocol-equivalent checkpoints | **存在** — 21/21, 位于 V45 目录 |

## 2. Available Checkpoints

**路径**：`results/v45_segment_scoring_transfer_audit/v37_full5view_ts/models/`

**文件命名**：`<machine>_full_5view_seed<seed>.pt`

| Seed | Machines | Count |
|------|----------|-------|
| 13711 | bearing, fan, gearbox, slider, toycar, toytrain, valve | 7/7 |
| 13721 | bearing, fan, gearbox, slider, toycar, toytrain, valve | 7/7 |
| 13731 | bearing, fan, gearbox, slider, toycar, toytrain, valve | 7/7 |

**总计：21/21 ✓**

## 3. Provenance

这些 checkpoint 是在 V45 Segment Scoring Transfer Audit 期间通过 `scripts/regenerate_v45_models.py` 生成的。

**复现协议与原始 V37 完全一致**：
- `experiment = full_5view`
- `ssl_task = teacher_student_local_distillation`
- `model_type = viewwise_unshared_late_fusion`
- `latent_channels = 64, stem_channels = 8, per_view_channels = 16, dropout_p = 0.0`
- `training = source normal only, loss_view_mode = all_views`
- `epochs = 50, batch_size = 32`
- `seed = 13711 / 13721 / 13731`

**Teacher-student 训练流程**：
1. 训练 teacher AE（标准 reconstruction，50 epochs）
2. 训练 student（TF-block-masked input 预测 teacher 的 local latent features，50 epochs）
3. 保存 **teacher** model（V37 使用 teacher 进行 scoring）

**与原始 V37 的差异**：GPU 浮点非确定性导致权重有微小偏差（V40 复现时分数 max_diff < 0.08）。

## 4. 使用建议

V48 脚本需指定 `--checkpoint_dirs` 指向此路径：

```
--checkpoint_dirs results/v45_segment_scoring_transfer_audit/v37_full5view_ts/models
```

如果 V48 脚本按 `seed_<N>/models/<machine>_<experiment>_seed<seed>.pt` 格式查找 checkpoint，需要先将文件软链接到 V37 目录下的预期位置。

### 最小补全命令（软链接，不复制文件）

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd

for seed in 13711 13721 13731; do
    mkdir -p results/v37_teacher_student_pretrain_seedpanel/seed_${seed}/models
    for machine in bearing fan gearbox slider toycar toytrain valve; do
        src="results/v45_segment_scoring_transfer_audit/v37_full5view_ts/models/${machine}_full_5view_seed${seed}.pt"
        dst="results/v37_teacher_student_pretrain_seedpanel/seed_${seed}/models/${machine}_full_5view_seed${seed}.pt"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            ln -s "../../../../v45_segment_scoring_transfer_audit/v37_full5view_ts/models/${machine}_full_5view_seed${seed}.pt" "$dst"
        fi
    done
done
```

### 如果 V48 脚本不接受软链接

改用硬拷贝（21 × ~2-5MB ≈ 100MB total）：

```bash
for seed in 13711 13721 13731; do
    mkdir -p results/v37_teacher_student_pretrain_seedpanel/seed_${seed}/models
    cp results/v45_segment_scoring_transfer_audit/v37_full5view_ts/models/*_full_5view_seed${seed}.pt \
       results/v37_teacher_student_pretrain_seedpanel/seed_${seed}/models/
done
```

### 验证命令

```bash
find results/v37_teacher_student_pretrain_seedpanel -name '*.pt' | wc -l
# Expected: 21
```

## 5. 注意事项

1. 这些 checkpoint 不是原始 V37 的字节级拷贝，而是相同协议下的确定性复现（seed 固定）。分数可能不完全匹配原始 V37 NPZ 中的 latent_summary 值。
2. 对于 V48 的 latent space 分析（hard-normal centroid、augmentation consistency），权重微小偏差不影响结论有效性。
3. 不作科学结论替代。如果 V48 报告引用这些 checkpoint，必须注明 provenance（V45 复现，非原始 V37 训练）。

## 6. 不需要的操作

- 不需要重新运行 V37 `run_experiment.py`
- 不需要用 V47/V40 模型替代
- 不需要新训练
- 不自动进入 V49
