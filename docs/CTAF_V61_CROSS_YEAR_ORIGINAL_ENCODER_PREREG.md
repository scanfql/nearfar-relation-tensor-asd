# CTAF V61 Cross-Year Original Encoder Prereg

## 1. Motivation

V60 and V60B completed the cross-year data schema audit and manifest lock for a first clean multi-year normal-only corpus.
The selected corpus already passed full_5view tensor feasibility checks on all included files.

V61 is therefore the first Stage 13A-1 question:

`Does expanding normal-only data support alone improve CTAF?`

This step is intentionally narrow.
It does **not** test:

- stronger encoder capacity
- new training objective
- UACS or other backend changes
- BEATs or any external pretrained representation
- pseudo-anomaly learning
- score fusion

V61 only tests whether cross-year normal-only data support itself is useful under the original CTAF/V37 architecture and scoring rule.

## 2. Hypothesis

The current V37 CTAF encoder may be limited by single-year normal support.
Using multi-year normal-only DCASE data for pretraining/adaptation may improve normal manifold coverage and low-FPR tail separation while preserving the same full_5view frontend, encoder architecture, latent_summary pooling, and kNN scoring.

中文解释：

如果 V37 的主要瓶颈来自 normal support 不足，那么在不改变模型结构的前提下，引入多年份 normal-only 数据，应该有机会改善 `normal q99`、`pAUC` 和 bottleneck machines 的 `TPR@99n`。

## 3. Fixed Data Protocol

V61 must use the locked manifest:

`results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv`

Included:

- 2023 train normal only
- 2024 train normal only
- 2025 current/dev train normal only

Excluded:

- all test split files
- all anomaly clips
- all 2025 `eval_data_*` branches
- all 2026 `Emu` / noise-aware / new-setting taxonomy files
- ambiguous files

This is an external normal-data pretraining protocol for paper research.
It is **not** a DCASE official competition submission protocol.

V61 must not use:

- anomaly labels
- test labels
- weak supervision
- semi-supervised anomaly training

## 4. Model Protocol

V61 must preserve the original V37 encoder line:

- frontend = `full_5view`
- encoder = `viewwise_unshared_late_fusion`
- latent_channels = `64`
- stem_channels = `8`
- per_view_channels = `16`
- dropout_p = `0.0`
- representation = global latent_summary = `mean + std + max`
- scoring = `latent_summary_knn`, `k=5`

Not allowed:

- Transformer
- Conformer
- larger encoder
- BEATs
- changed pooling
- changed scoring

## 5. Training Protocol Draft

V61 draft training logic is fixed conceptually as:

Phase 1:
- cross-year normal-only teacher AE pretraining on `manifest_v61_clean`

Phase 2:
- current target-year / source-train adaptation using the same V37 teacher-student protocol family

Evaluation:
- same current evaluation set as the V37 baseline
- same `latent_summary_knn` scoring rule

Important constraint:

If the implementation stage reveals that the original V37 pipeline cannot cleanly support this `pretrain -> adapt` structure, the process must stop and report the incompatibility.
It must **not** silently change into a different training logic.

## 6. Baseline

Primary baseline:

- `V37 full_5view latent_summary_knn current-year-only`

Reference single-seed values:

- `seed13711 hmean = 0.5631`
- `seed13711 pAUC = 0.5505`

V61 must be compared directly against this baseline.

## 7. Success Gates

V61 is positive only if all conditions hold:

1. mean `hmean >= V37 baseline`
2. mean `pAUC >= V37 baseline`
3. at least 2 bottleneck machines improve `TPR@99n`
4. `bearing` / `valve` do not collapse
5. `toycar` does not clearly degrade
6. normal `q99` does not clearly inflate
7. gains are not carried by only one machine
8. improvement must not come from accidental inclusion of excluded test / eval / anomaly data

## 8. Failure Criteria

V61 fails if any of the following occurs:

- `hmean` below V37
- `pAUC` below V37
- normal `q99` inflates
- `bearing` / `valve` collapse
- `toycar` degrades
- only one machine improves
- training requires including 2025 `eval_data_*`
- training requires including 2026 `Emu` / noise-aware data
- improvement depends on changing encoder / loss / scoring
- any data leakage is detected

## 9. Difference from Future Stages

Stage 13A-1 / V61:

- tests data support only

Stage 13B / V62:

- only after V61
- tests stronger encoder

Stage 13C / V63:

- only if A/B remain insufficient
- tests new training objective

## 10. Forbidden Actions

At this preregistration step, the following are forbidden:

- do not train any model
- do not write V61 training code
- do not open V62
- do not change `full_5view`
- do not change the encoder
- do not change `latent_summary`
- do not change `kNN`
- do not introduce BEATs
- do not introduce UACS
- do not use anomaly labels
- do not use test labels
- do not include 2025 `eval_data_*`
- do not include 2026 `Emu` / noise-aware data
- do not do score fusion
- do not do per-machine selection
- do not auto-open V62 or V63

## 11. Human Decisions Required Before Implementation

Before implementation, the following points must be explicitly confirmed:

1. whether V61 should use the `pretrain -> adapt` two-phase schedule
2. which year is the current target year
3. whether `manifest_v61_clean` is the only external normal corpus
4. whether 2026 is strictly excluded
5. whether 2025 `eval_data_*` is strictly excluded
6. whether the original V37 training script can support a cross-year manifest
7. the exact single-seed pilot command
8. whether to run a one-machine dry run first

## Final Sentence

V61 tests only the effect of cross-year normal-only data support under the original V37 CTAF architecture. It must not be treated as a stronger-encoder, new-loss, UACS, BEATs, or score-fusion experiment. No implementation or training is authorized until this preregistration is reviewed and approved.
