# CTAF V62 Cross-Year Strong Encoder Prereg

## 1. Motivation

V61 established that expanding cross-year normal-only data support can improve
global anomaly ranking under the original CTAF setup.

Observed V61 3-seed outcomes:

- `V37 hmean = 0.5539 ± 0.0065`
- `V61 hmean = 0.5629 ± 0.0079`
- `Δ hmean = +0.0090`
- `V37 pAUC = 0.5457 ± 0.0036`
- `V61 pAUC = 0.5493 ± 0.0046`
- `Δ pAUC = +0.0036`
- `q99` compressed on `5/7` machines
- no data leakage
- no protocol changes

However, V61 did not solve low-FPR tail separation:

- `TPR@99n` remained nearly flat overall
- `V37 TPR@99n = 0.0800`
- `V61 TPR@99n = 0.0786`
- only `1/5` bottleneck machines improved
- `toycar` remained a consistent failure case

Therefore, the V62 question is no longer whether cross-year data support is
useful. V61 already showed that it is useful for global ranking.

The V62 question is:

`Under stronger cross-year normal support, is the original small CNN encoder still the main remaining bottleneck?`

## 2. Hypothesis

The `full_5view` frontend contains useful multi-view acoustic structure, and
cross-year normal-only pretraining improves global representation robustness.
However, the original small CNN encoder may lack capacity to model fine-grained
within-machine tail structure. A stronger full_5view Transformer/Conformer-style
encoder may better exploit the cross-year normal corpus and improve low-FPR
tail separation while preserving the CTAF frontend and kNN scoring protocol.

中文解释：

`full_5view` 前端继续保留；V62 只测试 encoder capacity 是否是剩余瓶颈。目标不是再换前端，也不是再换 scoring，而是在 cross-year normal-only 数据已经证明有效的前提下，用更强 encoder 捕捉同一机器内部更细粒度的时频结构。

## 3. Fixed Data Protocol

V62 must reuse the exact V61 data protocol:

`results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv`

Included:

- 2023 train normal
- 2024 train normal
- 2025 current/dev train normal

Excluded:

- all test split files
- all anomaly clips
- all 2025 `eval_data_*` branches
- all 2026 `Emu` / noise-aware branches
- ambiguous files

Not allowed:

- redefining the manifest
- adding 2026
- adding 2025 `eval_data_*`
- using anomaly labels
- using test labels

## 4. Fixed Frontend and Scoring

V62 must preserve:

- frontend = `full_5view`
  - `[logmel, delta, delta-delta, edge, enhanced]`
- representation output must still produce a global file-level embedding
  compatible with latent-summary-style scoring
- scoring = `latent_summary_knn`
- `k = 5`
- source train-normal memory only for evaluation
- no score fusion
- no UACS
- no per-machine selection

## 5. What V62 Is Allowed to Change

V62 is only allowed to change encoder capacity.

Allowed change:

- replace the original V37 `viewwise_unshared_late_fusion` CNN encoder with one
  stronger full_5view encoder

Candidate family:

- full_5view Transformer encoder
  or
- full_5view Conformer encoder

The stronger encoder must still take the full_5view tensor as input.

Recommended first version:

- full_5view Conformer-style encoder

Reason:

- CNN/local convolution can preserve local time-frequency structure
- self-attention can model longer-range temporal-frequency dependencies
- Conformer is more suitable than a pure Transformer for spectrogram-like
  acoustic representation under the CTAF setting

## 6. Model Size Constraint

V62 first version must remain controlled:

- target parameter range: `1M–5M`
- not a `50M/100M` foundation model
- no BEATs
- no AST
- no EAT
- no external pretrained weights

Reason:

V62 tests encoder capacity inside CTAF, not external pretrained
representation.

## 7. Suggested Encoder Design Draft

This section is only a preregistration draft and does not authorize
implementation.

Input:

- full_5view tensor, approximately `[5, 128, T]`

Architecture draft:

- per-view lightweight CNN stem
- late fusion across 5 views
- patch/token projection
- 4-layer Conformer or Transformer encoder
- hidden dim around `128` or `192`
- 4 attention heads
- local convolution module if Conformer
- output latent map or latent tokens
- global summary = `mean + std + max` or equivalent fixed global summary
- final embedding dimension must be documented before implementation

Important:

- pooling must be fixed before implementation
- no patch/segment scoring
- no learned attention pooling unless explicitly pre-registered
- no architecture sweep
- no changing dimensions after seeing results

## 8. Training Protocol

V62 must reuse the V61 training logic:

Phase 1:

- cross-year normal-only teacher AE pretraining on `manifest_v61_clean`

Phase 2:

- current-year / source-train teacher adaptation
- current-year / source-train student distillation

Evaluation:

- same current evaluation set as V37/V61
- same `latent_summary_knn` scoring

No new loss is allowed in V62.

Forbidden training changes:

- no VICReg
- no AAM
- no pseudo-anomaly
- no contrastive objective
- no BEATs distillation
- no UACS
- no supervised anomaly labels

## 9. Baselines

V62 must compare against:

Primary baseline:

- V61 3-seed original encoder cross-year result

Reference:

- `V61 hmean = 0.5629 ± 0.0079`
- `V61 pAUC = 0.5493 ± 0.0046`
- `V61 TPR@99n = 0.0786`

Secondary baseline:

- V37 current-year-only original encoder

Reference:

- `V37 hmean = 0.5539 ± 0.0065`
- `V37 pAUC = 0.5457 ± 0.0036`
- `V37 TPR@99n = 0.0800`

## 10. Success Gates

V62 is positive only if all conditions hold:

1. mean `hmean >= V61 baseline`
2. mean `pAUC >= V61 baseline`
3. `TPR@99n` improves over V61 on at least 2 bottleneck machines
4. `toycar` does not further degrade relative to V61
5. `bearing` / `valve` do not collapse
6. normal `q99` does not systematically inflate
7. gains are not carried by only one machine
8. improvement must not depend on changing loss, scoring, manifest, or frontend
9. no data leakage

## 11. Failure Criteria

V62 fails if:

- `hmean` below V61
- `pAUC` below V61
- `TPR@99n` still does not improve on bottleneck machines
- `toycar` further degrades
- `bearing` / `valve` collapse
- `q99` inflates systemically
- the stronger encoder overfits and hurts `pAUC`
- the result depends on architecture tuning or hidden hyperparameter sweeps
- implementation requires changing loss/scoring/frontend/manifest
- any data leakage is detected

## 12. Difference from Other Stages

V61:

- changed data support only
- kept the original encoder
- showed global-ranking improvement but not low-FPR tail improvement

V62:

- keeps V61 data
- changes encoder capacity only
- tests whether model capacity is the remaining bottleneck

V63:

- only if V62 is insufficient
- may consider a new training objective
- not authorized now

## 13. Forbidden Actions

At this preregistration step, the following are forbidden:

- do not write training code
- do not train any model
- do not open a V62 run
- do not modify the manifest
- do not add 2026
- do not add 2025 `eval_data_*`
- do not use anomaly labels
- do not use test labels
- do not introduce BEATs / AST / EAT
- do not use external pretrained weights
- do not use UACS
- do not do score fusion
- do not do per-machine selection
- do not change kNN
- do not change the loss
- do not perform architecture sweeps
- do not auto-open V63

## 14. Human Decisions Required Before Implementation

Before implementation, the following points must be explicitly confirmed:

1. Transformer or Conformer
2. hidden dimension
3. number of layers
4. number of heads
5. convolution kernel size if Conformer
6. patch/token projection design
7. global summary method
8. target parameter count
9. whether to run a one-machine dry run first
10. which machine to use for dry run
11. exact single-seed pilot command
12. whether to run a parameter-count / shape sanity check first

## Final Sentence

V62 is a capacity hypothesis under the already validated cross-year normal-only CTAF protocol. It must not be treated as a new-loss, new-frontend, new-backend, BEATs, UACS, or score-fusion experiment. No implementation or training is authorized until this preregistration is reviewed and approved.
