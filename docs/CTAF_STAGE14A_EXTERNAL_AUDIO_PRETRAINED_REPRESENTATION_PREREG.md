# CTAF Stage 14A External Audio Pretrained Representation Prereg

## 1. Motivation

Stage 13 established a clear picture:

- cross-year normal-only data support helps
- stronger internal encoder capacity alone does not solve the remaining problem
- fixed V37/V61 rank fusion is the current best frozen CTAF route

At the same time, earlier evidence from `V55` showed that frozen external audio
pretrained representation can produce the strongest single-seed `hmean` signal
seen so far.

Specifically, `V55 frozen BEATs + kNN` produced:

- strong `hmean`
- but weaker `pAUC`
- and unstable low-FPR tail behavior (`q99` inflation / low-FPR instability)

This means external audio pretrained knowledge appears helpful for global
anomaly ranking, even though it does not automatically solve low-FPR tail
separation.

Therefore, if the next research priority is to improve `hmean`, then:

`external audio pretrained representation + minimal adaptation`

is a reasonable new branch.

Important clarification:

Stage 14A is a **hmean-oriented external representation branch**.
It is **not** the CTAF `full_5view` frontend mainline.
It must **not** be interpreted as proof that the CTAF frontend itself is
stronger.
It also does **not** claim to naturally solve low-FPR tail separation.

## 2. Relationship to CTAF Mainline

Stage 14A defines a dual-line structure.

CTAF mainline:

- `full_5view`
- `V61` cross-year normal-only support
- `V63` fixed V37/V61 rank fusion
- current best frozen CTAF route

External representation branch:

- BEATs / external audio pretrained representation
- minimal adaptation only if separately justified
- `hmean`-oriented
- used as a strong external branch / comparison / possible upper route

Important rule:

`Do not merge Stage 14A into CTAF frontend contribution unless separately justified.`

## 3. Prior Evidence

`V55 frozen BEATs + kNN`:

- `hmean = 0.578557`
- `pAUC = 0.538797`
- `KS = 0.508571`
- `hmean > V37 / V61 / V63`
- `pAUC` lower than the best CTAF routes
- `q99` inflation / low-FPR instability observed
- no fine-tuning occurred

Interpretation:

`Frozen BEATs has strong global ranking signal but weak low-FPR behavior.`

## 4. Research Question

Core question:

Can external audio pretrained representations improve `hmean` beyond CTAF V63,
and can minimal normal-only adaptation preserve or improve `pAUC` without
causing `q99` / low-FPR collapse?

中文解释：

外部音频预训练知识是否能进一步提高整体异常排序能力；如果 frozen 表示已经能提高 `hmean`，那么最小适配是否能在不牺牲 `pAUC` 的情况下进一步稳定它？

## 5. Candidate Model Scope

The first Stage 14A version allows only an already existing BEATs checkpoint.

Primary candidate:

- official BEATs AudioSet checkpoint already used in `V55`
- checkpoint path from `V55`:
  `/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt`

Forbidden:

- do not download new models
- do not switch to CLAP / AST / EAT
- do not search across multiple pretrained models
- do not introduce LLM or multimodal models
- do not compare several foundation models in the same stage

## 6. Candidate Stages

Stage 14A defines a strict progression.

### A0: Reproduce / verify V55 frozen BEATs baseline

If needed, first verify frozen BEATs under the same 3-seed evaluation protocol.

### A1: Frozen BEATs + fixed rank fusion with CTAF V63

No training.

Candidate examples:

- `C0 = V63 fixed CTAF route`
- `C1 = frozen BEATs score`
- `C2 = fixed 0.5/0.5 rank_norm(V63, BEATs)`

No weight tuning allowed.

### A2: Minimal adaptation only if A1 is positive or clearly promising

Allowed minimal adaptation candidates:

- linear projection head on frozen BEATs embeddings
- shallow MLP projection head
- LoRA / adapter only if separately approved later

The first Stage 14A version must **not** jump directly to full fine-tuning.

## 7. Fixed Rules for Frozen / Fusion Stage

For `A1`:

- no training
- no weight tuning
- no per-machine selection
- no oracle selection
- rank_norm per machine / per seed
- fixed fusion weight only
- no anomaly labels
- no test labels for selection

## 8. Minimal Adaptation Constraints

If `A2` is later approved, adaptation must be:

- normal-only
- no anomaly labels
- no test labels
- no per-machine tuning
- no full BEATs fine-tuning in the first version
- no architecture sweep
- no hidden weight search

Allowed first adaptation direction:

- frozen BEATs embedding
- train a small projection head using train-normal only
- scoring still kNN-style or fixed normal-boundary score
- compare against frozen BEATs and V63

## 9. Baselines

Stage 14A must compare against:

Primary CTAF frozen route:

- `V63 fixed V37/V61 rank fusion`
- `hmean = 0.562753`
- `pAUC = 0.549762`
- `TPR@99n = 0.089524`

External frozen baseline:

- `V55 frozen BEATs + kNN`
- `hmean = 0.578557`
- `pAUC = 0.538797`
- `KS = 0.508571`

Also include:

- `V37`
- `V61`

## 10. Success Criteria

Stage 14A is positive only if:

1. `hmean > V63` by a meaningful margin
2. `pAUC` does not fall far below `V63`
3. improvement is not carried by only one machine
4. `bearing / valve` do not collapse
5. `toycar` does not severely degrade
6. no anomaly labels are used
7. no test-label-based selection is used
8. no hidden model or weight search occurs

Suggested first threshold:

- `hmean >= V63 + 0.01`
- `pAUC >= V63 - 0.01`

If `pAUC` drops by more than `0.01`, report the route as:

- `hmean-positive`
- but `pAUC-negative`

and do **not** treat it as full success.

## 11. Failure Criteria

Stage 14A fails if:

- `hmean` does not exceed `V63`
- `pAUC` collapses
- gains are driven by one machine only
- BEATs hurts `bearing / fan / toycar` severely
- low-FPR tail becomes worse without `hmean` gain
- implementation requires full fine-tuning
- model selection uses anomaly/test labels
- the external model or checkpoint changes without preregistration

## 12. Blocked Actions

At this preregistration step, the following are forbidden:

- do not train any model
- do not write adaptation code
- do not download new checkpoints
- do not switch external model
- do not tune fusion weights
- do not do per-machine selection
- do not use anomaly labels
- do not use test labels
- do not open LoRA / adapter / fine-tuning
- do not describe Stage 14A as CTAF frontend improvement
- do not auto-open Stage 14B

## 13. Human Decisions Required Before Implementation

Before implementation, the following must be explicitly confirmed:

1. whether to first reproduce / verify the V55 frozen BEATs result
2. whether to first run a V55 3-seed panel
3. whether V63 + BEATs fixed rank fusion is allowed
4. the exact fixed fusion formula
5. whether to proceed to minimal adaptation
6. adaptation type: linear head / MLP head / LoRA / adapter
7. whether to use frozen embedding cache
8. the exact command
9. whether to first run a one-machine dry run
10. whether to keep V63 as the CTAF frozen mainline

## Final Statement

Stage 14A is a `hmean`-oriented external audio pretrained representation
branch. It must not be treated as CTAF frontend improvement, and it must not
use hidden weight tuning, per-machine selection, anomaly labels, or
test-label-based model selection. No implementation or training is authorized
until this preregistration is reviewed and approved.
