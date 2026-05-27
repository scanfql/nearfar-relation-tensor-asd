# CTAF Stage17B EAT LoRA-Light / Adapter-Light Prereg

## 1. Motivation

Stage 16 established three key facts:

- clean frozen `EAT` is the strongest single external route so far on `hmean`
- fixed score fusion can improve `hmean`, but no frozen fusion route clearly and stably dominates all metrics
- the route is still limited by machine-level tradeoffs, especially on `fan`, and by incomplete low-FPR stabilization

At the same time, Stage 17A showed that projection-head adaptation on top of
frozen `EAT` embeddings is not sufficient:

- it improved some weak machines
- but damaged strong machines such as `bearing` / `valve`
- and failed to beat the frozen reference route overall

Therefore the next question is no longer whether frozen `EAT` contains useful
signal. That has already been established.

The Stage 17B question is:

`Can a very small parameter-efficient adaptation of clean EAT improve weak machines such as fan / gearbox / toycar without damaging the strong machines enough to erase the global hmean gain?`

## 2. Positioning

Stage 17B is an **external pretrained representation adaptation branch**.

It is **not** a CTAF `full_5view` frontend result.
It must not be described as evidence that the CTAF frontend itself improved.

The current frozen reference routes remain:

- CTAF mainline frozen route:
  - `V63 fixed V37/V61 rank fusion`
- external frozen route:
  - `F_P1_mean_std_A1_065`

Stage 17B is a new adaptation branch on top of clean `EAT`, not a replacement
for that interpretation boundary.

## 3. Core Hypothesis

Clean frozen `EAT` already provides strong global anomaly-ranking signal.
However, fully frozen usage leaves residual machine-specific mismatch,
especially for weak machines like `fan`, `gearbox`, and `toycar`.

A very small, strictly controlled parameter-efficient adaptation may:

- preserve the strong global `hmean` signal of frozen `EAT`
- improve weaker machines
- avoid the broader damage caused by heavier adaptation such as projection-head
  CE training

In short:

`A very small adaptation may succeed where frozen EAT is not yet balanced, while avoiding the over-optimization problems seen in stronger adaptation attempts.`

## 4. Scope

Stage 17B is restricted to:

- clean `EAT` backbone already verified in Stage 16
- lightweight parameter-efficient adaptation only
- one-machine sanity first
- no immediate 7-machine run

Priority sanity machines:

- `fan`
- `gearbox`
- `toycar`

These are chosen because earlier stages showed they contain adaptation-relevant
signal, while still being weak points under frozen routes.

## 5. Allowed Adaptation Family

The branch may use one of the following **light** adaptation styles:

- LoRA-light
- adapter-light

The first implementation must choose **one** family only.
No parallel family search is allowed in the same preregistered run.

Recommended first branch:

- `LoRA-light`

Reason:

- simpler parameter accounting
- easier isolation of trainable subspace
- clearer comparison to the frozen backbone

## 6. Forbidden Adaptation Styles

The following are not allowed in Stage 17B first version:

- full fine-tuning
- projection-head-only CE retry
- contrastive objective
- pseudo-anomaly objective
- additional score fusion search
- per-machine adaptation type selection
- multi-model search
- hidden layer sweep before sanity

## 7. Fixed Backbone and Input

Backbone:

- clean `EAT`
- local path only
- `/data1/test/pretrained_models/eat_clean/checkpoints`

Input:

- current-year DCASE root
- `/data1/test/LAES_2025/data/dcase`
- raw waveform at `16 kHz`
- EAT’s local preprocessing path

Output representation:

- frame output `[B, T, 768]`
- utterance embedding pooling must be explicitly fixed before implementation

Current best frozen pooling reference:

- `P1_mean_std`

So Stage 17B should default to:

- `P1_mean_std`

unless a new prereg explicitly changes it.

## 8. Fixed Evaluation Protocol

Memory:

- source train-normal only

Test:

- same current-year test split as prior frozen audits

Scoring:

- `kNN`
- `k = 5`

No score fusion inside the first Stage 17B sanity.
No oracle route.
No per-machine route mixing.

## 9. One-Machine Sanity Rule

Stage 17B must begin with a one-machine sanity before any wider run.

Required sanity checklist:

- model loads
- trainable parameter count reported
- frozen vs trainable parameter split reported
- no unexpected gradient on frozen backbone outside the allowed adaptation block
- one-machine train/eval finishes
- outputs are saved correctly

A one-machine sanity must not automatically promote to 7-machine execution.
Human review is required after sanity.

## 10. Candidate Success Criteria

For one-machine sanity, the route is considered promising only if:

- `hmean` is not clearly below the frozen `EAT` reference for that machine
- `pAUC` does not collapse
- strong-machine failure pattern is not obviously reproduced
- the adaptation does not simply compress both normal and anomaly scores together

For later multi-machine promotion, success should require:

- global `hmean` improvement over the current external frozen route
- no major `bearing / valve` collapse
- meaningful `fan / gearbox / toycar` improvement

## 11. Failure Criteria

Stage 17B should be stopped early if any of the following occurs:

- one-machine sanity is worse than the frozen `EAT` baseline with no compensating gain
- score compression pattern reappears
- adaptation helps one weak machine but obviously destroys strong machines
- parameter count is no longer “light”
- implementation requires hidden hyperparameter search

## 12. Human Decisions Required Before Implementation

Before implementation, the following must be explicitly frozen:

1. `LoRA-light` or `adapter-light`
2. exact target modules
3. rank / bottleneck size
4. which pooling is fixed for the first run
5. first one-machine sanity choice:
   - `fan` / `gearbox` / `toycar`
6. whether the first sanity is train-normal only with source memory
7. exact command
8. whether to save a frozen-backbone reference in the same output bundle for direct comparison

## 13. Blocked Actions

Do not:

- open 7-machine Stage 17B immediately
- fine-tune full EAT
- add new loss families
- run multi-branch adaptation search
- do per-machine adaptation choice
- open Stage 17C automatically

## Final Statement

Stage 17B is a minimal parameter-efficient adaptation hypothesis on top of the
clean EAT external branch. It must not be treated as CTAF frontend improvement,
and it must not proceed beyond one-machine sanity without explicit human review.
