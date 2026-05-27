# CTAF Stage 10 / V54 — Frozen BEATs Representation Preregistration

## 1. Motivation

CTAF V39–V53 has already converged on a stable conclusion:

- `full_5view + global latent_summary_knn` is the current empirical local optimum inside the existing CTAF / AE+kNN framework
- frontend replacement, pooling replacement, local segment scoring, hard-normal consistency, learned one-class boundary scoring, and invariance SSL do not yield a stable low-FPR breakthrough
- further low-FPR improvement is unlikely to come from continuing local modifications inside the current representation-and-scoring loop

Therefore Stage 10 does **not** continue:

- CTAF frontend variants
- pooling variants
- patch / segment scoring variants
- hard-normal consistency variants
- OC-SVM / density / kNN scoring variants
- VICReg / BYOL / SimSiam style self-developed SSL variants

The reason to pivot is simple:

```text
Current CTAF representation learning is limited to DCASE normal-only machine sound data.
If low-FPR bottlenecks remain after exhausting internal frontend / pooling / scoring / SSL changes,
the next plausible source of gain is external acoustic prior learned from large-scale pretraining.
```

This is exactly the core lesson suggested by the paper:

**Exploring Self-Supervised Audio Models for Generalized Anomalous Sound Detection**

The paper motivates three important points for CTAF Stage 10:

1. **Audio pre-trained models are more suitable than speech pre-trained models for ASD.**  
   The paper explicitly compares speech-oriented and audio-oriented self-supervised models and argues that general audio pretraining is better aligned with machine anomalous sound detection than speech pretraining.

2. **BEATs is especially strong on DCASE 2024 dev/eval.**  
   The paper reports strong ASD performance when using BEATs-family representations, showing that audio-pretrained representations can transfer effectively to generalized anomalous sound detection.

3. **The gain comes from external pretraining knowledge, not merely architecture shape.**  
   The important implication for CTAF is that the value of BEATs is not that it is “just another encoder,” but that it has been trained on large-scale external audio data and therefore contains acoustic prior that CTAF’s in-domain normal-only training cannot easily reproduce.

So Stage 10 asks a different question from V39–V53:

```text
Can a frozen external audio representation improve low-FPR tail separation,
before any fine-tuning, LoRA, adapter, or custom loss is introduced?
```

## 2. Hypothesis

Core hypothesis:

```text
Frozen BEATs audio representation contains external acoustic prior not available in
CTAF full_5view AE/SSL training.
Using frozen BEATs embeddings with a fixed label-free ASD backend may improve
low-FPR tail separation compared with CTAF full_5view latent_summary_knn.
```

中文解释：

```text
当前 CTAF 自研前端和自监督表示都受限于 DCASE normal-only 数据。
BEATs 这类大规模音频预训练模型可能已经学到更通用的声学结构，
因此有可能改善 bottleneck machines 的 low-FPR 表现。
```

The hypothesis explicitly assumes:

- do **not** train the BEATs encoder
- do **not** modify the current CTAF frontend
- do **not** mix BEATs with CTAF scores in the first version
- do **not** use anomaly labels in representation training or model fitting

## 3. Primary Candidate

The first version allows only one primary candidate:

```text
Frozen BEATs embedding + kNN backend
```

Fixed requirements:

- BEATs encoder is frozen
- no fine-tuning
- no LoRA
- no adapter
- no DLCL
- no classifier training
- no anomaly labels
- backend = kNN
- `k` fixed, recommended `k = 5`
- scoring = distance to source train-normal memory bank

This first candidate is intentionally narrow:

```text
Stage 10 first validates whether external representation itself is useful,
before any adaptation or additional complexity is introduced.
```

## 4. Why Not Full Paper Reproduction First

The paper contains a much richer system than “frozen BEATs + kNN,” including:

- Fully-Connected Multi-Branch LoRA
- Machine-aware Group Adapter
- Dual-Level Contrastive Loss
- BEATs fine-tuning

We do **not** reproduce the full paper first because that would confound the source of improvement.

If we immediately include:

- external pretrained representation
- LoRA adaptation
- adapter modules
- contrastive loss
- fine-tuning

then we cannot tell whether any gain comes from:

- the frozen external representation itself
- the adaptation mechanism
- the optimization objective
- or the interaction of all of them

So Stage 10 first freezes the question to:

```text
Is external pretrained audio representation itself useful for low-FPR ASD,
before any adaptation is added?
```

## 5. Fixed Protocol Draft

If V55 is later implemented, the protocol should be:

- data = DCASE 2024 / same current CTAF data
- machines = `bearing, fan, gearbox, slider, toycar, toytrain, valve`
- training memory = source train-normal only
- test = same current CTAF protocol
- model = frozen BEATs
- embedding = fixed utterance-level embedding
- pooling = BEATs default / mean pooling  
  exact pooling choice must be manually confirmed before implementation
- backend = kNN
- `k = 5`
- no target train
- no anomaly label
- no score fusion
- no per-machine model selection

## 6. Baselines

V55 must compare against all of the following:

1. **CTAF current mainline**  
   `V37 / full_5view / global latent_summary_knn`

2. **Simple current baseline**  
   `logmel_only / latent_summary_knn` or the currently accepted simple baseline

3. **Frozen BEATs + kNN**

The key question is not whether BEATs “looks good” in isolation, but whether it
improves over the current CTAF mainline under the same low-FPR evaluation logic.

## 7. Success Gates

Frozen BEATs is positive only if all of the following hold:

1. `mean hmean >= V37 full_5view latent_summary_knn - 0.003`
2. `mean pAUC >= V37 full_5view latent_summary_knn`
3. at least `2` bottleneck machines improve `TPR@99n`
4. `bearing / valve` do not collapse
5. `toycar` does not clearly degrade
6. normal `q99` does not clearly inflate
7. gains are not carried by only one machine

## 8. Failure Criteria

V55 frozen BEATs fails if any of the following occur:

- `hmean` is clearly below V37
- `pAUC` is below V37
- `bearing / valve` collapse
- `toycar` degrades
- normal `q99` inflates
- only one machine improves
- the line only works after fine-tuning / LoRA / fusion

## 9. Forbidden Actions

At the current preregistration stage, the following are prohibited:

- do not write code
- do not train models
- do not download weights
- do not open V55
- do not fine-tune BEATs
- do not add LoRA
- do not add adapters
- do not reproduce DLCL
- do not do score fusion
- do not do per-machine selection
- do not tune `k`
- do not automatically open V56

## 10. Human Decisions Required Before Implementation

The following must be manually confirmed before any implementation starts:

1. which BEATs checkpoint to use
2. whether BEATs weights are already available on the server
3. input audio sampling rate
4. utterance embedding pooling method
5. kNN distance metric
6. whether to use only source train-normal memory
7. exact V55 execution command
8. whether to run a 1-machine sanity check first

## Final Sentence

Stage 10 is a new external representation hypothesis. It must not be treated as a continuation of CTAF V39–V53 local modification search. No implementation or training is authorized until this preregistration is reviewed and approved.
