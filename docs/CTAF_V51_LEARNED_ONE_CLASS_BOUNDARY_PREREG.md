# CTAF V51 Learned One-Class Boundary Preregistration

## 1. Motivation

CTAF V39–V50 has already converged on a stable conclusion:

- mechanism-style `3-view` frontends do not replace `full_5view`
- pooling variants do not preserve the current full-5view balance
- segment scoring is compensatory for weak frontends, not a universal upgrade
- hard-normal consistency can compress normal tails, but does not produce a
  deployable low-FPR improvement
- density/GMM scoring changes do not solve the low-FPR bottleneck
- kNN variants and local-tail score tricks do not reliably improve
  bottleneck-machine `TPR@99n`

Therefore V51 does **not** continue:

- frontend replacement
- pooling replacement
- segment scoring replacement
- hard-normal consistency search
- density/GMM micro-variants
- kNN score variants

The evidence instead suggests:

```text
V37 full_5view representation is already relatively strong,
but latent_summary_knn remains insufficient at the low-FPR tail.
```

So the next question is no longer:

```text
How do we alter the frontend / pooling / local tail score again?
```

The next question becomes:

```text
Can we learn a better normal boundary on top of the frozen V37 representation?
```

## 2. Hypothesis

Core hypothesis:

```text
Frozen V37 full_5view latent_summary already contains useful normal/anomaly
information, but kNN distance is not an adequate low-FPR normal boundary.
A learned one-class boundary trained only on normal latent summaries may
improve low-FPR tail separation without modifying the encoder or frontend.
```

This hypothesis explicitly assumes:

- do **not** train the encoder
- do **not** change `full_5view`
- do **not** change latent summary pooling
- only learn a normal boundary
- anomaly labels do not participate in training

## 3. Candidate Method

Only one primary candidate is allowed in the first V51 audit.

### Primary

```text
Frozen V37 latent_summary + One-Class SVM (RBF)
```

Fixed definition:

- feature = frozen V37 `full_5view` latent_summary
- latent_summary = channel-wise `mean + std + max` concat
- feature dimension = `192`
- preprocessing = train-normal feature-wise z-score
- boundary model = `One-Class SVM`
- kernel = `RBF`
- `nu = 0.05`
- `gamma = "scale"`
- anomaly score = `-decision_function(x)`
- each `seed / machine` trains one separate OC-SVM on source train-normal only

The key design intention is:

```text
preserve the current representation and replace only the low-FPR boundary model
```

Explicitly fixed and forbidden within V51:

- no kernel sweep
- no `nu` sweep
- no `gamma` sweep
- no simultaneous testing of linear / RBF / shallow SVDD and then choosing the
  best one
- no alternative boundary model inside V51 if OC-SVM fails

Explicitly out of scope for V51:

- trainable Deep SVDD with encoder updates
- large MLP boundary networks
- representation-learning changes that risk collapse
- optional diagnostic boundary baselines in the same execution round

## 4. Difference from Previous Attempts

### Difference from V38 Density Scoring

`V38` was:

- static density fitting
- no learned decision surface beyond density estimation

`V51` is:

- a learned normal boundary
- fitted explicitly to separate normal support from off-support points in the
  frozen feature space

### Difference from V49 Hard-Normal Consistency

`V49`:

- changed the training objective
- tried to compress the normal tail during representation learning

`V51`:

- does not change encoder training
- does not compress latent summaries
- only learns a boundary on top of the frozen representation

### Difference from V50 Patch Summary

`V50`:

- changed feature aggregation
- tested whether coarse spatial structure should replace global pooling

`V51`:

- keeps the same aggregation
- changes only the boundary scoring mechanism

### Difference from kNN

`kNN` is:

- neighborhood distance to train-normal samples

`V51` is:

- a learned one-class boundary
- potentially better able to distinguish rare-but-normal support regions from
  true off-support anomalies

## 5. Fixed Protocol

### Input

- V37 full_5view teacher-student pretrained checkpoints / score files
- latent_summary feature
- seeds = `13711, 13721, 13731`
- machines = `bearing, fan, gearbox, slider, toycar, toytrain, valve`

### Training Data

- source train-normal only
- no anomaly labels
- no test normal for boundary training
- no target-train unless already inherent in the frozen V37 protocol

Default interpretation for V51:

```text
source normal only
```

### Preprocessing

- fit train-normal feature-wise z-score on train-normal latent_summary
- apply the same normalization to test
- no label-based calibration

### Boundary Training

- train one OC-SVM boundary on train-normal latent_summary only
- hyperparameters are fixed before execution
- no sweep
- no per-machine model selection
- no score fusion
- one model per `seed / machine`

## 6. Forbidden Actions

The following are explicitly forbidden under this preregistration:

- do not train the V37 encoder
- do not change the frontend
- do not change latent_summary pooling
- do not introduce patch or segment pooling here
- do not retune kNN
- do not do score fusion
- do not do per-machine model selection
- do not use anomaly labels
- do not use dev labels to choose models
- do not dynamically choose hard-normal
- do not open V52
- do not replace OC-SVM with another boundary model inside V51 if results are
  weak
- do not implement code in this preregistration step
- do not run training in this preregistration step

## 7. Success Gates

V51 is positive only if all of the following are met:

1. `mean hmean >= V37 latent_summary_knn baseline - 0.003`
2. `mean pAUC >= V37 latent_summary_knn baseline`
3. at least `2` bottleneck machines improve `TPR@99n`
4. `bearing / valve` do not collapse
5. `toycar` does not show clear degradation
6. normal `q99` does not clearly inflate
7. gains are not carried by only one machine

## 8. Failure Criteria

V51 fails if any of the following occur:

- `hmean` is clearly below V37
- `pAUC` is below V37
- `bearing / valve` collapse
- `toycar` degrades
- only `gearbox / slider` improve while main machines worsen
- normal `q99` inflates
- usefulness depends on tuning or per-machine selection

## 9. Implementation Plan After Approval

This section is a plan only. No code implementation is authorized in the
current step.

After manual approval of this preregistration, the next step may implement:

```text
scripts/audit_v51_learned_one_class_boundary.py
```

That script should:

- load frozen V37 latent_summary features
- train a fixed one-class boundary model
- write `results/v51_learned_one_class_boundary_audit/`
- write `docs/CTAF_V51_LEARNED_ONE_CLASS_BOUNDARY_AUDIT.md`

But this script must **not** be implemented in the current step.

## 10. Final Sentence

V51 is a new pre-registered Stage-8 hypothesis. It must not be treated as a continuation of V39–V50 parameter search. No code implementation or training is authorized until this preregistration is reviewed and approved.
