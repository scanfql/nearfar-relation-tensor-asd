# CTAF Stage 14A A2 Minimal Adaptation Prereg

## 1. A2 Motivation

Stage 14A-A1 already established a positive external branch result:

- `A1` is `hmean`-positive
- `A1` is `pAUC`-preserved
- `A1` provides mild overall `TPR@99n` improvement

However, `A1` still shows:

- `fan / bearing` tradeoff
- no decisive low-FPR resolution
- no adaptation of the external branch itself

Therefore A2 asks a narrower question:

`Can a minimal normal-only adaptation stabilize the frozen BEATs branch without moving into fine-tuning, projection-head training, or architecture search?`

## 2. A2 Hypothesis

Frozen BEATs embeddings contain strong global-ranking information.
Normal-only PCA/whitening may reduce domain or scale mismatch and produce a
more stable BEATs branch.

Under this hypothesis:

- `hmean` should remain close to or above A1
- `pAUC` may improve
- `fan / bearing` tradeoffs may become less severe
- adaptation remains fully label-free and minimal

## 3. Fixed Protocol

A2 must use:

- the existing frozen BEATs branch from Stage 14A-A0
- the existing V63 frozen route from Stage 13 / Stage 14A-A1

Fixed rules:

- no new external model
- no new checkpoint download
- no BEATs fine-tuning
- no anomaly labels
- no test-label selection

## 4. Candidate Definitions

Candidates are fixed as:

- `B0 = raw frozen BEATs`
- `B1 = PCA-whitened BEATs, dim=256`
- `F0 = V63`
- `F1 = fixed V63 + B1 rank fusion`

PCA / whitening rule:

- fit PCA / whitening only on source train-normal BEATs embeddings
- fit independently per machine
- fixed dimension = `256`
- no dimension sweep
- apply the same transform to train-normal memory and test embeddings
- scoring = kNN, `k=5`

## 5. Baselines

A2 must compare against:

- `V63`
- `A0 frozen BEATs`
- `A1 V63 + raw BEATs fusion`

These baselines define whether PCA/whitening adds real value beyond the already
positive fixed-fusion route.

## 6. Success Gates

A2-F1 is positive only if:

- `hmean >= A1 - 0.002`
- `pAUC >= A1`
- `TPR@99n >= A1 - 0.005`
- `fan` does not further collapse relative to `A1`
- `bearing` does not further collapse relative to `A1`
- `toycar` does not severely degrade
- no anomaly/test label selection
- no dimension or weight tuning

Interpretation rules:

- if `hmean` improves but `pAUC` worsens, report:
  - `hmean-positive but pAUC-negative`
- if `pAUC` improves but `hmean` drops by more than `0.002`, report:
  - `pAUC-positive but hmean-negative`

## 7. Failure Criteria

A2 fails if:

- `hmean` drops clearly below `A1`
- `pAUC` drops below `A1`
- `fan / bearing` tradeoff worsens
- `toycar` collapses
- PCA dimension or fusion weight needs tuning
- any test-label-based choice is used

## 8. Blocked Actions

The following are forbidden:

- no LoRA
- no adapter
- no MLP projection head
- no BEATs fine-tuning
- no fusion weight tuning
- no per-machine selection
- no Stage 14B

## Final Statement

Stage 14A-A2 is the minimal normal-only adaptation step for the external BEATs
branch. It is restricted to frozen BEATs embedding with PCA/whitening and fixed
V63 fusion. No implementation involving LoRA, adapters, projection-head
training, or BEATs fine-tuning is authorized unless separately pre-registered.
