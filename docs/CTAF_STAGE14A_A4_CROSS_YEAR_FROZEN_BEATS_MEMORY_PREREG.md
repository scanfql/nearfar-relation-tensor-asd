# CTAF Stage 14A A4 Cross-Year Frozen BEATs Memory Prereg

## 1. Motivation

Stage 13 and Stage 14A already established two strong facts:

- `V61` proved that cross-year normal-only support helps CTAF global ranking.
- `A0/A1` proved that frozen BEATs contains strong global-ranking information and
  that fixed `V63 + raw BEATs` fusion is a valid positive external branch.

At the same time, `A2` showed that minimal PCA/whitening adaptation is not the
right way to stabilize the BEATs branch.

Therefore A4 asks a different question:

`Can the cross-year normal-only support idea from V61 be transferred to the frozen BEATs branch through memory-bank expansion only, without training or adaptation?`

Important definition:

Frozen BEATs is **not** trained.
So A4 is **not** cross-year pretraining of BEATs.
A4 is:

`cross-year normal-only memory-bank expansion for frozen BEATs kNN scoring`

## 2. Fixed Manifest

All cross-year memory support must be derived from:

`results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv`

This manifest contains:

- 2023 train normal
- 2024 train normal
- 2025 current/dev train normal
- source normal
- a small amount of target train-normal

Because of that, A4 must split the candidates into:

### Primary candidates

`B0 = A0 raw frozen BEATs, current-year source-only memory`

`B2s = cross-year frozen BEATs kNN using source-only memory`

Rules for `B2s`:

- years = 2023 / 2024 / 2025
- `label_status = normal`
- `domain/source_target = source only`
- exclude target train-normal
- exclude test
- exclude anomaly
- exclude 2025 `eval_data_*`
- exclude 2026

`F2s = fixed rank fusion`

`F2s = 0.5 * rank_norm(V63_score) + 0.5 * rank_norm(B2s_score)`

### Diagnostic candidates

`B2f = cross-year frozen BEATs kNN using full V61-clean manifest memory`

Rules for `B2f`:

- years = 2023 / 2024 / 2025
- `label_status = normal`
- source + target train-normal
- still exclude test
- still exclude anomaly
- still exclude 2025 `eval_data_*`
- still exclude 2026

`F2f = fixed rank fusion`

`F2f = 0.5 * rank_norm(V63_score) + 0.5 * rank_norm(B2f_score)`

Interpretation rule:

- `B2s / F2s` are the primary A4 results
- `B2f / F2f` are diagnostic / upper-support ablations only
- if only `B2f/F2f` improves, the conclusion must be:
  - improvement depends on target-domain normal memory, not purely cross-year
    source-only support
- do not treat full-manifest memory as the primary result

## 3. Core Research Question

The A4 question is:

`Does cross-year source-only normal memory improve frozen BEATs enough to strengthen the external branch without training?`

This is intentionally narrower than adaptation or fine-tuning.
The only change is the normal memory bank.

## 4. Fixed Inputs

BEATs checkpoint:

- `/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt`

Current data root:

- `/data1/test/LAES_2025/data/dcase`

CTAF frozen route:

- `results/v63_fixed_v37_v61_rankfusion`

Frozen BEATs baseline:

- `results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats`

Seeds:

- `13711`
- `13721`
- `13731`

Machines:

- `bearing, fan, gearbox, slider, toycar, toytrain, valve`

## 5. Script Modes

The implementation stage must support:

### `--mode check`

- check BEATs checkpoint exists
- check V61-clean manifest exists
- check V63 root exists
- check A0 root exists
- check `data_root` exists
- report source-only count and full-manifest count per machine/year
- no model load
- no embedding extraction
- no scoring

### `--mode extract_cross_year_beats`

- load BEATs checkpoint
- `eval()` + `torch.no_grad()`
- extract frozen BEATs embeddings for both:
  - source-only memory rows
  - full-manifest memory rows
- reuse existing A0 current test embeddings if available
- if current test embeddings are missing, extract them
- save separate caches:
  - `b2_source_memory_embeddings`
  - `b2_full_memory_embeddings`
  - `current_test_embeddings`
- no scoring if possible

### `--mode score_cross_year_beats`

- compute `B2s` kNN score using source-only memory
- compute `B2f` kNN score using full-manifest memory
- `k = 5`
- output separate score NPZ and metrics:
  - `b2_source_beats`
  - `b2_full_beats`
- compare `B2s` and `B2f` against:
  - `B0 / A0 raw BEATs current-year source-only`
  - `V63`

### `--mode fuse_v63_cross_year_beats`

- load `V63` scores
- load `B2s` scores
- load `B2f` scores
- alignment check
- compute:
  - `F2s = 0.5 * rank_norm(V63) + 0.5 * rank_norm(B2s)`
  - `F2f = 0.5 * rank_norm(V63) + 0.5 * rank_norm(B2f)`
- output separate metrics and diagnostics
- `F2s` is primary
- `F2f` is diagnostic only

## 6. Success Gates

`B2s` is positive only if:

- `hmean >= A0 raw BEATs`
- `pAUC >= A0 raw BEATs`
- no severe `bearing / fan / toycar` collapse

`F2s` is positive only if:

- `hmean >= A1`
- `pAUC >= A1 - 0.005`
- `TPR@99n >= A1 - 0.005`
- improvement is not driven by one machine only
- `fan / bearing` do not further collapse relative to A1
- `toycar` does not severely degrade
- no data leakage
- no test-label selection

`B2f / F2f`:

- diagnostic only
- do not use them to declare primary success
- if `F2f` beats `F2s`, report it as:
  - target-domain memory effect / upper-support diagnostic

Interpretation rules:

- if `B2` improves `hmean` but hurts `pAUC`, report:
  - `cross-year BEATs hmean-positive but pAUC-negative`
- if `F2` improves `hmean` but hurts `pAUC/TPR`, report:
  - `fusion hmean-positive but not balanced`

## 7. Blocked Actions

The following are forbidden:

- no training
- no BEATs fine-tuning
- no PCA / whitening
- no projection head
- no LoRA / adapter
- no fusion weight tuning
- no per-machine selection
- no 2026
- no 2025 `eval_data_*`
- no anomaly labels
- no Stage 14B opened

## Final Statement

Stage 14A-A4 tests whether V61-style cross-year normal support can be
transferred to frozen BEATs through memory-bank expansion only. The primary
result must remain source-only (`B2s/F2s`), while full-manifest memory
(`B2f/F2f`) is diagnostic only. No implementation that fine-tunes BEATs or
introduces hidden adaptation is authorized unless separately pre-registered.
