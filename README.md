# Localized Near/Far Relation Tensor ASD

Reference implementation for:

**Localized Near/Far Relation Tensor Residual Evidence for Low-Support Dual-Channel Anomalous Sound Detection**

This repository contains the public-facing code path for constructing compact
near/far relation tensors, fitting source-normal MPCA residual models, extracting
HybridTail and PatchMax residual evidence, and running development-protocol
score-pool routing.

## Scope

Included:

- Paired near/far audio loading.
- Log-Mel extraction and compact `96 x 64 x 12` relation tensor construction.
- Source-normal robust channel normalization.
- Machine-wise MPCA residual modeling from normal source-domain tensors.
- HybridTail residual scoring.
- PatchMax localized residual scoring and channel-share diagnostics.
- Development-protocol score-pool routing utilities.

Not included:

- DCASE raw audio data.
- Generated tensor caches or score manifests.
- Internal exploratory stage directories.
- Temporary paper drafting files.
- Blind-evaluation submissions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Manifest Format

The extraction script expects a CSV with:

```text
basename,near_path,far_path,machine_type,domain,split,label
```

Use `examples/manifest_example.csv` as a template. The `label` column is optional
for tensor construction and MPCA fitting. It is only needed for development-set
metric reporting or development-protocol routing.

## Basic Workflow

Extract relation tensors:

```bash
python scripts/extract_relation_tensors.py \
  --manifest examples/manifest_example.csv \
  --out-dir outputs/tensors
```

Fit MPCA and score HybridTail/PatchMax:

```bash
python scripts/score_mpca_patchmax.py \
  --tensor-manifest outputs/tensors/tensor_manifest.csv \
  --out-dir outputs/scores
```

Run development-protocol routing from a score manifest:

```bash
python scripts/route_score_pool.py \
  --score-manifest outputs/scores/score_pool_manifest.csv \
  --base-column frozen_base_score \
  --score-columns score_hybridtail,score_patchmax \
  --out-dir outputs/routing
```

## Protocol Notes

- MPCA statistics and bases must be fitted from normal source-domain tensors
  only.
- Anomaly clips must not be used for tensor normalization, MPCA fitting, or
  tensor-score construction.
- `route_score_pool.py` uses development labels to select score/alpha per group.
  It should be reported as development-protocol routing, not as an unlabeled
  deployment rule.
- The unrestricted route is an upper-reference analysis and should not be
  described as a clean deployment setting.

## Reproducibility

The paper's locked numerical results depend on the exact DCASE2026 development
data layout, tensor cache, frozen non-tensor anchor scores, and score manifests.
Those large or licensed artifacts are intentionally not bundled here. This
package provides the method code and reproducibility interface needed to rebuild
the tensor branch when those inputs are available locally.

## Repository Hygiene

Do not commit:

- `data/`
- `outputs/`
- `tensor_cache/`
- `.npz`, `.npy`, `.wav`, or `.flac` files
- internal `stage*` exploration directories

