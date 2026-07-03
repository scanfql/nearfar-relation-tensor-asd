# Release Scope

This package is the recommended public repository seed. It separates the
paper-facing implementation from the local exploratory workspace.

## Include

- `src/nearfar_relation_tensor_asd/`
- `scripts/`
- `configs/default.yaml`
- `examples/manifest_example.csv`
- `README.md`
- `pyproject.toml`
- `.gitignore`

## Exclude

- `baseline_reproduction/`
- `paper_workspace/`
- `paper_figures/`
- all `stage*` experiment directories
- generated tensor caches
- generated score manifests unless explicitly prepared as small examples
- DCASE audio files
- `.DS_Store`, `__pycache__`, `.pytest_cache`

## Mapping From Manuscript to Code

| Manuscript component | Public code |
| --- | --- |
| Near/far relation tensor construction | `features.py`, `scripts/extract_relation_tensors.py` |
| Source-normal robust normalization | `normalization.py` |
| MPCA residual tensor | `mpca.py`, `scripts/score_mpca_patchmax.py` |
| HybridTail score | `mpca.hybridtail_score` |
| PatchMax localized score | `patchmax.py` |
| Relation/energy channel share | `patchmax.channel_share` |
| Development-protocol score-pool routing | `routing.py`, `scripts/route_score_pool.py` |

## Internal Provenance

The public package was distilled from the final manuscript route:

- full relation tensor cache construction;
- MPCA residual scoring;
- PatchMax mechanism analysis;
- Beyond-MPCA score-pool routing.

Internal stage numbers and exploratory branches are intentionally omitted from
public-facing code and documentation.

