# Release Pack Audit

Created package:

`release/nearfar_relation_tensor_asd`

Purpose:

Provide a clean GitHub-ready code seed for the paper method without exposing
local experimental stages, generated data, DCASE audio, or manuscript work files.

Files added:

- `pyproject.toml`
- `.gitignore`
- `README.md`
- `configs/default.yaml`
- `examples/manifest_example.csv`
- `docs/release_scope.md`
- `docs/reproducibility_checklist.md`
- `src/nearfar_relation_tensor_asd/*.py`
- `scripts/extract_relation_tensors.py`
- `scripts/score_mpca_patchmax.py`
- `scripts/route_score_pool.py`

Excluded by design:

- raw audio;
- tensor cache `.npz`;
- internal stage scripts and outputs;
- paper draft workspace;
- figure generation intermediates;
- server-specific paths.

