# Reproducibility Checklist

Before publishing or tagging a release:

- [ ] Confirm DCASE data are not committed.
- [ ] Confirm generated `.npz`, `.npy`, `.wav`, `.flac` files are not committed.
- [ ] Confirm no local absolute paths are required by public scripts.
- [ ] Confirm README commands run after `pip install -e .`.
- [ ] Confirm MPCA fitting uses only normal source-domain tensors.
- [ ] Confirm development labels are used only in metric/routing scripts.
- [ ] Confirm routing outputs are described as development-protocol results.
- [ ] Confirm frozen base score inputs are documented as external score manifests.
- [ ] Confirm paper figures are kept separate from code unless deliberately added
  as static manuscript assets.

