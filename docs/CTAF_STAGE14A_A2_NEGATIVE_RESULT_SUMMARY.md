# CTAF Stage14A A2 Negative Result Summary

## 1. A2 目标回顾

`A2` tested whether frozen BEATs embedding plus minimal normal-only
PCA/whitening could stabilize the external branch without moving into training,
fine-tuning, or architecture search.

Fixed A2 constraints:

- frozen BEATs embedding only
- normal-only PCA/whitening
- fixed `PCA dim = 256`
- no dimension sweep
- no fusion weight tuning
- no BEATs fine-tuning
- no projection head
- no LoRA / adapter

The purpose was to test whether a minimal linear normalization step could
reduce the instability of frozen BEATs while preserving its strong global
ranking signal.

## 2. B1 PCA-BEATs 结果

`A0 raw frozen BEATs`:

- `hmean = 0.5785`
- `pAUC = 0.5388`
- `TPR@99n = 0.0500`

`B1 PCA-whitened BEATs`:

- `hmean = 0.5673`
- `pAUC = 0.5238`
- `TPR@99n = 0.0314`

Conclusion:

`PCA/whitening weakens the raw BEATs global-ranking signal.`

The drop is visible on all three main summary dimensions:

- lower `hmean`
- lower `pAUC`
- lower `TPR@99n`

So the adaptation step did not stabilize raw BEATs.
It degraded it.

## 3. F1 fusion 结果

`A1 = V63 + raw BEATs fixed rank fusion`:

- `hmean ≈ 0.5813`
- `pAUC ≈ 0.5484`
- `TPR@99n ≈ 0.0976`

`F1 = V63 + PCA-BEATs fixed rank fusion`:

- `hmean ≈ 0.5781`
- `pAUC ≈ 0.5312`
- `TPR@99n ≈ 0.0571`

Relative to `A1`, `F1` is clearly worse:

- `hmean` falls below the A1 target
- `pAUC` degrades substantially
- `TPR@99n` also degrades substantially

Therefore the PCA/whitened BEATs route does not preserve the successful A1
external-branch behavior.

## 4. Gate Judgment

`A2` fails because:

- `hmean < A1 - 0.002`
- `pAUC < A1`
- `TPR@99n < A1 - 0.005`
- `fan / bearing` tradeoff is not resolved
- PCA/whitening does not stabilize frozen BEATs

This is not a near-miss.
It is a clear negative result under the fixed preregistered A2 rules.

## 5. Final Verdict

`A2 minimal PCA/whitening adaptation = FAIL.`

## 6. Updated Frozen Routes

Current CTAF mainline:

- `V63 fixed V37/V61 rank fusion`

Current external hmean-oriented branch:

- `Stage 14A-A1 fixed V63 + raw frozen BEATs rank fusion`

`A2` status:

- negative result
- do not continue PCA/whitening

## 7. Blocked Actions

Do not:

- tune PCA dimension
- tune fusion weight
- do per-machine selection
- train an MLP head
- add LoRA / adapter
- fine-tune BEATs
- open Stage 14B automatically

## 8. Optional Future Work

Only as future preregistered work:

- Stage 14B task adaptation, if explicitly opened later
- `fan / bearing` tradeoff analysis
- paper comparison table for `V63 / A0 / A1 / A2`
