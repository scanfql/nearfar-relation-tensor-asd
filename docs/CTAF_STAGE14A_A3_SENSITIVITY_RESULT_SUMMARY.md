# CTAF Stage14A A3 Sensitivity Result Summary

## 1. Objective

Stage 14A-A3 evaluated fixed-weight sensitivity for the external branch:

- no training
- no BEATs fine-tuning
- no projection head
- no per-machine selection
- no oracle selection

The only variable was the fixed global fusion weight:

`score_alpha = alpha * rank_norm(V63) + (1 - alpha) * rank_norm(raw BEATs)`

with:

- `alpha ∈ {0.2, 0.35, 0.5, 0.65, 0.8}`

Interpretation:

- lower `alpha` = more BEATs-dominant, more hmean-oriented
- higher `alpha` = more CTAF-dominant, more conservative / pAUC-stable

## 2. Reference Routes

Reference routes:

- `V63`
  - `hmean = 0.562753`
  - `pAUC = 0.549762`
  - `TPR@99n = 0.089524`

- `A0 raw BEATs`
  - `hmean = 0.578538`
  - `pAUC = 0.538797`
  - `TPR@99n = 0.050000`

- `A1 fixed 0.5 / 0.5 rank fusion`
  - `hmean = 0.581330`
  - `pAUC = 0.548434`
  - `TPR@99n = 0.097619`

`A1` remains the preregistered frozen external route baseline for this
sensitivity study.

## 3. Alpha Sensitivity Table

| candidate | alpha | mean hmean | mean pAUC | mean KS | mean TPR@99n |
|---|---:|---:|---:|---:|---:|
| alpha_0.20 | 0.20 | 0.582533 | 0.542820 | 0.500952 | 0.076190 |
| alpha_0.35 | 0.35 | 0.582404 | 0.546679 | 0.479048 | 0.088095 |
| alpha_0.50 | 0.50 | 0.581330 | 0.548434 | 0.421905 | 0.097619 |
| alpha_0.65 | 0.65 | 0.576797 | 0.549173 | 0.367619 | 0.099524 |
| alpha_0.80 | 0.80 | 0.572227 | 0.553170 | 0.312381 | 0.093333 |

## 4. Key Findings

### 4.1 Highest hmean

The highest `hmean` occurs at:

- `alpha = 0.20`
- `hmean = 0.582533`

This is only slightly above the frozen `A1` route:

- `Δ hmean vs A1 = +0.001203`

But the cost is substantial:

- `Δ pAUC vs A1 = -0.005614`
- `Δ TPR@99n vs A1 = -0.021429`

So `alpha = 0.20` is only an **exploratory hmean-best point**, not a better
balanced route.

### 4.2 Highest pAUC

The highest `pAUC` occurs at:

- `alpha = 0.80`
- `pAUC = 0.553170`

This exceeds `A1`, but:

- `hmean = 0.572227`
- `Δ hmean vs A1 = -0.009103`

So `alpha = 0.80` is a more conservative, CTAF-dominant option, but not the
best hmean route.

### 4.3 Highest TPR@99n

The highest overall `TPR@99n` occurs at:

- `alpha = 0.65`
- `TPR@99n = 0.099524`

This is slightly above `A1`, but:

- `hmean = 0.576797`
- below `A1` by `0.004533`

So `alpha = 0.65` is a weak tail-oriented exploratory point, not a dominant
replacement.

## 5. Machine-Level Tradeoff Pattern

### BEATs-dominant side (`alpha = 0.20`)

Advantages:

- stronger `gearbox`
- stronger `toycar`
- stronger `toytrain`
- strongest `valve` hmean

Disadvantages:

- `bearing` drops substantially
- `fan` collapses further
- `TPR@99n` falls relative to `A1`

### CTAF-dominant side (`alpha = 0.80`)

Advantages:

- `bearing` recovers
- `fan` improves strongly relative to lower-alpha BEATs-heavy routes
- `pAUC` is highest

Disadvantages:

- `hmean` is clearly below `A1`
- `gearbox / toycar / toytrain` lose part of the BEATs-driven hmean gain

### Balanced point (`alpha = 0.50`)

Advantages:

- best balance between global ranking and pAUC preservation
- strong `valve`
- acceptable `bearing`
- non-catastrophic `toycar`
- better overall `TPR@99n` than both raw BEATs and V63

Disadvantages:

- `fan` remains a tradeoff machine
- hardest low-FPR failures are not solved

## 6. Interpretation

The sensitivity curve shows a clear tradeoff surface:

- lower `alpha` improves hmean slightly
- higher `alpha` improves pAUC and stabilizes fan/bearing
- no single `alpha` dominates all metrics simultaneously

Therefore:

- `alpha = 0.20` is the exploratory hmean-best point
- `alpha = 0.80` is the exploratory pAUC-best point
- `alpha = 0.65` is the exploratory TPR@99n-best point
- but `alpha = 0.50` remains the best **balanced preregistered frozen route**

This is exactly why A3 must remain a sensitivity analysis, not an automatic
route replacement step.

## 7. Final Conclusion

Stage 14A-A3 shows that fixed-weight sensitivity reveals a real tradeoff
between:

- BEATs-driven hmean gain
- CTAF-driven stability and pAUC preservation

However, no fixed alpha clearly dominates `A1` on all important axes.

Final conclusion:

`Keep A1 (alpha = 0.50) as the frozen external route.`

Additional alpha values may be reported only as exploratory sensitivity points:

- `alpha = 0.20`: hmean-oriented exploratory best
- `alpha = 0.80`: pAUC-oriented exploratory best
- `alpha = 0.65`: TPR@99n-oriented exploratory best

But none should replace the A1 frozen route without a new preregistration.

## 8. Blocked Actions

Do not:

- replace `A1` automatically
- tune weights beyond the preregistered alpha grid
- do per-machine alpha selection
- claim A3 solved the external-branch tradeoff
- open Stage 14B automatically
