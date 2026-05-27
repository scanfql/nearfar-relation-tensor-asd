# CTAF Stage 14A A0/A1 Result Summary

## 1. Stage 14A Positioning

Stage 14A is a **hmean-oriented external audio pretrained representation
branch**.

It is **not** the CTAF `full_5view` frontend mainline.
Therefore the gains from Stage 14A must **not** be interpreted as evidence that
the CTAF `full_5view` frontend itself became stronger.

The current CTAF frozen mainline remains:

- `V63 fixed V37/V61 rank fusion`

Stage 14A-A1 should instead be treated as:

- a positive result for an **external representation branch**
- complementary to the CTAF frozen route
- useful for improving global anomaly ranking, especially `hmean`

## 2. A0 Frozen BEATs Summary

`A0` evaluates:

- frozen BEATs
- utterance embedding
- time mean pooling
- kNN backend
- no fine-tuning
- no projection head
- no adaptation

Overall result:

- `hmean = 0.578538`
- `pAUC = 0.538797`
- `TPR@99n = 0.050000`

Interpretation:

- frozen BEATs provides a strong `hmean` / global-ranking signal
- but its `pAUC`, `q99`, and low-FPR behavior are weaker than the best CTAF route
- therefore `A0` cannot directly replace the CTAF mainline

Per-machine summary:

| machine | hmean | pAUC | KS | TPR@99n |
|---|---:|---:|---:|---:|
| bearing | 0.5333 | 0.5663 | 0.24 | 0.07 |
| fan | 0.5074 | 0.4826 | 1.00 | 0.00 |
| gearbox | 0.5754 | 0.5416 | 0.40 | 0.04 |
| slider | 0.5788 | 0.4989 | 0.28 | 0.02 |
| toycar | 0.5538 | 0.5326 | 0.78 | 0.04 |
| toytrain | 0.6164 | 0.5095 | 0.66 | 0.00 |
| valve | 0.6848 | 0.6400 | 0.20 | 0.18 |

Major strengths:

- `valve` is very strong
- `gearbox` and `toycar` have useful global-ranking signal
- `toytrain` hmean is high

Major failures / tradeoffs:

- `fan` is clearly weak
- `toytrain` has strong hmean but `TPR@99n = 0.00`
- `slider` hmean is high but pAUC remains weak
- low-FPR behavior is not robust enough to use A0 as the main route

## 3. A1 Fixed V63 + BEATs Rank Fusion

Fixed formula:

`C2 = 0.5 * rank_norm(V63_score) + 0.5 * rank_norm(BEATs_score)`

Rules:

- rank_norm per machine / per seed
- no weight tuning
- no per-machine selection
- no oracle selection
- no training
- no BEATs fine-tuning
- no projection head

Core results:

`V63`

- `hmean ≈ 0.5628`
- `pAUC ≈ 0.5498`
- `TPR@99n ≈ 0.0895`

`V63 + BEATs fixed rank fusion`

- `hmean ≈ 0.5813`
- `pAUC ≈ 0.5484`
- `TPR@99n ≈ 0.0976`

Delta vs `V63`:

- `hmean +0.0186`
- `pAUC -0.0013`
- `TPR@99n +0.0081`

## 4. Gate Judgment

Under the Stage 14A prereg:

`A1` satisfies:

- `hmean >= V63 + 0.01`
- `pAUC >= V63 - 0.01`
- no weight tuning
- no per-machine selection
- no oracle selection
- no anomaly/test-label-based selection

Final gate verdict:

`Stage 14A-A1 PASS`

This should be phrased carefully:

`A1 is hmean-positive and pAUC-preserved, with mild overall TPR@99n improvement.`

## 5. Machine-Level Interpretation

Positive:

- `gearbox` TPR@99n improves over `V63`
- `toycar` TPR@99n improves over `V63`
- `valve` improves strongly
- `toycar` does not further degrade
- the gain is not purely one-machine-driven

Negative / tradeoff:

- `bearing` decreases
- `fan` decreases clearly, especially on `TPR@99n`
- `slider` remains unresolved
- `toytrain` remains unresolved

Per-machine interpretation of the fusion route:

| machine | V63 hmean | A1 hmean | V63 pAUC | A1 pAUC | V63 TPR@99n | A1 TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.6353 | 0.5828 | 0.6242 | 0.6002 | 0.2067 | 0.1900 |
| fan | 0.5935 | 0.5553 | 0.5795 | 0.5282 | 0.1567 | 0.0533 |
| gearbox | 0.5412 | 0.5571 | 0.5226 | 0.5209 | 0.0500 | 0.0633 |
| slider | 0.5301 | 0.5660 | 0.5288 | 0.5133 | 0.0300 | 0.0267 |
| toycar | 0.5366 | 0.5510 | 0.5110 | 0.5180 | 0.0133 | 0.0300 |
| toytrain | 0.5208 | 0.5888 | 0.4991 | 0.4954 | 0.0133 | 0.0033 |
| valve | 0.5817 | 0.6684 | 0.5832 | 0.6631 | 0.1567 | 0.3167 |

## 6. Complementarity Interpretation

`V63` and frozen `BEATs` are not redundant.

Evidence:

- `toycar` Spearman correlation is negative
- `valve / gearbox / slider` show detect-miss complementarity
- `A1` gains come from rank-level complementarity between the CTAF frozen route
  and the external audio pretrained representation

Examples of the complementarity pattern:

- `toycar` has negative score correlation, so BEATs is not simply re-ranking the
  same samples as CTAF
- `valve` shows strong mutual detect/miss asymmetry
- `gearbox` also shows non-trivial complementary detections

However:

- missed anomaly overlap remains non-trivial
- the hardest low-FPR failures are still not fully solved

So A1 should be interpreted as:

- meaningful external-branch complementarity
- but not complete resolution of the hard-tail problem

## 7. Final Interpretation

Stage 14A-A1 demonstrates that external audio pretrained representation
contains complementary global-ranking information relative to the CTAF frozen
route.

Fixed `V63 + BEATs` rank fusion:

- substantially improves `hmean`
- preserves `pAUC`
- and slightly improves overall `TPR@99n`

However, the improvement is not uniform across machines, and clear
`fan / bearing` tradeoffs remain.

Therefore:

`A1 is a valid positive external-branch result, not a proof that low-FPR tail separation is fully solved.`

## 8. Current Status

Current CTAF mainline frozen route:

- `V63 fixed V37/V61 rank fusion`

Current external hmean-oriented frozen route:

- `Stage 14A-A1 fixed V63 + BEATs rank fusion`

## 9. Blocked Actions

Do not:

- start minimal adaptation yet
- train a projection head
- fine-tune BEATs
- tune fusion weight
- do per-machine selection
- claim full_5view frontend improvement
- claim low-FPR is solved
- open Stage 14B automatically

## 10. Optional Next Steps

Future work only, not authorized now:

### A. Stage 14A-A2 minimal adaptation prereg

### B. weight sensitivity as diagnostic only

### C. fan / bearing tradeoff analysis

### D. external branch comparison table for paper
