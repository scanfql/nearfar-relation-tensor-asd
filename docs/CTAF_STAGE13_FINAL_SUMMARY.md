# CTAF Stage 13 Final Summary

## 1. Stage 13 Motivation

Stage 13 marks a deliberate shift away from the local modification search that
dominated `V39–V59`.

Earlier stages had already shown that the following directions could not
reliably solve the CTAF low-FPR bottleneck:

- frontend replacement
- pooling replacement
- local/segment scoring variants
- pseudo-anomaly training
- hard-normal consistency and related tail-compression losses
- discriminative proxy / AAM-style machine supervision
- frozen external representation substitution alone
- UACS-on-current-score style backend correction

In other words, `V39–V59` strongly suggested that simply changing frontend,
pooling, scoring, pseudo-anomaly construction, AAM-style supervision, frozen
BEATs usage, or UACS-on-current-score would not stably solve the remaining
problem.

Therefore Stage 13 re-centered the research question around three more global
axes:

- cross-year normal-only data support
- strong encoder capacity under the same CTAF protocol
- score-level complementarity between the current-year and cross-year routes

The goal was not to continue micro-search.
The goal was to test whether the remaining bottleneck came from:

- insufficient normal support
- insufficient encoder capacity
- or useful complementarity between independently trained routes

## 2. V60 / V60B Data Protocol Summary

Stage 13 first established a clean cross-year data protocol before any new
training logic was attempted.

`V60` completed a schema and protocol audit of:

- 2023
- 2024
- 2025
- 2026

`V60B` then locked a clean manifest:

- `results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv`

`V61-clean` includes only:

- 2023 train normal
- 2024 train normal
- 2025 current/dev train normal

Total usable normal-only clips:

- `21,700`

Explicitly excluded:

- all test split files
- all anomaly clips
- all 2025 `eval_data_*`
- all 2026 `Emu` / noise-aware branches

This protocol is an external normal-data pretraining protocol for research.
It is **not** a DCASE official competition submission protocol.

## 3. V61 Conclusion: Cross-Year Data Helps

`V61` kept the original V37 encoder, frontend, latent_summary pooling, and
latent_summary_knn scoring, and changed only the normal data support.

V61 3-seed results:

- `V37 hmean = 0.5539 ± 0.0065`
- `V61 hmean = 0.5629 ± 0.0079`
- `Δ hmean = +0.0090`
- `V37 pAUC = 0.5457 ± 0.0036`
- `V61 pAUC = 0.5493 ± 0.0046`
- `Δ pAUC = +0.0036`
- `q99` compressed on `5/7` machines
- no data leakage
- no protocol changes

Conclusion:

`Cross-year normal-only data support improves global anomaly ranking.`

However, the main caveat is equally important:

- `TPR@99n` was not meaningfully improved overall
- `V37 TPR@99n = 0.0800`
- `V61 TPR@99n = 0.0786`
- bottleneck machines improved on only `1/5`
- `toycar` remained difficult

So V61 solves a global-ranking problem, not the hardest low-FPR tail
separation problem.

## 4. V62A Conclusion: Strong Encoder Capacity Alone Fails

`V62A` implemented a stronger full_5view Conformer-style encoder while keeping
the same CTAF data protocol and latent_summary_knn interface.

Implementation facts:

- full_5view Conformer-style encoder implemented
- parameter count = `1,639,413`
- latent_summary dim = `384`
- shape_check = `PASS`
- gearbox dryrun completed

Gearbox comparison:

- `V37 gearbox hmean = 0.5222, pAUC = 0.5200, q99 = 0.1296, TPR@99n = 0.0600`
- `V61 gearbox hmean = 0.5356, pAUC = 0.5189, q99 = 0.1208, TPR@99n = 0.0500`
- `V62A gearbox hmean = 0.4963, pAUC = 0.5184, q99 = 0.0481, TPR@99n = 0.0300`

Mechanistic interpretation:

- the strong encoder compresses normal scores strongly
- but anomaly scores are also compressed
- `anomaly_median = 0.0335`
- `anomaly_q75 = 0.0377`
- both are below `normal q99 = 0.0481`

Therefore low-FPR separation becomes worse, not better.

Conclusion:

- `V62A implementation sanity: PASS`
- `V62A method dryrun: FAIL / early-stop`

Blocked actions after V62A:

- do not run full V62
- do not tune V62 architecture
- do not open a new training-objective branch based on V62A alone

## 5. V63 Conclusion: Fixed V37/V61 Rank Fusion Is Current Frozen Route

`V63` freezes the only allowed score-level complementarity route:

Fixed formula:

`C2 = 0.5 * rank_norm(V37_score) + 0.5 * rank_norm(V61_score)`

Rules:

- `rank_norm` per machine, per seed
- no weight tuning
- no per-machine selection
- no oracle selection
- no training
- no original score modification

3-seed comparison:

`V37`

- `hmean = 0.553898`
- `pAUC = 0.545689`
- `TPR@99n = 0.080000`

`V61`

- `hmean = 0.562946`
- `pAUC = 0.549323`
- `TPR@99n = 0.078571`

`V63 fixed rank fusion`

- `hmean = 0.562753`
- `pAUC = 0.549762`
- `TPR@99n = 0.089524`

Interpretation:

`V63` essentially preserves the global-ranking gain of `V61`, while improving
`TPR@99n` beyond both `V37` and `V61`.

Important caveats:

- `V63` uses rank-normalized scores
- `V63 q95/q99` cannot be directly compared to raw-score `q95/q99` from `V37`
  or `V61`
- `V63` does not fully solve the hardest bottleneck machines
- `gearbox / toycar / toytrain` remain weak

Final verdict:

`V63 fixed rank-fusion frozen route PASS.`

## 6. Stage 13 Final Interpretation

Stage 13 supports the following interpretation:

1. **Data support helps**

   `V61` proves that cross-year normal-only data improves CTAF global ranking.

2. **Capacity alone does not help**

   `V62A` shows that a stronger encoder under the same AE/teacher-student
   objective compresses scores, but does not lift the anomaly tail.

3. **Complementarity helps**

   `V63` shows that the current-year route (`V37`) and the cross-year route
   (`V61`) contain useful rank-level complementarity.

4. **The remaining bottleneck is still low-FPR tail separation**

   The hardest low-FPR failure cases, especially `gearbox / toycar / toytrain`,
   remain unsolved.

## 7. Current Best Route

Current best frozen CTAF route:

`V63 fixed V37/V61 rank fusion`

Components:

- V37 current-year original encoder score
- V61 cross-year original encoder score
- fixed `0.5 / 0.5` rank-normalized average
- no training
- no tuning
- no per-machine selection

## 8. Blocked Actions

The following actions are explicitly blocked:

- do not continue V62 full run
- do not tune V62 architecture
- do not tune fusion weight for the main result
- do not do per-machine score selection
- do not open V64 automatically
- do not claim low-FPR is fully solved
- do not claim V63 q99 is directly comparable to raw q99

## 9. Optional Future Work

The following are future-work options only.
They are not authorized by this summary.

### A. V63B weight sensitivity analysis

- diagnostic / appendix only
- not to change the frozen route
- not to select best alpha using test labels

### B. Stage 13C training objective mismatch

- only if a new prereg is written
- should target anomaly-tail lift / anti-score-compression
- not authorized now

### C. Toycar-specific failure analysis

- diagnostic only
- no per-machine tuning unless separately preregistered

## 10. Final Statement

Stage 13 establishes that cross-year normal-only data support and fixed
V37/V61 rank-level complementarity are the most reliable CTAF improvements so
far. The current frozen route is `V63 fixed rank fusion`. Further improvement
should not be pursued through unregistered architecture tuning or fusion-weight
search.
