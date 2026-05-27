# CTAF V40 Transient Frontend Training Validation Audit

## 1. V39 Stage-0 Summary

| Frontend | Gate A | Redundancy max_r | Stage 0 |
|----------|--------|------------------|---------|
| transient_3view | 5/5 PASS | 0.0538 | **PASS** |
| modulation_3view | 5/5 PASS | 0.9688 | FAIL (Gate C: too redundant) |
| cepstral_3view | 0/5 FAIL | 0.6089 | FAIL (Gate A: NaN distances) |

- `transient_3view` enters V40: strong input-level separation signal (mean robust_d_missed=0.738), very low redundancy (r=0.054), 4/5 machines pass all gates.
- `modulation_3view` does NOT enter V40: modulation_low r=0.969 vs logmel_base, fails pre-registered redundancy gate C.
- `cepstral_3view` does NOT enter V40: Gate A fails on all 5 machines (NaN in distance computation).

## 2. V16 Seed-Panel Mean Baseline

Pre-registered V16 5-seed panel (full_5view, latent_summary_knn, viewwise_unshared_late_fusion):

| Metric | Value |
|--------|-------|
| Mean hmean (dcase_auc) | 0.549547 |
| Mean pAUC | 0.542331 |
| Mean source-target KS | 0.294880 |

## 3. Overall Comparison

Score mode: `latent_summary_knn`. Mean across 5 machines.

| Experiment | Views | Mean hmean | Mean pAUC | Mean KS | vs V16 hmean | vs V16 pAUC | vs logmel_only |
|------------|-------|------------|-----------|---------|--------------|-------------|----------------|
| full_5view         | ? | 0.5353 | 0.5353 | 0.3640 | -0.0089 | -0.0071 | -0.0041 |
| logmel_only        | ? | 0.5394 | 0.5168 | 0.4760 | +0.0046 | -0.0255 | +0.0000 |
| transient_3view    | ? | 0.5441 | 0.5213 | 0.3320 | +0.0097 | -0.0211 | +0.0048 |

## 4. Per-Machine hmean / pAUC / KS

| Machine | Experiment | hmean | pAUC | src-tgt KS |
|---------|------------|-------|------|------------|
| fan        | full_5view         | 0.5782 | 0.5805 | 1.0000 |
| fan        | logmel_only        | 0.6027 | 0.5811 | 1.0000 |
| fan        | transient_3view    | 0.5907 | 0.5432 | 0.9000 |
| gearbox    | full_5view         | 0.5528 | 0.5458 | 0.2000 |
| gearbox    | logmel_only        | 0.5511 | 0.5153 | 0.3800 |
| gearbox    | transient_3view    | 0.5388 | 0.5147 | 0.1400 |
| slider     | full_5view         | 0.5458 | 0.5342 | 0.2200 |
| slider     | logmel_only        | 0.5208 | 0.5005 | 0.2200 |
| slider     | transient_3view    | 0.5288 | 0.5126 | 0.2400 |
| toycar     | full_5view         | 0.4716 | 0.5047 | 0.2200 |
| toycar     | logmel_only        | 0.4998 | 0.4842 | 0.4200 |
| toycar     | transient_3view    | 0.4517 | 0.4874 | 0.1400 |
| toytrain   | full_5view         | 0.5280 | 0.5111 | 0.1800 |
| toytrain   | logmel_only        | 0.5224 | 0.5032 | 0.3600 |
| toytrain   | transient_3view    | 0.6108 | 0.5484 | 0.2400 |

## 5. Bottleneck-Machine Tail Table

Bottleneck = gearbox, slider, toycar, toytrain, fan (V39 frontend-prescreen machines).

| Machine | Experiment | q95 | q99 | Anom.median | Anom.q75 | TPR@95n | TPR@99n |
|---------|------------|-----|-----|-------------|----------|---------|---------|
| gearbox    | full_5view         | 0.114580 | 0.120419 | 0.0931 | 0.1020 | 0.1600 | 0.1000 |
| gearbox    | logmel_only        | 0.140971 | 0.157250 | 0.0714 | 0.1052 | 0.0800 | 0.0300 |
| gearbox    | transient_3view    | 0.182631 | 0.282872 | 0.0792 | 0.1066 | 0.1100 | 0.0200 |
| slider     | full_5view         | 0.126747 | 0.132301 | 0.1090 | 0.1183 | 0.1000 | 0.0700 |
| slider     | logmel_only        | 0.149506 | 0.166947 | 0.1102 | 0.1251 | 0.0500 | 0.0300 |
| slider     | transient_3view    | 0.198777 | 0.374241 | 0.1018 | 0.1335 | 0.0700 | 0.0100 |
| toycar     | full_5view         | 0.079222 | 0.086943 | 0.0699 | 0.0729 | 0.0800 | 0.0300 |
| toycar     | logmel_only        | 0.075775 | 0.110197 | 0.0564 | 0.0614 | 0.0000 | 0.0000 |
| toycar     | transient_3view    | 0.234054 | 0.267145 | 0.1770 | 0.1947 | 0.0200 | 0.0000 |
| toytrain   | full_5view         | 0.106095 | 0.114992 | 0.0697 | 0.0777 | 0.0600 | 0.0500 |
| toytrain   | logmel_only        | 0.086753 | 0.116630 | 0.0521 | 0.0596 | 0.0600 | 0.0100 |
| toytrain   | transient_3view    | 0.261240 | 0.303183 | 0.1620 | 0.2120 | 0.1700 | 0.0800 |
| fan        | full_5view         | 0.231016 | 0.233771 | 0.1789 | 0.2243 | 0.2000 | 0.1800 |
| fan        | logmel_only        | 0.227307 | 0.235844 | 0.1680 | 0.2241 | 0.2000 | 0.1900 |
| fan        | transient_3view    | 0.292333 | 0.314663 | 0.2138 | 0.2707 | 0.1600 | 0.0600 |

## 6. Final Verdict

- **[PASS]** 1. transient_3view hmean (0.544149) > logmel_only hmean (0.539370)
  delta = +0.004779
- **[PASS]** 2. transient_3view hmean (0.544149) > full_5view hmean (0.535273)
  delta = +0.008876
- **[PASS]** 3. dcase_auc (0.559240) > V16 baseline (0.549547)
  delta = +0.009693
- **[FAIL]** 4. pAUC (0.521263) >= V16 baseline (0.542331)
  delta = -0.021068
- **[PASS]** 5. KS (0.332000) not significantly worse than V16 (0.294880)
  delta = +0.037120
- **[FAIL]** 6. >= 2 bottleneck machines improve TPR@99n (1 machines improved)

### Verdict: V40 SINGLE-SEED **FAIL**

`transient_3view` did not meet all pre-registered success criteria.

## 7. Failure Analysis

- V39 Stage-0 input-level separation signal did **not reliably survive** encoder training. While transient_3view showed strong robust_d at the raw-input level (mean 0.738), the AE encoder's learned representation did not consistently preserve that separation in the latent summary space.

- `transient_3view` improves `toytrain` strongly (hmean 0.5224 -> 0.6108, TPR@99n 0.01 -> 0.08), but degrades or fails to improve on toycar, gearbox, slider, and fan. The improvement is mechanism-specific, not general.

- The **dominant failure mechanism is normal-tail inflation**: q95 and q99 thresholds increase substantially under transient_3view (e.g., gearbox q99: 0.157 -> 0.283, slider q99: 0.167 -> 0.374), while anomaly scores do not move far enough beyond the new normal extreme tail. This suppresses pAUC and TPR@99n even when mean separation exists.

- Therefore `transient_3view` is a **mechanism-specific positive diagnostic**, not a general frontend replacement. Its value is in demonstrating that specific frontend views can reveal structure invisible to logmel alone on some machines (toytrain).

- V40 should be recorded as evidence that **input-level separability alone is insufficient** for low-FPR anomaly detection. The representation and scoring pipeline must be designed to preserve tail separation after training, not just before it.

### Actions blocked by this result

- Do NOT run 3-seed panel on transient_3view.
- Do NOT tune transient parameters.
- Do NOT attempt transient_6view or frontend fusion.
- Do NOT perform per-machine frontend selection.
- Do NOT auto-proceed to V41.

---

Models were trained only under the fixed V40 protocol; no score was tuned, no frontend parameters were tuned, and no per-machine selection was performed.
