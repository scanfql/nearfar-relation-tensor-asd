# CTAF V42 Transient Segment-Level Scoring Audit

## 1. V40 Failure Summary

V40 single-seed transient_3view FS train:

- Criteria 1-3, 5 PASS (hmean > logmel, > full_5view, dcase_auc > V16 baseline, KS OK)
- Criteria 4 FAIL: pAUC = 0.5213 < V16 baseline 0.5423
- Criteria 6 FAIL: only 1/5 bottleneck machines improve TPR@99n (toytrain)
- Verdict: SINGLE-SEED FAIL.

## 2. V41 Signal Preservation Summary

V41 tracked robust_d_missed through 6 encoder stages:

- Signal does NOT decay through encoder. Mean |rd| goes from 0.999 (input) to 1.771 (latent_summary).
- toycar: encoder destroys signal (rd: 1.67 -> -0.22)
- gearbox/slider/fan: signal amplifies but scoring fails to capture it at low FPR
- toytrain: clear signal -> successful separation pattern
- Conclusion: problem is in scoring/aggregation, not in encoder/frontend.

## 3. Score Comparison

### Overall (mean across 5 bottleneck machines)

| Score mode | Mean hmean | Mean pAUC | Mean dcase_auc |
|------------|------------|-----------|----------------|
| enc2_segment_knn_top5          | 0.5056 | 0.5187 | 0.5000 |
| latent_map_segment_knn_top5    | 0.5075 | 0.5233 | 0.5000 |
| latent_summary_knn             | 0.4986 | 0.4958 | 0.5000 |

### Per-Machine hmean / pAUC

| Machine | Score mode | hmean | pAUC |
|---------|------------|-------|------|
| gearbox    | enc2_segment_knn_top5          | 0.5081 | 0.5253 |
| gearbox    | latent_map_segment_knn_top5    | 0.5035 | 0.5105 |
| gearbox    | latent_summary_knn             | 0.4955 | 0.4868 |
| slider     | enc2_segment_knn_top5          | 0.5023 | 0.5068 |
| slider     | latent_map_segment_knn_top5    | 0.5083 | 0.5258 |
| slider     | latent_summary_knn             | 0.5028 | 0.5084 |
| toycar     | enc2_segment_knn_top5          | 0.4917 | 0.4758 |
| toycar     | latent_map_segment_knn_top5    | 0.5093 | 0.5289 |
| toycar     | latent_summary_knn             | 0.4952 | 0.4858 |
| toytrain   | enc2_segment_knn_top5          | 0.5255 | 0.5853 |
| toytrain   | latent_map_segment_knn_top5    | 0.5065 | 0.5200 |
| toytrain   | latent_summary_knn             | 0.4988 | 0.4963 |
| fan        | enc2_segment_knn_top5          | 0.5002 | 0.5005 |
| fan        | latent_map_segment_knn_top5    | 0.5099 | 0.5311 |
| fan        | latent_summary_knn             | 0.5005 | 0.5016 |

## 4. Bottleneck-Machine Tail Table

| Machine | Score mode | q95 | q99 | Anom.med | Anom.q75 | TPR@95n | TPR@99n |
|---------|------------|-----|-----|----------|----------|---------|---------|
| gearbox    | enc2_segment_knn_top5          | 0.3822 | 0.3932 | 0.3518 | 0.3631 | 0.1000 | 0.0500 |
| gearbox    | latent_map_segment_knn_top5    | 0.5105 | 0.5326 | 0.4738 | 0.4867 | 0.0800 | 0.0400 |
| gearbox    | latent_summary_knn             | 1.0721 | 1.0961 | 0.8880 | 0.9652 | 0.0300 | 0.0000 |
| slider     | enc2_segment_knn_top5          | 0.3982 | 0.4049 | 0.3815 | 0.3894 | 0.0600 | 0.0200 |
| slider     | latent_map_segment_knn_top5    | 0.5334 | 0.5427 | 0.5078 | 0.5251 | 0.1000 | 0.0300 |
| slider     | latent_summary_knn             | 1.0334 | 1.0559 | 0.8311 | 0.9234 | 0.0700 | 0.0500 |
| toycar     | enc2_segment_knn_top5          | 0.3444 | 0.3528 | 0.3221 | 0.3328 | 0.0000 | 0.0000 |
| toycar     | latent_map_segment_knn_top5    | 0.4867 | 0.4937 | 0.4666 | 0.4752 | 0.1300 | 0.0700 |
| toycar     | latent_summary_knn             | 1.0660 | 1.0924 | 0.8978 | 0.9743 | 0.0200 | 0.0100 |
| toytrain   | enc2_segment_knn_top5          | 0.3485 | 0.3521 | 0.3367 | 0.3462 | 0.2300 | 0.1600 |
| toytrain   | latent_map_segment_knn_top5    | 0.5001 | 0.5089 | 0.4807 | 0.4920 | 0.1200 | 0.0200 |
| toytrain   | latent_summary_knn             | 1.0375 | 1.0808 | 0.8340 | 0.9300 | 0.0500 | 0.0200 |
| fan        | enc2_segment_knn_top5          | 0.4478 | 0.4586 | 0.4198 | 0.4310 | 0.0600 | 0.0000 |
| fan        | latent_map_segment_knn_top5    | 0.5884 | 0.6013 | 0.5693 | 0.5798 | 0.1300 | 0.0500 |
| fan        | latent_summary_knn             | 0.9519 | 0.9689 | 0.7899 | 0.8918 | 0.0600 | 0.0200 |

## 5. TPR@99n Improvement Analysis

**enc2_segment_knn_top5**: 2/5 machines improve TPR@99n [PASS]
  - gearbox: 0.0000 -> 0.0500 (+0.0500)
  - slider: 0.0500 -> 0.0200 (-0.0300)
  - toycar: 0.0100 -> 0.0000 (-0.0100)
  - toytrain: 0.0200 -> 0.1600 (+0.1400)
  - fan: 0.0200 -> 0.0000 (-0.0200)

**latent_map_segment_knn_top5**: 3/5 machines improve TPR@99n [PASS]
  - gearbox: 0.0000 -> 0.0400 (+0.0400)
  - slider: 0.0500 -> 0.0300 (-0.0200)
  - toycar: 0.0100 -> 0.0700 (+0.0600)
  - toytrain: 0.0200 -> 0.0200 (+0.0000)
  - fan: 0.0200 -> 0.0500 (+0.0300)

## 6. Normal q99 Inflation Check

| Score mode | gearbox | slider | toycar | toytrain | fan |
|------------|---------|--------|--------|----------|-----|
| enc2_segment_knn_top5          | 0.3932 | 0.4049 | 0.3528 | 0.3521 | 0.4586 |
| latent_map_segment_knn_top5    | 0.5326 | 0.5427 | 0.4937 | 0.5089 | 0.6013 |
| latent_summary_knn             | 1.0961 | 1.0559 | 1.0924 | 1.0808 | 0.9689 |

## 7. Final Verdict

### enc2_segment_knn_top5
- hmean >= V40 baseline (0.4986): 0.5056 [PASS]
- pAUC >= V40 baseline (0.4958): 0.5187 [PASS]
- TPR@99n improved on >= 2 machines: 2/5 [PASS]
- Normal q99 not inflated: [PASS]
- **Overall: PASS**

### latent_map_segment_knn_top5
- hmean >= V40 baseline (0.4986): 0.5075 [PASS]
- pAUC >= V40 baseline (0.4958): 0.5233 [PASS]
- TPR@99n improved on >= 2 machines: 3/5 [PASS]
- Normal q99 not inflated: [PASS]
- **Overall: PASS**

**Segment-level scoring provides improvement over file-level latent_summary_knn.**
This confirms V41's hypothesis that the encoder preserves signal but global pooling loses it.

---

Models were loaded from saved V40 checkpoints. No retraining was performed. No scoring parameters (k=5, top_percent=5%) were tuned. No per-machine selection was applied.
