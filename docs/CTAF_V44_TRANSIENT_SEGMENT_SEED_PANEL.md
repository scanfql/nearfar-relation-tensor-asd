# CTAF V44 Transient Segment Scoring 3-Seed Panel

## 1. V43 Full-Machine Validation Summary

V43 validated segment scoring on all 7 machines (single seed 13711):
- enc2_segment_knn_top5: hmean=0.5044, pAUC=0.5148, 2/5 bottleneck TPR@99n improve
- latent_map_segment_knn_top5: hmean=0.5059, pAUC=0.5185, 3/5 bottleneck TPR@99n improve
- bearing/valve sanity check passed. q99 not inflated.
- V43 Verdict: PASS

## 2. Configuration

- Seeds: 13711, 13721, 13731
- Experiment: transient_3view
- Model: viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0)
- Training: source normal only, epochs=50, bs=32, seed fixed
- Scoring: k=5, top_percent=5%

## 3. Seed-Level Aggregate (7 machines, mean hmean/pAUC)

| Seed | Score mode | Mean hmean | Mean pAUC |
|------|------------|------------|-----------|
| 13711 | latent_summary_knn             | 0.4981 | 0.4944 |
| 13711 | enc2_segment_knn_top5          | 0.5044 | 0.5148 |
| 13711 | latent_map_segment_knn_top5    | 0.5059 | 0.5185 |
| 13721 | latent_summary_knn             | 0.4973 | 0.4923 |
| 13721 | enc2_segment_knn_top5          | 0.5029 | 0.5094 |
| 13721 | latent_map_segment_knn_top5    | 0.5019 | 0.5062 |
| 13731 | latent_summary_knn             | 0.4975 | 0.4929 |
| 13731 | enc2_segment_knn_top5          | 0.5040 | 0.5131 |
| 13731 | latent_map_segment_knn_top5    | 0.5020 | 0.5062 |

## 4. Aggregate Across Seeds (mean ± std)

| Score mode | Mean hmean ± std | Mean pAUC ± std |
|------------|------------------|-----------------|
| latent_summary_knn             | 0.4976 ± 0.0003 | 0.4932 ± 0.0009 |
| enc2_segment_knn_top5          | 0.5038 ± 0.0006 | 0.5124 ± 0.0023 |
| latent_map_segment_knn_top5    | 0.5033 ± 0.0019 | 0.5103 ± 0.0058 |

## 5. Per-Machine (mean across 3 seeds)

| Machine | Score mode | Mean hmean | Mean pAUC |
|---------|------------|------------|-----------|
| bearing    | latent_summary_knn             | 0.5016 | 0.5047 |
| bearing    | enc2_segment_knn_top5          | 0.4974 | 0.4923 |
| bearing    | latent_map_segment_knn_top5    | 0.4951 | 0.4856 |
| fan        | latent_summary_knn             | 0.4967 | 0.4904 |
| fan        | enc2_segment_knn_top5          | 0.5008 | 0.5023 |
| fan        | latent_map_segment_knn_top5    | 0.5056 | 0.5174 |
| gearbox    | latent_summary_knn             | 0.4995 | 0.4988 |
| gearbox    | enc2_segment_knn_top5          | 0.5118 | 0.5374 |
| gearbox    | latent_map_segment_knn_top5    | 0.5049 | 0.5151 |
| slider     | latent_summary_knn             | 0.5021 | 0.5065 |
| slider     | enc2_segment_knn_top5          | 0.5005 | 0.5014 |
| slider     | latent_map_segment_knn_top5    | 0.5047 | 0.5144 |
| toycar     | latent_summary_knn             | 0.4956 | 0.4872 |
| toycar     | enc2_segment_knn_top5          | 0.4937 | 0.4816 |
| toycar     | latent_map_segment_knn_top5    | 0.5027 | 0.5084 |
| toytrain   | latent_summary_knn             | 0.4966 | 0.4898 |
| toytrain   | enc2_segment_knn_top5          | 0.5176 | 0.5577 |
| toytrain   | latent_map_segment_knn_top5    | 0.5039 | 0.5119 |
| valve      | latent_summary_knn             | 0.4913 | 0.4747 |
| valve      | enc2_segment_knn_top5          | 0.5047 | 0.5144 |
| valve      | latent_map_segment_knn_top5    | 0.5062 | 0.5191 |

## 6. Bottleneck TPR@95n / TPR@99n (mean across 3 seeds)

| Machine | Score mode | TPR@95n | TPR@99n | q99 |
|---------|------------|---------|---------|-----|
| gearbox    | latent_summary_knn             | 0.0533 | 0.0167 | 1.0822 |
| gearbox    | enc2_segment_knn_top5          | 0.1400 | 0.0633 | 0.3915 |
| gearbox    | latent_map_segment_knn_top5    | 0.0767 | 0.0433 | 0.5423 |
| slider     | latent_summary_knn             | 0.0667 | 0.0300 | 1.0594 |
| slider     | enc2_segment_knn_top5          | 0.0467 | 0.0167 | 0.4112 |
| slider     | latent_map_segment_knn_top5    | 0.0767 | 0.0367 | 0.5440 |
| toycar     | latent_summary_knn             | 0.0233 | 0.0033 | 1.0929 |
| toycar     | enc2_segment_knn_top5          | 0.0067 | 0.0000 | 0.3532 |
| toycar     | latent_map_segment_knn_top5    | 0.0733 | 0.0433 | 0.4958 |
| toytrain   | latent_summary_knn             | 0.0300 | 0.0100 | 1.0764 |
| toytrain   | enc2_segment_knn_top5          | 0.1800 | 0.0933 | 0.3624 |
| toytrain   | latent_map_segment_knn_top5    | 0.0900 | 0.0300 | 0.5065 |
| fan        | latent_summary_knn             | 0.0400 | 0.0100 | 0.9892 |
| fan        | enc2_segment_knn_top5          | 0.0567 | 0.0200 | 0.4489 |
| fan        | latent_map_segment_knn_top5    | 0.0800 | 0.0300 | 0.5981 |

## 7. TPR@99n Improvement vs latent_summary_knn

**enc2_segment_knn_top5**: 3/5 improve [PASS]
  - gearbox: 0.0167 -> 0.0633
  - slider: 0.0300 -> 0.0167
  - toycar: 0.0033 -> 0.0000
  - toytrain: 0.0100 -> 0.0933
  - fan: 0.0100 -> 0.0200
**latent_map_segment_knn_top5**: 5/5 improve [PASS]
  - gearbox: 0.0167 -> 0.0433
  - slider: 0.0300 -> 0.0367
  - toycar: 0.0033 -> 0.0433
  - toytrain: 0.0100 -> 0.0300
  - fan: 0.0100 -> 0.0300

## 8. Bearing / Valve Sanity Check (mean across 3 seeds)

| Machine | Score mode | Mean hmean | Collapse? |
|---------|------------|------------|-----------|
| bearing    | latent_summary_knn             | 0.5016 | OK |
| bearing    | enc2_segment_knn_top5          | 0.4974 | OK |
| bearing    | latent_map_segment_knn_top5    | 0.4951 | OK |
| valve      | latent_summary_knn             | 0.4913 | OK |
| valve      | enc2_segment_knn_top5          | 0.5047 | OK |
| valve      | latent_map_segment_knn_top5    | 0.5062 | OK |

## 9. Normal q99 Inflation Check (mean across seeds/machines)

| Score mode | mean q99 |
|------------|----------|
| latent_summary_knn             | 1.0440 |
| enc2_segment_knn_top5          | 0.3937 |
| latent_map_segment_knn_top5    | 0.5466 |

## 10. Final Verdict

### latent_map_segment_knn_top5 (primary candidate)
- hmean >= latent_summary (0.4976): 0.5033 [PASS]
- pAUC >= latent_summary (0.4932): 0.5103 [PASS]
- >= 2 bottleneck TPR@99n improve: 5/5 [PASS]
- bearing/valve not collapsed: [PASS]
- q99 not inflated: [PASS]
- **Overall: PASS**

---

All models trained under fixed V40 protocol (seed=13711, 13721, 13731). 
No k or top_percent tuning. No per-machine layer selection. 
No score fusion. No frontend parameter changes.
