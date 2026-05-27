# CTAF V45 Segment Scoring Transfer Audit

## 1. V44 Summary

V44 3-seed panel confirmed that transient_3view + latent_map_segment_knn_top5 is stable:
- hmean = 0.5033 ± 0.0019, pAUC = 0.5103 ± 0.0058
- 5/5 bottleneck TPR@99n improve; q99 not inflated
- But absolute hmean/pAUC still below V16/V37 mainline.
- V45 tests whether segment scoring transfers to stronger full_5view representations.

## 2. Configuration
- V16: full_5view AE, seed 13711
- V37: full_5view teacher-student pretraining, seeds 13711/13721/13731
- Model: viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0)
- Scoring: k=5, top_percent=5%
- No retraining; models regenerated under original V16/V37 protocol.

## 3. V16_full5view Comparison

### Aggregate (mean across seeds/machines)

| Score mode | Mean hmean ± std | Mean pAUC ± std |
|------------|------------------|-----------------|
| latent_summary_knn             | 0.5447 ± 0.0000 | 0.5404 ± 0.0000 |
| enc2_segment_knn_top5          | 0.5020 ± 0.0000 | 0.5081 ± 0.0000 |
| latent_map_segment_knn_top5    | 0.5298 ± 0.0000 | 0.5074 ± 0.0000 |

### Per-Machine (mean across seeds)

| Machine | Score mode | Mean hmean | Mean pAUC |
|---------|------------|------------|-----------|
| bearing    | latent_summary_knn             | 0.6238 | 0.6084 |
| bearing    | enc2_segment_knn_top5          | 0.4931 | 0.5189 |
| bearing    | latent_map_segment_knn_top5    | 0.5207 | 0.4921 |
| fan        | latent_summary_knn             | 0.5897 | 0.5932 |
| fan        | enc2_segment_knn_top5          | 0.5062 | 0.5168 |
| fan        | latent_map_segment_knn_top5    | 0.5055 | 0.5032 |
| gearbox    | latent_summary_knn             | 0.5389 | 0.5037 |
| gearbox    | enc2_segment_knn_top5          | 0.5338 | 0.5121 |
| gearbox    | latent_map_segment_knn_top5    | 0.5469 | 0.5205 |
| slider     | latent_summary_knn             | 0.5666 | 0.5263 |
| slider     | enc2_segment_knn_top5          | 0.5162 | 0.4911 |
| slider     | latent_map_segment_knn_top5    | 0.5299 | 0.5021 |
| toycar     | latent_summary_knn             | 0.4662 | 0.5016 |
| toycar     | enc2_segment_knn_top5          | 0.4892 | 0.5011 |
| toycar     | latent_map_segment_knn_top5    | 0.5438 | 0.5295 |
| toytrain   | latent_summary_knn             | 0.4978 | 0.5168 |
| toytrain   | enc2_segment_knn_top5          | 0.5196 | 0.5121 |
| toytrain   | latent_map_segment_knn_top5    | 0.5550 | 0.4989 |
| valve      | latent_summary_knn             | 0.5294 | 0.5326 |
| valve      | enc2_segment_knn_top5          | 0.4558 | 0.5047 |
| valve      | latent_map_segment_knn_top5    | 0.5072 | 0.5058 |

### Bottleneck TPR@95n / TPR@99n

| Machine | Score mode | TPR@95n | TPR@99n | q99 |
|---------|------------|---------|---------|-----|
| gearbox    | latent_summary_knn             | 0.0700 | 0.0400 | 0.1425 |
| gearbox    | enc2_segment_knn_top5          | 0.1000 | 0.0400 | 0.4735 |
| gearbox    | latent_map_segment_knn_top5    | 0.1200 | 0.0200 | 0.4341 |
| slider     | latent_summary_knn             | 0.1200 | 0.0400 | 0.1339 |
| slider     | enc2_segment_knn_top5          | 0.0400 | 0.0100 | 0.5298 |
| slider     | latent_map_segment_knn_top5    | 0.0500 | 0.0200 | 0.4684 |
| toycar     | latent_summary_knn             | 0.0500 | 0.0300 | 0.0783 |
| toycar     | enc2_segment_knn_top5          | 0.0700 | 0.0100 | 0.4733 |
| toycar     | latent_map_segment_knn_top5    | 0.1100 | 0.1000 | 0.4119 |
| toytrain   | latent_summary_knn             | 0.0900 | 0.0300 | 0.1161 |
| toytrain   | enc2_segment_knn_top5          | 0.0900 | 0.0000 | 0.5133 |
| toytrain   | latent_map_segment_knn_top5    | 0.0700 | 0.0100 | 0.4229 |
| fan        | latent_summary_knn             | 0.2300 | 0.2000 | 0.2198 |
| fan        | enc2_segment_knn_top5          | 0.1100 | 0.0300 | 0.5865 |
| fan        | latent_map_segment_knn_top5    | 0.0700 | 0.0200 | 0.4766 |

### Bearing / Valve Sanity

| Machine | Score mode | Mean hmean | Collapse? |
|---------|------------|------------|-----------|
| bearing    | latent_summary_knn             | 0.6238 | OK |
| bearing    | enc2_segment_knn_top5          | 0.4931 | WARNING |
| bearing    | latent_map_segment_knn_top5    | 0.5207 | OK |
| valve      | latent_summary_knn             | 0.5294 | OK |
| valve      | enc2_segment_knn_top5          | 0.4558 | WARNING |
| valve      | latent_map_segment_knn_top5    | 0.5072 | OK |

## 4. V37_full5view_TS Comparison

### Aggregate (mean across seeds/machines)

| Score mode | Mean hmean ± std | Mean pAUC ± std |
|------------|------------------|-----------------|
| latent_summary_knn             | 0.5502 ± 0.0042 | 0.5409 ± 0.0014 |
| enc2_segment_knn_top5          | 0.4979 ± 0.0042 | 0.5027 ± 0.0038 |
| latent_map_segment_knn_top5    | 0.5196 ± 0.0097 | 0.5033 ± 0.0035 |

### Per-Machine (mean across seeds)

| Machine | Score mode | Mean hmean | Mean pAUC |
|---------|------------|------------|-----------|
| bearing    | latent_summary_knn             | 0.6404 | 0.6137 |
| bearing    | enc2_segment_knn_top5          | 0.4805 | 0.5146 |
| bearing    | latent_map_segment_knn_top5    | 0.5436 | 0.5046 |
| fan        | latent_summary_knn             | 0.5695 | 0.5658 |
| fan        | enc2_segment_knn_top5          | 0.5087 | 0.5088 |
| fan        | latent_map_segment_knn_top5    | 0.4796 | 0.4905 |
| gearbox    | latent_summary_knn             | 0.5601 | 0.5214 |
| gearbox    | enc2_segment_knn_top5          | 0.5414 | 0.5111 |
| gearbox    | latent_map_segment_knn_top5    | 0.5009 | 0.5011 |
| slider     | latent_summary_knn             | 0.5404 | 0.5246 |
| slider     | enc2_segment_knn_top5          | 0.5208 | 0.4909 |
| slider     | latent_map_segment_knn_top5    | 0.5228 | 0.4949 |
| toycar     | latent_summary_knn             | 0.4836 | 0.5093 |
| toycar     | enc2_segment_knn_top5          | 0.4881 | 0.5000 |
| toycar     | latent_map_segment_knn_top5    | 0.5179 | 0.5061 |
| toytrain   | latent_summary_knn             | 0.5246 | 0.5175 |
| toytrain   | enc2_segment_knn_top5          | 0.5018 | 0.4954 |
| toytrain   | latent_map_segment_knn_top5    | 0.5469 | 0.5093 |
| valve      | latent_summary_knn             | 0.5329 | 0.5340 |
| valve      | enc2_segment_knn_top5          | 0.4440 | 0.4982 |
| valve      | latent_map_segment_knn_top5    | 0.5253 | 0.5163 |

### Bottleneck TPR@95n / TPR@99n

| Machine | Score mode | TPR@95n | TPR@99n | q99 |
|---------|------------|---------|---------|-----|
| gearbox    | latent_summary_knn             | 0.0933 | 0.0667 | 0.1307 |
| gearbox    | enc2_segment_knn_top5          | 0.0800 | 0.0300 | 0.4895 |
| gearbox    | latent_map_segment_knn_top5    | 0.0700 | 0.0100 | 0.4480 |
| slider     | latent_summary_knn             | 0.1100 | 0.0367 | 0.1394 |
| slider     | enc2_segment_knn_top5          | 0.0367 | 0.0067 | 0.5291 |
| slider     | latent_map_segment_knn_top5    | 0.0300 | 0.0100 | 0.4785 |
| toycar     | latent_summary_knn             | 0.0700 | 0.0333 | 0.0842 |
| toycar     | enc2_segment_knn_top5          | 0.0467 | 0.0167 | 0.4530 |
| toycar     | latent_map_segment_knn_top5    | 0.0533 | 0.0400 | 0.4036 |
| toytrain   | latent_summary_knn             | 0.0967 | 0.0333 | 0.1137 |
| toytrain   | enc2_segment_knn_top5          | 0.0467 | 0.0033 | 0.5040 |
| toytrain   | latent_map_segment_knn_top5    | 0.0800 | 0.0233 | 0.4165 |
| fan        | latent_summary_knn             | 0.1900 | 0.1133 | 0.2364 |
| fan        | enc2_segment_knn_top5          | 0.0867 | 0.0200 | 0.5786 |
| fan        | latent_map_segment_knn_top5    | 0.0400 | 0.0067 | 0.4752 |

### Bearing / Valve Sanity

| Machine | Score mode | Mean hmean | Collapse? |
|---------|------------|------------|-----------|
| bearing    | latent_summary_knn             | 0.6404 | OK |
| bearing    | enc2_segment_knn_top5          | 0.4805 | WARNING |
| bearing    | latent_map_segment_knn_top5    | 0.5436 | OK |
| valve      | latent_summary_knn             | 0.5329 | OK |
| valve      | enc2_segment_knn_top5          | 0.4440 | WARNING |
| valve      | latent_map_segment_knn_top5    | 0.5253 | OK |

## 5. Final Verdict

### V37 + latent_map_segment_knn_top5
- hmean >= ls_knn (0.5502): 0.5196 (-0.0306) [FAIL]
- pAUC >= ls_knn (0.5409): 0.5033 (-0.0376) [FAIL]
- >= 2 bottleneck TPR@99n improve: 1/5 [FAIL]
- bearing/valve hmean OK: [PASS]
- **Overall: FAIL**

---

Models regenerated under original V16/V37 protocol (same seed, architecture, data). 
No retraining with modifications. k=5, top_percent=5% fixed. No per-machine tuning.
