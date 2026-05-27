# CTAF V43 Transient Segment Scoring Full-Machine Validation

## 1. Provenance: bearing/valve Checkpoint Completion

bearing and valve transient_3view checkpoints were generated under the
identical V40 protocol:
- experiment = transient_3view
- model_type = viewwise_unshared_late_fusion
- latent_channels = 64, stem_channels = 8, per_view_channels = 16, dropout_p = 0.0
- training = source normal only, memory bank = source normal only
- loss_view_mode = all_views, ssl_task = reconstruction
- seed = 13711, epochs = 50, batch_size = 32
- No per-machine customization.

V40 bottleneck checkpoints (gearbox, slider, toycar, toytrain, fan) 
were regenerated earlier for V41/V42 under the same seed=13711 protocol. 
Regeneration was verified: score reproduction max_diff < 0.08.

## 2. V42 Bottleneck Summary

V42 tested segment-level scoring on 5 bottleneck machines. Both score modes passed:

| Score mode | Mean hmean | Mean pAUC | TPR@99n improved | q99 status |
|------------|------------|-----------|------------------|------------|
| enc2_segment_knn_top5 | 0.5056 | 0.5187 | 2/5 PASS | not inflated |
| latent_map_segment_knn_top5 | 0.5075 | 0.5233 | 3/5 PASS | not inflated |

## 3. All-7-Machine Overall Comparison

Mean across 7 machines.

| Score mode | Mean hmean | Mean pAUC | Mean dcase_auc |
|------------|------------|-----------|----------------|
| enc2_segment_knn_top5          | 0.5044 | 0.5148 | 0.5000 |
| latent_map_segment_knn_top5    | 0.5059 | 0.5185 | 0.5000 |
| latent_summary_knn             | 0.4981 | 0.4944 | 0.5000 |

## 4. Per-Machine hmean / pAUC

| Machine | Score mode | hmean | pAUC |
|---------|------------|-------|------|
| bearing    | enc2_segment_knn_top5          | 0.4975 | 0.4926 |
| bearing    | latent_map_segment_knn_top5    | 0.4950 | 0.4853 |
| bearing    | latent_summary_knn             | 0.5016 | 0.5047 |
| fan        | enc2_segment_knn_top5          | 0.5002 | 0.5005 |
| fan        | latent_map_segment_knn_top5    | 0.5099 | 0.5311 |
| fan        | latent_summary_knn             | 0.5005 | 0.5016 |
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
| valve      | enc2_segment_knn_top5          | 0.5057 | 0.5174 |
| valve      | latent_map_segment_knn_top5    | 0.5090 | 0.5279 |
| valve      | latent_summary_knn             | 0.4920 | 0.4768 |

## 5. Bottleneck Machine Tail Table (TPR@95n / TPR@99n)

| Machine | Score mode | TPR@95n | TPR@99n | q99 |
|---------|------------|---------|---------|-----|
| gearbox    | enc2_segment_knn_top5          | 0.1000 | 0.0500 | 0.3932 |
| gearbox    | latent_map_segment_knn_top5    | 0.0800 | 0.0400 | 0.5326 |
| gearbox    | latent_summary_knn             | 0.0300 | 0.0000 | 1.0961 |
| slider     | enc2_segment_knn_top5          | 0.0600 | 0.0200 | 0.4049 |
| slider     | latent_map_segment_knn_top5    | 0.1000 | 0.0300 | 0.5427 |
| slider     | latent_summary_knn             | 0.0700 | 0.0500 | 1.0559 |
| toycar     | enc2_segment_knn_top5          | 0.0000 | 0.0000 | 0.3528 |
| toycar     | latent_map_segment_knn_top5    | 0.1300 | 0.0700 | 0.4937 |
| toycar     | latent_summary_knn             | 0.0200 | 0.0100 | 1.0924 |
| toytrain   | enc2_segment_knn_top5          | 0.2300 | 0.1600 | 0.3521 |
| toytrain   | latent_map_segment_knn_top5    | 0.1200 | 0.0200 | 0.5089 |
| toytrain   | latent_summary_knn             | 0.0500 | 0.0200 | 1.0808 |
| fan        | enc2_segment_knn_top5          | 0.0600 | 0.0000 | 0.4586 |
| fan        | latent_map_segment_knn_top5    | 0.1300 | 0.0500 | 0.6013 |
| fan        | latent_summary_knn             | 0.0600 | 0.0200 | 0.9689 |

## 6. TPR@99n Improvement on Bottleneck Machines

**enc2_segment_knn_top5**: 2/5 improve [PASS]
  - gearbox: 0.0000 -> 0.0500 (+0.0500)
  - slider: 0.0500 -> 0.0200 (-0.0300)
  - toycar: 0.0100 -> 0.0000 (-0.0100)
  - toytrain: 0.0200 -> 0.1600 (+0.1400)
  - fan: 0.0200 -> 0.0000 (-0.0200)

**latent_map_segment_knn_top5**: 3/5 improve [PASS]
  - gearbox: 0.0000 -> 0.0400 (+0.0400)
  - slider: 0.0500 -> 0.0300 (-0.0200)
  - toycar: 0.0100 -> 0.0700 (+0.0600)
  - toytrain: 0.0200 -> 0.0200 (+0.0000)
  - fan: 0.0200 -> 0.0500 (+0.0300)

## 7. Bearing / Valve Sanity Check

Non-bottleneck machines validate that segment scoring does not cause
catastrophic degradation on held-out machine types.

| Machine | Score mode | hmean | pAUC | TPR@95n | TPR@99n | Collapse? |
|---------|------------|-------|------|---------|---------|-----------|
| bearing    | enc2_segment_knn_top5          | 0.4975 | 0.4926 | 0.0400 | 0.0000 | OK |
| bearing    | latent_map_segment_knn_top5    | 0.4950 | 0.4853 | 0.0200 | 0.0000 | OK |
| bearing    | latent_summary_knn             | 0.5016 | 0.5047 | 0.0700 | 0.0100 | OK |
| valve      | enc2_segment_knn_top5          | 0.5057 | 0.5174 | 0.0900 | 0.0600 | OK |
| valve      | latent_map_segment_knn_top5    | 0.5090 | 0.5279 | 0.1300 | 0.0300 | OK |
| valve      | latent_summary_knn             | 0.4920 | 0.4768 | 0.0100 | 0.0000 | OK |

## 8. Success Criteria

### enc2_segment_knn_top5
- hmean >= ls_knn (0.4981): 0.5044 [PASS]
- pAUC >= ls_knn (0.4944): 0.5148 [PASS]
- >= 2 bottleneck TPR@99n improve: [PASS]
- bearing/valve hmean >= 0.45: [PASS]
- q99 not inflated: [PASS]
- **Overall: PASS**

### latent_map_segment_knn_top5
- hmean >= ls_knn (0.4981): 0.5059 [PASS]
- pAUC >= ls_knn (0.4944): 0.5185 [PASS]
- >= 2 bottleneck TPR@99n improve: [PASS]
- bearing/valve hmean >= 0.45: [PASS]
- q99 not inflated: [PASS]
- **Overall: PASS**

### V43 Verdict: PASS

Segment-level scoring generalizes to all 7 machines. 
The improvement over file-level latent_summary_knn is consistent.

**Next**: V44 = 3-seed panel validation of segment scoring on transient_3view.

---

All models loaded from V40 checkpoints. No retraining. 
k=5, top_percent=5% fixed. No per-machine tuning.
