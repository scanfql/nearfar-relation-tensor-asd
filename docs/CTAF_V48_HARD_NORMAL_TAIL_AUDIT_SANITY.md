# CTAF V48 Hard-Normal Tail Audit

## 1. Stage 5 Summary

- mechanism 3-view frontends cannot replace `full_5view`
- current mainline remains `full_5view + latent_summary_knn`

## 2. Hard-Normal Definition

- hard-normal main ratio: `0.10` of train-normal by highest `latent_summary_knn` score
- easy-normal main definition: `lowest 50%` of train-normal by `latent_summary_knn` score
- anomaly labels are not used in hard-normal definition

| machine | mean train count | mean hard-normal count | mean easy-normal count |
|---|---:|---:|---:|

## 3. Normal-Tail Composition Table

| machine | mean q95 | mean q99 | mean normal-tail count | mean hard-normal-like ratio in q95-q99 | mean hard-normal-like ratio near q99 |
|---|---:|---:|---:|---:|---:|

## 4. Hard-Normal vs Easy-Normal Latent Statistics

| machine | hard centroid dist | easy centroid dist | hard kNN score | easy kNN score | hard intra-var | easy intra-var | domain composition |
|---|---:|---:|---:|---:|---:|---:|---|

## 5. Missed Anomaly Overlap Table

| machine | mean missed count | mean hard-NN overlap ratio | mean dist to hard centroid | mean dist to easy centroid | mean closer-to-hard ratio |
|---|---:|---:|---:|---:|---:|

## 6. Augmentation Consistency Table

Checkpoint-based augmentation consistency could not be computed because checkpoint paths were missing.

## 7. Theoretical Tail Compression Proxy

Fixed proxy compression ratio on hard-normal-like tail samples: `0.10`

| machine | base q99 | proxy q99 | delta q99 | base TPR@99n | proxy TPR@99n | delta TPR@99n |
|---|---:|---:|---:|---:|---:|---:|

## 8. Final Diagnostic Conclusion

`unclear`

Missing-path policy:

| seed | machine | kind | path |
|---|---|---|---|
| 13711 | gearbox | score_file | results/v37_teacher_student_pretrain_seedpanel/seed_13711/seed_13711/scores/gearbox_full_5view_latent_summary_knn_scores.npz |

This script performs no training, no optimizer step, no score tuning, and no per-machine method selection.
