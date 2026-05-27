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
| bearing | 990.0 | 99.0 | 495.0 |
| fan | 990.0 | 99.0 | 495.0 |
| gearbox | 990.0 | 99.0 | 495.0 |
| slider | 990.0 | 99.0 | 495.0 |
| toycar | 990.0 | 99.0 | 495.0 |
| toytrain | 990.0 | 99.0 | 495.0 |
| valve | 990.0 | 99.0 | 495.0 |

## 3. Normal-Tail Composition Table

| machine | mean q95 | mean q99 | mean normal-tail count | mean hard-normal-like ratio in q95-q99 | mean hard-normal-like ratio near q99 |
|---|---:|---:|---:|---:|---:|
| bearing | 0.156983 | 0.166041 | 4.0 | 0.8333 | 1.0000 |
| fan | 0.229545 | 0.238231 | 4.0 | 0.7500 | 1.0000 |
| gearbox | 0.116848 | 0.128176 | 4.0 | 0.9167 | 0.8333 |
| slider | 0.126665 | 0.149407 | 4.0 | 0.5000 | 0.6667 |
| toycar | 0.082653 | 0.087029 | 4.0 | 0.5000 | 0.2222 |
| toytrain | 0.092780 | 0.119308 | 4.0 | 1.0000 | 1.0000 |
| valve | 0.197595 | 0.215004 | 4.0 | 0.5000 | 0.6667 |

## 4. Hard-Normal vs Easy-Normal Latent Statistics

| machine | hard centroid dist | easy centroid dist | hard kNN score | easy kNN score | hard intra-var | easy intra-var | domain composition |
|---|---:|---:|---:|---:|---:|---:|---|
| bearing | 4.467469 | 2.126704 | 0.130242 | 0.080045 | 13.735158 | 4.199304 | source_only |
| fan | 2.910019 | 2.762577 | 0.104103 | 0.072859 | 7.338168 | 7.950520 | source_only |
| gearbox | 5.035747 | 2.483297 | 0.104077 | 0.064315 | 17.444673 | 7.496882 | source_only |
| slider | 4.052079 | 4.428168 | 0.104997 | 0.079609 | 15.348551 | 26.357027 | source_only |
| toycar | 1.037523 | 0.781928 | 0.068494 | 0.055315 | 1.017013 | 0.614583 | source_only |
| toytrain | 4.162119 | 1.870662 | 0.069660 | 0.048742 | 9.346876 | 2.320689 | source_only |
| valve | 5.512254 | 7.085626 | 0.161022 | 0.105071 | 27.092008 | 52.126502 | source_only |

## 5. Missed Anomaly Overlap Table

| machine | mean missed count | mean hard-NN overlap ratio | mean dist to hard centroid | mean dist to easy centroid | mean closer-to-hard ratio |
|---|---:|---:|---:|---:|---:|
| bearing | 78.7 | 0.0755 | 3.893644 | 2.667442 | 0.1884 |
| fan | 87.3 | 0.0908 | 3.418021 | 3.494304 | 0.7242 |
| gearbox | 93.0 | 0.0143 | 4.145757 | 3.070728 | 0.2704 |
| slider | 98.3 | 0.0339 | 4.770429 | 4.866799 | 0.5060 |
| toycar | 97.0 | 0.0104 | 0.893589 | 0.848337 | 0.2857 |
| toytrain | 96.7 | 0.0484 | 3.466139 | 2.372514 | 0.4034 |
| valve | 93.0 | 0.0359 | 6.066053 | 6.786236 | 0.6773 |

## 6. Augmentation Consistency Table

Checkpoint-based augmentation consistency could not be computed because checkpoint paths were missing.

## 7. Theoretical Tail Compression Proxy

Fixed proxy compression ratio on hard-normal-like tail samples: `0.10`

| machine | base q99 | proxy q99 | delta q99 | base TPR@99n | proxy TPR@99n | delta TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.166041 | 0.159686 | -0.006355 | 0.2133 | 0.2633 | +0.0500 |
| fan | 0.238231 | 0.230828 | -0.007403 | 0.1267 | 0.1800 | +0.0533 |
| gearbox | 0.128176 | 0.120937 | -0.007239 | 0.0700 | 0.0933 | +0.0233 |
| slider | 0.149407 | 0.143427 | -0.005980 | 0.0167 | 0.0400 | +0.0233 |
| toycar | 0.087029 | 0.086874 | -0.000156 | 0.0300 | 0.0333 | +0.0033 |
| toytrain | 0.119308 | 0.107508 | -0.011800 | 0.0333 | 0.0467 | +0.0133 |
| valve | 0.215004 | 0.206959 | -0.008045 | 0.0700 | 0.1200 | +0.0500 |

## 8. Final Diagnostic Conclusion

`unclear`

Missing-path policy:

| seed | machine | kind | path |
|---|---|---|---|
| 13711 | bearing | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/bearing_full_5view_seed13711.pt |
| 13711 | fan | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/fan_full_5view_seed13711.pt |
| 13711 | gearbox | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/gearbox_full_5view_seed13711.pt |
| 13711 | slider | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/slider_full_5view_seed13711.pt |
| 13711 | toycar | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/toycar_full_5view_seed13711.pt |
| 13711 | toytrain | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/toytrain_full_5view_seed13711.pt |
| 13711 | valve | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13711/models/valve_full_5view_seed13711.pt |
| 13721 | bearing | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/bearing_full_5view_seed13721.pt |
| 13721 | fan | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/fan_full_5view_seed13721.pt |
| 13721 | gearbox | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/gearbox_full_5view_seed13721.pt |
| 13721 | slider | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/slider_full_5view_seed13721.pt |
| 13721 | toycar | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/toycar_full_5view_seed13721.pt |
| 13721 | toytrain | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/toytrain_full_5view_seed13721.pt |
| 13721 | valve | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13721/models/valve_full_5view_seed13721.pt |
| 13731 | bearing | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/bearing_full_5view_seed13731.pt |
| 13731 | fan | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/fan_full_5view_seed13731.pt |
| 13731 | gearbox | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/gearbox_full_5view_seed13731.pt |
| 13731 | slider | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/slider_full_5view_seed13731.pt |
| 13731 | toycar | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/toycar_full_5view_seed13731.pt |
| 13731 | toytrain | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/toytrain_full_5view_seed13731.pt |
| 13731 | valve | checkpoint | results/v37_teacher_student_pretrain_seedpanel/seed_13731/models/valve_full_5view_seed13731.pt |

This script performs no training, no optimizer step, no score tuning, and no per-machine method selection.
