# CTAF V55 Frozen BEATs Representation Audit

## 1. V54 prereg summary

- Stage 10 tests a new external representation hypothesis
- no fine-tuning, no LoRA, no adapter, no DLCL
- backend is fixed kNN on frozen BEATs utterance embeddings

## 2. BEATs checkpoint path and provenance

- checkpoint path: `/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt`
- provenance: official BEATs 90M AudioSet-pretrained checkpoint found on server

## 3. Confirmation that BEATs is frozen

- encoder loaded in eval mode
- `torch.no_grad` used for feature extraction
- no parameter updates
- no classifier head training
- no pooling head training

## 4. Embedding extraction protocol

- input: raw waveform
- resample: 16 kHz via existing waveform loader
- feature extraction: BEATs internal fbank preprocessing
- embedding: final-layer frame embeddings
- pooling: time mean pooling
- backend: cosine-distance kNN with `k=5` on source train-normal memory only

## 5. Overall comparison

| system | mean hmean | mean pAUC | mean KS |
|---|---:|---:|---:|
| V37 full_5view latent_summary_knn | 0.563084 | 0.550451 | 0.3143 |
| logmel_only latent_summary_knn | 0.534091 | 0.520150 | 0.4057 |
| frozen BEATs + kNN | 0.578557 | 0.538797 | 0.508571 |

## 6. Per-machine hmean / pAUC / KS

| machine | hmean | pAUC | KS |
|---|---:|---:|---:|
| bearing | 0.533344 | 0.566316 | 0.240000 |
| fan | 0.507386 | 0.482632 | 1.000000 |
| gearbox | 0.575497 | 0.541579 | 0.400000 |
| slider | 0.578752 | 0.498947 | 0.280000 |
| toycar | 0.553766 | 0.532632 | 0.780000 |
| toytrain | 0.616360 | 0.509474 | 0.660000 |
| valve | 0.684797 | 0.640000 | 0.200000 |

## 7. Tail table

| machine | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |
|---|---:|---:|---:|---:|---:|---:|
| bearing | 0.4504 | 0.5003 | 0.3534 | 0.4366 | 0.2200 | 0.0700 |
| fan | 0.8503 | 0.8573 | 0.6005 | 0.7367 | 0.0000 | 0.0000 |
| gearbox | 0.4616 | 0.5058 | 0.3043 | 0.4210 | 0.1400 | 0.0400 |
| slider | 0.4399 | 0.4983 | 0.3394 | 0.3900 | 0.0600 | 0.0200 |
| toycar | 0.3504 | 0.3884 | 0.2676 | 0.3094 | 0.1400 | 0.0400 |
| toytrain | 0.3304 | 0.3912 | 0.2120 | 0.2503 | 0.0500 | 0.0000 |
| valve | 0.1663 | 0.1915 | 0.1600 | 0.1787 | 0.3700 | 0.1800 |

## 8. q99 inflation check

- machines with >5% q99 inflation relative to V37 baseline: `bearing, fan, gearbox, slider, toycar, toytrain`

## 9. Final verdict

`V55 negative`

Success gates:

- mean hmean >= V37 - 0.003: `PASS`
- mean pAUC >= V37: `FAIL`
- >=2 bottleneck machines improve TPR@99n: `1/5`
- bearing / valve do not collapse: `FAIL`
- toycar does not clearly degrade: `PASS`
- normal q99 does not clearly inflate: `FAIL`

No BEATs fine-tuning occurred. No protocol substitutions were made. No V56 was opened.
