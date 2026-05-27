# CTAF V59A Score-File Schema Audit

## 1. Checked Files

| Machine | Path | Keys | train_scores | test_scores | y_test |
|---------|------|------|-------------|------------|--------|
| bearing    | .../bearing_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |
| fan        | .../fan_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |
| gearbox    | .../gearbox_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |
| slider     | .../slider_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |
| toycar     | .../toycar_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |
| toytrain   | .../toytrain_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |
| valve      | .../valve_full_5view_latent_summary_knn_scores.npz | 24 keys | (990,) | (200,) | (200,) |

## 2. Key Availability per Machine

| Machine | d_test | source | target | target_train_normal |
|---------|--------|--------|--------|---------------------|
| bearing    | YES | 100 | 100 | NO |
| fan        | YES | 100 | 100 | NO |
| gearbox    | YES | 100 | 100 | NO |
| slider     | YES | 100 | 100 | NO |
| toycar     | YES | 100 | 100 | NO |
| toytrain   | YES | 100 | 100 | NO |
| valve      | YES | 100 | 100 | NO |

d_test metadata: ALL present
target train-normal: does NOT exist (V37 is source-only)

## 3. Candidate Availability

| Machine | C0 raw | C1 zscore | C2 robust | C3 domain | C4 min | C5 fallback | C3/C4 blocker |
|---------|--------|-----------|-----------|-----------|--------|-------------|---------------|
| bearing    | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |
| fan        | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |
| gearbox    | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |
| slider     | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |
| toycar     | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |
| toytrain   | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |
| valve      | available | available | available | unavailable | unavailable | available | no target train-normal reference (V37 is source-only) |

C3 domain calibration: 0/7 available
C4 min(source,target): 0/7 available

## 4. Final Verdict

**V59A limited: d_test exists but C3/C4 unavailable (no target train-normal reference). Only rank-preserving candidates available. Full UACS unlikely to improve AUC/pAUC.**

---

No UACS selection performed. No anomaly labels used for selection. 
No score modification. No training.
