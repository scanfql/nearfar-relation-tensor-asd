# CTAF Stage19G Fan LORA LDN Failure Analysis Implementation

Stage19G is an analysis-only pass over existing Stage18 and Stage19F outputs.
It does not train, forward, extract new embeddings, or score new candidates.

## Inputs

- Stage19F summary and verdict
- Stage19F adapted embedding NPZ
- Stage19F comparison CSV
- Stage18 fan score NPZs
- Stage16D min-clamp mean embedding NPZ

Important detail:

- the Stage18 clean train geometry is reconstructed from `results/stage16d_minclamp_mean_full7/embeddings/fan_minclamp_mean_embeddings.npz`
- the Stage18 score NPZs provide the frozen baseline and frozen LDN test scores
- the Stage19F embedding NPZ provides the adapted train/test embeddings for the anchor-LoRA route

## Modes

### `--mode check`

Checks only:

- Stage19F summary exists
- Stage19F comparison exists
- Stage19F verdict exists
- Stage19F embeddings exist
- Stage18 baseline and LDN score NPZs exist
- Stage16D embedding NPZ exists
- output root writable
- imports: `numpy`, `scipy`, `sklearn`, `csv`, `json`

No diagnostics, no recomputation, no model load, no forward, no extraction, no scoring.

### `--mode analyze`

Reads existing embeddings and scores only, then writes analysis tables:

- `stage18_vs_stage19f_metric_recap.csv`
- `score_distribution_comparison.csv`
- `rho_distribution_comparison.csv`
- `ldn_transform_shift.csv`
- `normal_tail_comparison.csv`
- `anomaly_separation_comparison.csv`
- `nearest_reference_density_diagnostics.csv`
- `score_delta_correlation.csv`
- `failure_interpretation_table.csv`
- `ldn_failure_summary.json`
- `stage19h_recommendation.json`
- `docs/CTAF_STAGE19G_FAN_LORA_LDN_FAILURE_ANALYSIS_AUDIT.md`

## Analysis focus

- compare Stage18 vs Stage19F train-density geometry
- recompute K1 reference density on the stored train embeddings
- inspect how the LDN transformation shifts score distributions
- check whether the anomaly margin shrinks after density subtraction
- decide whether Stage19H is justified

## Recommendation

If the baseline route remains close to Stage18 while the LDN route still fails, Stage19H is the next conservative step:

- `L = L_recon_masked + lambda_anchor * L_anchor_embedding + lambda_dist * L_pairwise_distance_preservation`
- default `lambda_anchor = 1.0`
- default `lambda_dist = 0.1`
- trainable scope: `last_block_attention_lora`
- machine: `fan`
- Stage19H trains only
- Stage19I evaluates
