# CTAF Stage19H Fan Anchor Geometry LoRA Sanity Audit

## Summary

Stage19H is a fan-only training scaffold. It starts from clean EAT, injects LoRA only into last-block attention `qkv/proj`, and adds a fixed pairwise distance-preservation term to the Stage19E anchor-reconstruction objective.

## Guardrails

- no ASD scoring in Stage19H
- no full7
- no selector
- no fusion
- no anomaly-label training
- no test-label checkpoint selection
- no Stage19B continuation
- no Stage19E continuation

## Fixed Loss

`L = L_recon_masked + lambda_anchor * L_anchor_embedding + lambda_dist * L_pairwise_distance_preservation`

- `lambda_anchor = 1.0`
- `lambda_dist = 0.1`

## Training Verdict

Stage19I will evaluate the resulting adapted embeddings with the fixed backend after training is complete.
