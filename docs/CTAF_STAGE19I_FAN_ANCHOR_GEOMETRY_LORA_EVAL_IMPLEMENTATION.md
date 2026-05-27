# CTAF Stage19I Fan Anchor Geometry LoRA Evaluation Implementation

## Goal

Evaluate the Stage19H fan anchor + distance-geometry LoRA checkpoint with the fixed fan backend.

## Fixed Protocol

- machine: `fan`
- adapted checkpoint: `results/stage19h_fan_anchor_geometry_lora_sanity/checkpoints/fan_stage19h_anchor_geometry_lora_best.pt`
- preprocess:
  - 16 kHz
  - 128-bin Kaldi fbank
  - frame shift 10 ms
  - target length 1024
  - normalization: `(mel - (-4.268)) / (4.569 * 2)`
- EAT input shape: `[B, 1, T, F]`
- LoRA targets:
  - `model.blocks.11.attn.qkv`
  - `model.blocks.11.attn.proj`
- scoring candidates:
  - `adapted_baseline_knn_k5`
  - `adapted_ldn_diff_K1_scorek5`

## Gates

- minimum pass: adapted LDN hmean `> 0.5527191284766243`
- strong pass: adapted LDN hmean `>= 0.5820070015781968`
- hard fail: adapted LDN hmean `<= 0.5527191284766243`
- representation guardrail: adapted baseline hmean must stay within `0.02` of the Stage18 baseline

## Modes

1. `--mode check`
   - checks paths, inventory, and imports only
   - no model load, no forward, no extraction, no scoring

2. `--mode inspect_checkpoint`
   - loads clean EAT
   - injects the fixed LoRA scaffold
   - loads the Stage19H checkpoint
   - audits checkpoint and trainable boundary details

3. `--mode extract_adapted_embeddings`
   - loads the adapted model
   - extracts fan train/test embeddings
   - writes the embedding NPZ for later scoring

4. `--mode score_adapted_fan`
   - reads embeddings only
   - computes `adapted_baseline_knn_k5` and `adapted_ldn_diff_K1_scorek5`
   - writes comparison tables and verdict

## References

- Stage18 baseline hmean: `0.5820070015781968`
- Stage18 LDN hmean: `0.5527191284766243`
- Stage19F anchor-LoRA LDN hmean: `0.5501054732664112`

Stage19I is fan-only and does not use selector, fusion, oracle, or test-label checkpoint selection.
