# CTAF Stage19H Fan Anchor Geometry LoRA Sanity Implementation

## Goal

Train a conservative fan-only LoRA adaptation scaffold that preserves Stage18F geometry better than masked reconstruction alone.

## Fixed Protocol

- machine: `fan`
- training data: `source train-normal only`
- split: `90/10 by seed 13711`
- teacher: frozen clean EAT
- student: clean EAT with LoRA only on last block attention qkv/proj
- LoRA targets:
  - `model.blocks.11.attn.qkv`
  - `model.blocks.11.attn.proj`
- LoRA rank/alpha/dropout: `4 / 8 / 0.05`
- reconstruction head: same as Stage19E
- preprocessing:
  - 16 kHz
  - 128-bin Kaldi fbank
  - frame shift 10 ms
  - target length 1024
  - norm_mean = `-4.268`
  - norm_std = `4.569`
  - mel normalization: `(mel - norm_mean) / (norm_std * 2)`

## Loss

`L = L_recon_masked + lambda_anchor * L_anchor_embedding + lambda_dist * L_pairwise_distance_preservation`

- `mask_ratio = 0.30`
- `lambda_anchor = 1.0`
- `lambda_dist = 0.1`

Definitions:

- `L_recon_masked`: masked-position MSE on normalized fbank reconstruction
- `L_anchor_embedding`: mean `1 - cosine(student_embedding, teacher_embedding)`
- `L_pairwise_distance_preservation`: MSE between teacher and student pairwise chord-distance matrices, excluding diagonal

Numerical safety:

- embedding normalization uses `eps = 1e-12`
- pairwise chord distance uses clamped squared distance and `sqrt(d2 + 1e-12)`
- any non-finite loss raises an error and writes a diagnostic JSON when possible

## Modes

1. `--mode check`
   - checks data root, clean EAT checkpoint, fan inventory, Stage19G recommendation, imports, and output-root writability
   - no model load, no forward, no training, no extraction, no scoring

2. `--mode inspect_lora`
   - loads clean EAT
   - injects LoRA only into last-block attention qkv/proj
   - freezes all base parameters
   - creates reconstruction head
   - audits trainable parameters

3. `--mode debug_one_batch`
   - loads clean EAT and the fixed Stage19H LoRA scaffold
   - runs exactly one train batch forward/loss computation
   - writes a loss diagnostic JSON with embedding norm and pairwise-distance stats
   - no parameter update

4. `--mode train_fan_anchor_geometry_lora`
   - trains on fan source train-normal only
   - validates on held-out train-normal split
   - saves best checkpoint by lowest validation total loss
   - writes train log with total/recon/anchor/pairwise losses

## Outputs

- `results/stage19h_fan_anchor_geometry_lora_sanity/`
- `docs/CTAF_STAGE19H_FAN_ANCHOR_GEOMETRY_LORA_SANITY_AUDIT.md`

Stage19I will evaluate the adapted embeddings later with the fixed backend.
