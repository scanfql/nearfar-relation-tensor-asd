# CTAF Stage19E Fan Anchor LoRA Sanity Implementation

Stage19E is the first anchor-regularized fan adaptation scaffold.

## Goal

Preserve the Stage18F frozen embedding geometry while allowing only a conservative fan-only adaptation path.

## Protocol

- machine: `fan`
- training data: `source train-normal only`
- split: `90/10 by seed 13711`
- teacher: frozen clean EAT
- student: clean EAT with LoRA only on last block attention qkv/proj
- LoRA rank/alpha/dropout: `4 / 8 / 0.05`
- mask ratio: `0.30`
- loss: `L = L_recon_masked + lambda_anchor * L_anchor_embedding`
- lambda anchor: `1.0`
- epochs: `5`
- lr: `1e-5`
- early stop patience: `2`
- checkpoint selection: lowest validation total loss

## Anchor

- teacher embedding: clean EAT frame output -> min_clamp -> mean pool -> L2 normalize
- student embedding: adapted EAT frame output -> min_clamp -> mean pool -> L2 normalize
- anchor loss: `mean(1 - cosine(student_embedding, teacher_embedding))`

## Modes

- `check`: validates data/checkpoint/protocol-file availability, output-root writability, and imports
- `inspect_lora`: loads clean EAT, injects LoRA only into the last block attention qkv/proj, and audits trainable parameters
- `train_fan_anchor_lora`: training scaffold only; no ASD scoring in Stage19E

## Guardrails

- no anomaly labels
- no test-label checkpoint selection
- no full last-block fine-tuning
- no full7
- no selector
- no fusion
- no Stage19B checkpoint continuation
- no LDN K/r/score_k tuning

Stage19F is the first ASD evaluation stage for the adapted embeddings.
