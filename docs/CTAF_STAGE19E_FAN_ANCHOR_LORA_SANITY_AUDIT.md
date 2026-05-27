# CTAF Stage19E Fan Anchor LoRA Sanity Audit

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
- checkpoint selection: `lowest validation total loss`

## Anchor

- teacher embedding: clean EAT frame output -> min_clamp -> mean pool -> L2 normalize
- student embedding: adapted EAT frame output -> min_clamp -> mean pool -> L2 normalize
- anchor loss: `mean(1 - cosine(student_embedding, teacher_embedding))`

## Guardrails

- minimum pass: `adapted_ldn fan hmean > 0.5527191284766243`
- strong pass: `adapted_ldn fan hmean >= 0.5820070015781968`
- hard fail: `adapted_ldn fan hmean <= 0.5527191284766243`
- additional guardrail: `adapted_baseline_knn_k5 should not be far below Stage18 baseline`
- no ASD scoring in Stage19E
- no test-label checkpoint selection
- no anomaly-label training
- no full7
- no selector
- no fusion
- no Stage19B checkpoint continuation
- no LDN K/r/score_k tuning

Stage19F will evaluate adapted embeddings later with a fixed backend.
