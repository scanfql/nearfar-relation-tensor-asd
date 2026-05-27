# CTAF Stage19D Anchor-Regularized Adaptation Design Audit

- current baseline route: `EAT min_clamp__mean + ldn_diff_K1_scorek5`
- current baseline mean_hmean: `0.615947`
- lesson from Stage19C: `pure masked reconstruction damaged ASD geometry`
- anchor definition: frozen clean EAT teacher vs adapted EAT student, both min_clamp -> mean pool -> L2 normalize
- recommended Stage19E: `fan_anchor_regularized_adaptation_scaffold`
- recommended first scope: `last_block_attention_lora`
- LoRA rank/alpha/dropout: `4 / 8 / 0.05`
- full last block tuning is not recommended

## Gates

- minimum pass: `adapted_ldn fan hmean > 0.5527191284766243`
- strong pass: `adapted_ldn fan hmean >= 0.5820070015781968`
- hard fail: `adapted_ldn fan hmean <= 0.5527191284766243`
- additional guardrail: `adapted_baseline_knn_k5 should not be far below Stage18 baseline`

## Negative Summary

- stage18 fan baseline hmean: `0.582007`
- stage18 fan ldn hmean: `0.552719`
- stage19c adapted baseline hmean: `0.552055`
- stage19c adapted ldn hmean: `0.531725`
- conclusion: `pure masked reconstruction damaged ASD geometry`

## Stage19E Protocol

- machine: `fan`
- training data: `source train-normal only`
- split: `90/10 by seed 13711`
- trainable scope: `last_block_attention_lora`
- LoRA rank/alpha/dropout: `4 / 8 / 0.05`
- mask ratio: `0.30`
- lambda anchor: `1.0`
- epochs: `5`
- lr: `1e-5`
- early stop patience: `2`
- checkpoint selection: `lowest validation total loss`

## Forbidden

- continue Stage19B checkpoint
- train more epochs on Stage19B
- open trainable_last_blocks=2
- expand the failed checkpoint to gearbox or full7
- tune LDN K/r/score_k
- add selector or fusion
- use test labels for checkpoint selection
- use anomaly labels for training

Stage19E should only train. Stage19F should evaluate fan adapted embeddings with the fixed backend.
