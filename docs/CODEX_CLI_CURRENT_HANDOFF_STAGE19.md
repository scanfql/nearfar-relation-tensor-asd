# Codex CLI Current Handoff Stage19

- confirmed best frozen route: `EAT min_clamp__mean + ldn_diff_K1_scorek5`
- stage19c lesson: `pure masked reconstruction damaged ASD geometry`
- next valid direction: Stage19E fan anchor-regularized LoRA adaptation scaffold
- recommended trainable scope: `last_block_attention_lora`
- LoRA rank/alpha/dropout: `4 / 8 / 0.05`
- mask ratio: `0.30`
- lambda anchor: `1.0`
- epochs: `5`
- lr: `1e-5`
- early stop patience: `2`
- checkpoint selection: `lowest validation total loss`

- do not continue Stage19B checkpoint
- do not train more epochs on Stage19B
- do not open trainable_last_blocks=2
- do not expand the failed checkpoint to gearbox or full7
- do not tune K, r, or score_k
- do not add selector or fusion
- do not use test labels for checkpoint selection
- do not use anomaly labels for training

- stage18 fan baseline hmean: `0.582007`
- stage18 fan ldn hmean: `0.552719`
- stage19c adapted baseline hmean: `0.552055`
- stage19c adapted ldn hmean: `0.531725`
