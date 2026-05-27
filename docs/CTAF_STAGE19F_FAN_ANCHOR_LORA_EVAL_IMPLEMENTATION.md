# CTAF Stage19F Fan Anchor-LoRA Evaluation Implementation

Stage19F is the fan-only evaluation pass for the Stage19E anchor-LoRA checkpoint.
It reuses the Stage19C embedding protocol, but loads the Stage19E LoRA checkpoint correctly and evaluates only two fixed candidates:

- `adapted_baseline_knn_k5`
- `adapted_ldn_diff_K1_scorek5`

## Fixed protocol

- machine: `fan`
- checkpoint: `results/stage19e_fan_anchor_lora_sanity/checkpoints/fan_stage19e_anchor_lora_best.pt`
- clean EAT checkpoint: `/data1/test/pretrained_models/eat_clean/checkpoints`
- data root: `/data1/test/LAES_2025/data/dcase`
- LoRA targets: `model.blocks.11.attn.qkv`, `model.blocks.11.attn.proj`
- LoRA config: rank `4`, alpha `8`, dropout `0.05`
- preprocess: 16 kHz, 128-bin Kaldi fbank, frame shift 10 ms, target length 1024
- normalization: `(mel - (-4.268)) / (4.569 * 2)`
- frame output: `[T, 768]`
- min-clamp: `abs(x) < 0.1 -> 0`
- pooling: mean
- score K: `5`
- no selector, no oracle, no fusion, no full7
- no test-label checkpoint selection
- no anomaly-label training

## Modes

### `--mode check`

Checks:

- clean EAT checkpoint exists
- Stage19E adapted checkpoint exists
- data root exists
- Stage19E root exists
- Stage18E root exists
- Stage18F root exists
- fan train/test inventory counts
- imports: `torch`, `torchaudio`, `soundfile`, `numpy`, `scipy`, `sklearn`, `safetensors`
- output root writable

No model load, no forward, no extraction, no scoring.

### `--mode inspect_checkpoint`

Loads clean EAT, injects the Stage19E LoRA modules, and loads the Stage19E checkpoint.
Outputs:

- `checkpoint_status.json`
- `lora_load_status.json`
- `adapted_parameter_audit.csv`
- `trainable_boundary_recap.json`

This mode verifies:

- checkpoint exists and loads
- LoRA targets match the Stage19E layout
- checkpoint carries the adapted student state
- no unexpected trainable base parameters

### `--mode extract_adapted_embeddings`

Runs the adapted student on fan train-normal and test audio, then stores min-clamped mean-pooled embeddings only.
No scoring.

Outputs:

- `embeddings/fan_anchor_lora_minclamp_mean_embeddings.npz`
- `embedding_summary.csv`

### `--mode score_adapted_fan`

Reads the embedding NPZ and computes the two fixed scores only.
Outputs:

- `scores/adapted_baseline_knn_k5/fan_scores.npz`
- `scores/adapted_ldn_diff_K1_scorek5/fan_scores.npz`
- `summary.csv`
- `tail_table.csv`
- `score_geometry.csv`
- `comparison_vs_stage18_fan.csv`
- `stage19f_fan_anchor_lora_eval_verdict.json`
- `docs/CTAF_STAGE19F_FAN_ANCHOR_LORA_EVAL_AUDIT.md`

## Gates

- minimum pass: `adapted_ldn fan hmean > 0.5527191284766243`
- strong pass: `adapted_ldn fan hmean >= 0.5820070015781968`
- hard fail: `adapted_ldn fan hmean <= 0.5527191284766243`
- representation guardrail: adapted baseline should stay close to Stage18 baseline; a drop greater than `0.02` absolute is treated as collapse

