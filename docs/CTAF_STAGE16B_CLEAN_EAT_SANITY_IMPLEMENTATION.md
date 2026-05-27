# CTAF Stage16B Clean EAT Sanity Implementation

## 1. Modified File List

- `scripts/audit_stage16b_clean_eat_sanity.py`
- `docs/CTAF_STAGE16B_CLEAN_EAT_SANITY_IMPLEMENTATION.md`

## 2. Script Modes

- `--mode local_verify`
  - checks EAT model directory exists
  - checks config/model files exist
  - checks processor/preprocessor files if present
  - checks sha256/download log if present
  - checks import availability for:
    - `transformers`
    - `torch`
    - `torchaudio`
    - `soundfile`
    - `librosa`
  - no model load
  - no forward
  - no extraction

- `--mode one_wav_sanity`
  - loads clean EAT from local path using local `checkpoints.*` import
  - avoids Hugging Face dynamic-module cache
  - falls back to local `safetensors` loading if needed
  - loads exactly one wav
  - handles resampling to 16 kHz
  - uses `torch.no_grad()` and `model.eval()`
  - writes one JSON summary only
  - no batch DCASE extraction
  - no kNN
  - no scores
  - no hmean

## 3. Expected Inputs

- clean EAT model path:
  - `/data1/test/pretrained_models/eat_clean/checkpoints`
- one wav path:
  - manually supplied
- output root:
  - `results/stage16b_clean_eat_sanity`

## 4. Expected Outputs

- `results/stage16b_clean_eat_sanity/local_verify.json`
- `results/stage16b_clean_eat_sanity/one_wav_sanity.json`

## 5. Server Command for local_verify

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_sanity.py \
  --mode local_verify \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --output_root results/stage16b_clean_eat_sanity
```

## 6. Server Command for one_wav_sanity

Replace `YOUR_WAV_PATH.wav` manually, for example from:

- `/data1/test/LAES_2025/data/dcase/*/*/train/*.wav`

```bash
cd /data1/test/LAES_2025/new_projects/clean_tensor_asd && \
source /home/yinpin/miniconda3/etc/profile.d/conda.sh && \
conda activate scanf && \
python scripts/audit_stage16b_clean_eat_sanity.py \
  --mode one_wav_sanity \
  --eat_model_path /data1/test/pretrained_models/eat_clean/checkpoints \
  --wav_path YOUR_WAV_PATH.wav \
  --device cuda:0 \
  --output_root results/stage16b_clean_eat_sanity
```

## 7. Confirmation

- no command was run in Codex
- no training was run in Codex
- no evaluation was run in Codex
- no embedding extraction was run in Codex
