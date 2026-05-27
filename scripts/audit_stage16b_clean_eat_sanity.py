#!/usr/bin/env python
"""Stage 16B clean EAT local verification + one-wav sanity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Stage 16B clean EAT sanity")
    p.add_argument("--mode", required=True, choices=["local_verify", "one_wav_sanity"])
    p.add_argument("--eat_model_path", required=True)
    p.add_argument("--wav_path", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", required=True)
    return p.parse_args()


def _import_status():
    status = {}
    for name in ["transformers", "torch", "torchaudio", "soundfile", "librosa", "timm", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_clean_eat_local(model_path: Path, device: str):
    """Load clean EAT from local checkpoint package without HF dynamic-module cache."""
    import torch

    package_parent = model_path.parent.resolve()  # .../eat_clean
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from checkpoints.configuration_eat import EATConfig
    from checkpoints.modeling_eat import EATModel

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    try:
        model = EATModel.from_pretrained(str(model_path), trust_remote_code=False)
    except Exception:
        from safetensors.torch import load_file

        cfg = EATConfig.from_pretrained(str(model_path))
        model = EATModel(cfg)
        state = load_file(str(model_path / "model.safetensors"))
        model.load_state_dict(state, strict=True)

    model.eval()
    model.to(requested)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def run_local_verify(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"EAT model path does not exist: {model_path}. Please pass the correct --eat_model_path."
        )

    files = {p.name for p in model_path.iterdir()} if model_path.is_dir() else set()
    expected_config = ["config.json", "configuration_eat.py", "modeling_eat.py", "eat_model.py", "model_core.py"]
    expected_model = ["model.safetensors", "pytorch_model.bin"]
    processor_candidates = ["preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]
    sha_candidates = ["README_DOWNLOAD.md", "SHA256SUMS", "sha256.txt"]

    result = {
        "mode": "local_verify",
        "eat_model_path": str(model_path),
        "exists": model_path.exists(),
        "is_dir": model_path.is_dir(),
        "config_files_present": [f for f in expected_config if f in files],
        "model_files_present": [f for f in expected_model if f in files],
        "processor_or_preprocessor_files_present": [f for f in processor_candidates if f in files],
        "sha256_or_download_log_present": [f for f in sha_candidates if f in files or (model_path.parent / f).exists()],
        "import_status": _import_status(),
    }
    (output_root / "local_verify.json").write_text(json.dumps(result, indent=2) + "\n")


def run_one_wav_sanity(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    wav_path = Path(args.wav_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"EAT model path does not exist: {model_path}. Please pass the correct --eat_model_path."
        )
    if not wav_path.exists():
        raise FileNotFoundError(
            f"WAV path does not exist: {wav_path}. Please pass an explicit --wav_path."
        )

    import torch
    import torchaudio
    import soundfile as sf

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model, device = load_clean_eat_local(model_path, str(device))

    wav, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    orig_sr = int(sr)
    target_sr = 16000

    waveform = torch.tensor(wav, dtype=torch.float32)
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    final_sr = target_sr

    with torch.no_grad():
        # EAT repo guidance: 128-bin fbank, pad/truncate to 1024 frames
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=final_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        target_length = 1024
        if mel.shape[0] < target_length:
            pad = target_length - mel.shape[0]
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        else:
            mel = mel[:target_length, :]
        norm_mean = -4.268
        norm_std = 4.569
        mel = (mel - norm_mean) / (norm_std * 2)
        x = mel.unsqueeze(0).unsqueeze(0).to(device)

        output = model(x)
        output_keys = list(output.keys()) if hasattr(output, "keys") else []
        last_hidden = getattr(output, "last_hidden_state", None)
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            chosen = pooled
        elif last_hidden is not None:
            chosen = last_hidden
        elif isinstance(output, torch.Tensor):
            chosen = output
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            chosen = output[0]
        else:
            chosen = None

    result = {
        "mode": "one_wav_sanity",
        "eat_model_path": str(model_path),
        "input_wav_path": str(wav_path),
        "original_sample_rate": orig_sr,
        "resampled_sample_rate": final_sr,
        "device": str(device),
        "dtype": str(chosen.dtype) if isinstance(chosen, torch.Tensor) else "unknown",
        "output_type": str(type(output)),
        "output_keys": output_keys,
        "last_hidden_state_shape": list(last_hidden.shape) if last_hidden is not None else None,
        "pooler_output_shape": list(pooled.shape) if pooled is not None else None,
        "chosen_embedding_shape": list(chosen.shape) if isinstance(chosen, torch.Tensor) else None,
    }
    (output_root / "one_wav_sanity.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    args = parse_args()
    if args.mode == "local_verify":
        run_local_verify(args)
        return
    if args.mode == "one_wav_sanity":
        run_one_wav_sanity(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
