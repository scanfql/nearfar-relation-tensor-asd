#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from nearfar_relation_tensor_asd.audio import load_audio, split_near_far
from nearfar_relation_tensor_asd.features import DEFAULT_CHANNELS, build_relation_tensor, compute_logmel
from nearfar_relation_tensor_asd.manifest import read_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract compact near/far relation tensors from a paired-audio manifest.")
    p.add_argument("--manifest", required=True, help="CSV with basename, near_path, far_path, machine_type, domain, split.")
    p.add_argument("--out-dir", required=True, help="Output directory for .npz tensors and tensor manifest.")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--bins", type=int, default=64)
    return p.parse_args()


def load_mono(path: str, sr: int) -> np.ndarray:
    wav, _ = load_audio(path, sr=sr)
    if wav.shape[0] == 1:
        return wav[0]
    near, _far = split_near_far(wav)
    return near


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row in rows:
        near = load_mono(row["near_path"], args.sr)
        far = load_mono(row["far_path"], args.sr)
        near_lm = compute_logmel(near, args.sr, args.n_fft, args.hop_length, args.n_mels, args.frames, args.bins)
        far_lm = compute_logmel(far, args.sr, args.n_fft, args.hop_length, args.n_mels, args.frames, args.bins)
        tensor, channel_names = build_relation_tensor(near_lm, far_lm)
        name = f"{row['split']}__{row['machine_type']}__{row['domain']}__{Path(row['basename']).stem}.npz"
        path = tensor_dir / name
        np.savez_compressed(path, x=tensor.astype(np.float32), channel_names=np.asarray(channel_names))
        out = dict(row)
        out.update({"tensor_path": str(path), "shape": "x".join(map(str, tensor.shape))})
        out_rows.append(out)

    fields = list(rows[0]) + ["tensor_path", "shape"] if rows else ["tensor_path", "shape"]
    with (out_dir / "tensor_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    (out_dir / "channel_names.txt").write_text("\n".join(DEFAULT_CHANNELS) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
