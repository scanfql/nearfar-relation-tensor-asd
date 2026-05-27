#!/usr/bin/env python
"""Stage 18D: PaSST one-machine frozen kNN sanity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


CKPT_DEFAULT = "/data1/test/pretrained_models/passt_clean/passt-s-f128-p16-s10-ap.476-swa.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage18d_passt_one_machine_knn"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18D_PASST_ONE_MACHINE_KNN_AUDIT.md"
MACHINE_DEFAULT = "gearbox"


def parse_args():
    p = argparse.ArgumentParser(description="Stage18D PaSST one-machine frozen kNN sanity")
    p.add_argument("--mode", required=True, choices=["check", "one_wav_sanity", "one_machine_knn"])
    p.add_argument("--passt_ckpt_path", default=CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--wav_path", default="")
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def import_status():
    status = {}
    for name in ["torch", "torchaudio", "soundfile", "librosa", "timm", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def build_passt_model(device: str):
    import torch.nn as nn
    from timm.models.vision_transformer import Block

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))

        def forward(self, x):
            return self.proj(x)

    class PaSSTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.dist_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.new_pos_embed = nn.Parameter(torch.zeros(1, 2, 768))
            self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, 768, 12, 1))
            self.time_new_pos_embed = nn.Parameter(torch.zeros(1, 768, 1, 99))
            self.patch_embed = PatchEmbed()
            self.blocks = nn.ModuleList(
                [Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(12)]
            )
            self.norm = nn.LayerNorm(768)

        def forward_tokens(self, x):
            # x: [B, 1, 128, 998]
            x = self.patch_embed(x)  # [B,768,12,99]
            x = x + self.freq_new_pos_embed[:, :, : x.shape[2], :]
            x = x + self.time_new_pos_embed[:, :, :, : x.shape[3]]
            x = x.flatten(2).transpose(1, 2)
            cls = self.cls_token + self.new_pos_embed[:, 0:1, :]
            dist = self.dist_token + self.new_pos_embed[:, 1:2, :]
            cls = cls.expand(x.shape[0], -1, -1)
            dist = dist.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, dist, x], dim=1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    model = PaSSTBackbone().to(requested)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def load_passt_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported PaSST checkpoint format: {type(ckpt)}")

    model, requested = build_passt_model(device)
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected PaSST checkpoint keys: {unexpected[:20]}")
    ignorable = {"head.1.weight", "head.1.bias"}
    missing_filtered = [k for k in missing if k not in ignorable]
    if missing_filtered:
        raise RuntimeError(f"Missing PaSST keys: {missing_filtered[:20]}")
    model.eval()
    model.to(requested)
    return model, requested, ckpt


def load_waveform_32k(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != 32000:
        tensor = torchaudio.functional.resample(tensor, int(sr), 32000)
    return tensor.detach().cpu().numpy().astype(np.float32), int(sr)


def waveform_to_passt_input(wav_32k: np.ndarray):
    import torchaudio

    wav = torch.tensor(wav_32k, dtype=torch.float32)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=32000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )  # [T,128]
    target_length = 998
    if mel.shape[0] < target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
    else:
        mel = mel[:target_length, :]
    mel = mel.transpose(0, 1).contiguous().unsqueeze(0)  # [1,128,998]
    return mel.to(dtype=torch.float32)


def passt_embed(model, waveform_32k: np.ndarray, device):
    x = waveform_to_passt_input(waveform_32k).unsqueeze(0).to(device)  # [1,1,128,998]
    with torch.no_grad():
        tokens = model.forward_tokens(x)
        patch_tokens = tokens[:, 2:, :]
        emb = patch_tokens.mean(dim=1)
    return tokens, emb


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5):
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, len(train_n))
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray):
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95)) if len(normals) else 0.0
    q99 = float(np.quantile(normals, 0.99)) if len(normals) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)) if len(anomalies) else 0.0,
        "anomaly_q75": float(np.quantile(anomalies, 0.75)) if len(anomalies) else 0.0,
        "TPR@95n": float(np.mean(anomalies > q95)) if len(anomalies) else 0.0,
        "TPR@99n": float(np.mean(anomalies > q99)) if len(anomalies) else 0.0,
    }


def build_check_markdown(result: dict) -> list[str]:
    failed = [k for k, ok in result["imports"].items() if not ok]
    return [
        "# CTAF Stage18D PaSST One-Machine KNN Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no checkpoint load",
        "- no forward",
        "- no embedding extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- checkpoint path exists: `{result['ckpt_exists']}`",
        f"- data root exists: `{result['data_root_exists']}`",
        f"- machine exists: `{result['machine_exists']}`",
        f"- source train-normal count: `{result['source_train_normal_count']}`",
        f"- test count: `{result['test_count']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.passt_ckpt_path)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, args.machine) if items else []
    train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])

    result = {
        "mode": "check",
        "passt_ckpt_path": str(ckpt_path),
        "ckpt_exists": ckpt_path.exists(),
        "data_root_exists": data_root.exists(),
        "machine": args.machine,
        "machine_exists": len(machine_items) > 0,
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "imports": import_status(),
        "notes": "No checkpoint load, no forward, no embedding extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", result)
    write_markdown(Path(args.doc_path), build_check_markdown(result))


def run_one_wav_sanity(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.passt_ckpt_path)
    wav_path = Path(args.wav_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PaSST checkpoint path does not exist: {ckpt_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV path does not exist: {wav_path}")

    waveform_32k, orig_sr = load_waveform_32k(wav_path)
    model, device, ckpt = load_passt_checkpoint(ckpt_path, args.device)
    tokens, emb = passt_embed(model, waveform_32k, device)

    result = {
        "mode": "one_wav_sanity",
        "passt_ckpt_path": str(ckpt_path),
        "wav_path": str(wav_path),
        "original_sample_rate": orig_sr,
        "resampled_sample_rate": 32000,
        "device": str(device),
        "token_shape": list(tokens.shape),
        "embedding_shape": list(emb.shape),
        "embedding_dtype": str(emb.dtype),
        "checkpoint_top_keys": list(ckpt.keys())[:20] if isinstance(ckpt, dict) else [],
    }
    write_json(out_root / "one_wav_sanity.json", result)


def run_one_machine_knn(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_dir = out_root / "embeddings"
    scores_dir = out_root / "scores"
    emb_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.passt_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PaSST checkpoint path does not exist: {ckpt_path}")

    model, device, ckpt = load_passt_checkpoint(ckpt_path, args.device)
    items = filter_items(scan_dataset(Path(args.data_root)), args.machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)}, test={len(test_items)}")

    train_embs = []
    test_embs = []
    sample_token_shape = None
    y_test = []
    d_test = []
    test_paths = []

    for item in train_items:
        wav32k, _ = load_waveform_32k(Path(item.path))
        tokens, emb = passt_embed(model, wav32k, device)
        train_embs.append(emb.squeeze(0).detach().cpu().numpy())
        sample_token_shape = list(tokens.shape)

    for item in test_items:
        wav32k, _ = load_waveform_32k(Path(item.path))
        tokens, emb = passt_embed(model, wav32k, device)
        test_embs.append(emb.squeeze(0).detach().cpu().numpy())
        y_test.append(int(item.label))
        d_test.append(item.domain)
        test_paths.append(str(item.path))
        sample_token_shape = list(tokens.shape)

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)
    d_test_arr = np.asarray(d_test)

    scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
    metrics = compute_metrics(y_test_arr, scores, d_test_arr)
    geometry = compute_score_geometry(scores, y_test_arr, d_test_arr)
    tail = tail_metrics(scores, y_test_arr)

    np.savez_compressed(
        emb_dir / f"{args.machine}_passt_embeddings.npz",
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test_arr,
        d_test=d_test_arr.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        passt_ckpt_path=str(ckpt_path),
    )
    np.savez(
        scores_dir / f"{args.machine}_passt_knn_scores.npz",
        test_scores=scores,
        y_test=y_test_arr,
        d_test=d_test_arr.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        score_mode="PaSST_knn",
        passt_ckpt_path=str(ckpt_path),
    )

    write_csv(out_root / "embedding_summary.csv", [{
        "machine": args.machine,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "token_shape": sample_token_shape,
        "utterance_embedding_dim": int(train_emb.shape[1]),
        "sample_rate": 32000,
        "device": str(device),
        "checkpoint_path": str(ckpt_path),
    }])
    write_csv(out_root / "summary.csv", [{
        "candidate": "PaSST_knn",
        "hmean": float(metrics["hmean"]),
        "pAUC": float(metrics["pauc"]),
        "KS": float(geometry["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
    }])
    write_csv(out_root / "tail_table.csv", [{**tail}])
    write_csv(out_root / "score_geometry.csv", [{**geometry}])

    lines = [
        "# CTAF Stage18D PaSST One-Machine KNN Audit",
        "",
        f"- machine: `{args.machine}`",
        f"- hmean: `{metrics['hmean']:.6f}`",
        f"- pAUC: `{metrics['pauc']:.6f}`",
        f"- TPR@99n: `{tail['TPR@99n']:.6f}`",
        "",
        "No training occurred.",
    ]
    write_markdown(Path(args.doc_path), lines)


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "one_wav_sanity":
        run_one_wav_sanity(args)
        return
    if args.mode == "one_machine_knn":
        run_one_machine_knn(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
