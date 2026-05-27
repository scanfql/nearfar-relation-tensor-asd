#!/usr/bin/env python
"""Stage 18C: HTSAT one-machine frozen kNN sanity."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test


REPO_DEFAULT = "/data1/test/pretrained_models/htsat_clean/HTS-Audio-Transformer"
CKPT_DEFAULT = "/data1/test/pretrained_models/htsat_clean/HTSAT_AudioSet_Saved_1.ckpt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage18c_htsat_one_machine_knn"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18C_HTSAT_ONE_MACHINE_KNN_AUDIT.md"
MACHINE_DEFAULT = "gearbox"


def parse_args():
    p = argparse.ArgumentParser(description="Stage18C HTSAT one-machine frozen kNN sanity")
    p.add_argument("--mode", required=True, choices=["check", "one_wav_sanity", "one_machine_knn"])
    p.add_argument("--htsat_repo_path", default=REPO_DEFAULT)
    p.add_argument("--htsat_ckpt_path", default=CKPT_DEFAULT)
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
    for name in ["torch", "torchaudio", "soundfile", "librosa", "timm", "torchlibrosa", "einops", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_waveform(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != target_sr:
        tensor = torchaudio.functional.resample(tensor, int(sr), target_sr)
    return tensor.detach().cpu().numpy().astype(np.float32), int(sr)


def load_htsat(repo_path: Path, ckpt_path: Path, device: str):
    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    repo_root = str(repo_path.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    for name in ["museval", "musdb", "sed_eval"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    importlib.invalidate_caches()
    config_mod = importlib.import_module("config")
    model_mod = importlib.import_module("model.htsat")

    HTSAT = getattr(model_mod, "HTSAT_Swin_Transformer")
    config = config_mod

    model = HTSAT(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported HTSAT checkpoint format: {type(state_dict)}")

    stripped = {}
    for old_key, value in state_dict.items():
        new_key = old_key.replace("sed_model.", "")
        stripped[new_key] = value

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    ignorable = {"head.weight", "head.bias"}
    missing_filtered = [k for k in missing if k not in ignorable]
    if unexpected:
        raise RuntimeError(f"Unexpected HTSAT checkpoint keys: {unexpected[:20]}")
    if missing_filtered:
        raise RuntimeError(f"Missing HTSAT keys: {missing_filtered[:20]}")

    model.eval()
    model.to(requested)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, requested, ckpt


def htsat_embed(model, waveform: np.ndarray, device):
    import torch

    x = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x, None, True)
        if "latent_output" not in output or "clipwise_output" not in output:
            raise RuntimeError(f"Unexpected HTSAT output keys: {list(output.keys())}")
        latent = output["latent_output"]
        clipwise = output["clipwise_output"]
        framewise = output["framewise_output"] if "framewise_output" in output else None
        if latent.ndim != 2:
            raise RuntimeError(f"Expected latent_output [B,D], got {tuple(latent.shape)}")
    return latent, clipwise, framewise


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
        "# CTAF Stage18C HTSAT One-Machine KNN Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no embedding extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- repo path exists: `{result['repo_exists']}`",
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
    repo_path = Path(args.htsat_repo_path)
    ckpt_path = Path(args.htsat_ckpt_path)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, args.machine) if items else []
    train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
    result = {
        "mode": "check",
        "htsat_repo_path": str(repo_path),
        "repo_exists": repo_path.exists(),
        "htsat_ckpt_path": str(ckpt_path),
        "ckpt_exists": ckpt_path.exists(),
        "data_root_exists": data_root.exists(),
        "machine": args.machine,
        "machine_exists": len(machine_items) > 0,
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "imports": import_status(),
        "notes": "No model load, no forward, no embedding extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", result)
    write_markdown(Path(args.doc_path), build_check_markdown(result))


def run_one_wav_sanity(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    repo_path = Path(args.htsat_repo_path)
    ckpt_path = Path(args.htsat_ckpt_path)
    wav_path = Path(args.wav_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"HTSAT repo path does not exist: {repo_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"HTSAT checkpoint path does not exist: {ckpt_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV path does not exist: {wav_path}")

    model, config, device, ckpt = load_htsat(repo_path, ckpt_path, args.device)
    waveform, orig_sr = load_waveform(wav_path, config.sample_rate)
    latent, clipwise, framewise = htsat_embed(model, waveform, device)

    result = {
        "mode": "one_wav_sanity",
        "htsat_repo_path": str(repo_path),
        "htsat_ckpt_path": str(ckpt_path),
        "wav_path": str(wav_path),
        "original_sample_rate": orig_sr,
        "resampled_sample_rate": int(config.sample_rate),
        "device": str(device),
        "latent_shape": list(latent.shape),
        "clipwise_shape": list(clipwise.shape),
        "framewise_shape": list(framewise.shape) if framewise is not None else None,
        "checkpoint_top_keys": list(ckpt.keys())[:20] if isinstance(ckpt, dict) else [],
        "model_class": model.__class__.__name__,
    }
    write_json(out_root / "one_wav_sanity.json", result)


def run_one_machine_knn(args):
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.score_geometry import compute_score_geometry

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_dir = out_root / "embeddings"
    scores_dir = out_root / "scores"
    emb_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(args.htsat_repo_path)
    ckpt_path = Path(args.htsat_ckpt_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"HTSAT repo path does not exist: {repo_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"HTSAT checkpoint path does not exist: {ckpt_path}")

    model, config, device, ckpt = load_htsat(repo_path, ckpt_path, args.device)
    items = filter_items(scan_dataset(Path(args.data_root)), args.machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)}, test={len(test_items)}")

    train_embs = []
    test_embs = []
    sample_latent_shape = None
    sample_framewise_shape = None
    y_test = []
    d_test = []
    test_paths = []

    for item in train_items:
        wav, _ = load_waveform(Path(item.path), config.sample_rate)
        latent, clipwise, framewise = htsat_embed(model, wav, device)
        train_embs.append(latent.squeeze(0).detach().cpu().numpy())
        sample_latent_shape = list(latent.shape)
        sample_framewise_shape = list(framewise.shape) if framewise is not None else None

    for item in test_items:
        wav, _ = load_waveform(Path(item.path), config.sample_rate)
        latent, clipwise, framewise = htsat_embed(model, wav, device)
        test_embs.append(latent.squeeze(0).detach().cpu().numpy())
        y_test.append(int(item.label))
        d_test.append(item.domain)
        test_paths.append(str(item.path))
        sample_latent_shape = list(latent.shape)
        sample_framewise_shape = list(framewise.shape) if framewise is not None else None

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)
    d_test_arr = np.asarray(d_test)

    scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
    metrics = compute_metrics(y_test_arr, scores, d_test_arr)
    geometry = compute_score_geometry(scores, y_test_arr, d_test_arr)
    tail = tail_metrics(scores, y_test_arr)

    np.savez_compressed(
        emb_dir / f"{args.machine}_htsat_embeddings.npz",
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test_arr,
        d_test=d_test_arr.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        htsat_ckpt_path=str(ckpt_path),
    )
    np.savez(
        scores_dir / f"{args.machine}_htsat_knn_scores.npz",
        test_scores=scores,
        y_test=y_test_arr,
        d_test=d_test_arr.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        score_mode="HTSAT_knn",
        htsat_ckpt_path=str(ckpt_path),
    )

    write_csv(out_root / "embedding_summary.csv", [{
        "machine": args.machine,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "latent_shape": sample_latent_shape,
        "framewise_shape": sample_framewise_shape,
        "utterance_embedding_dim": int(train_emb.shape[1]),
        "sample_rate": int(config.sample_rate),
        "device": str(device),
        "checkpoint_path": str(ckpt_path),
    }])
    write_csv(out_root / "summary.csv", [{
        "candidate": "HTSAT_knn",
        "hmean": float(metrics["hmean"]),
        "pAUC": float(metrics["pauc"]),
        "KS": float(geometry["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
    }])
    write_csv(out_root / "tail_table.csv", [{**tail}])
    write_csv(out_root / "score_geometry.csv", [{**geometry}])

    lines = [
        "# CTAF Stage18C HTSAT One-Machine KNN Audit",
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
