#!/usr/bin/env python
"""Stage 18B: CED one-machine frozen kNN sanity."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MODEL_DEFAULT = "/data1/test/pretrained_models/ced_clean"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage18b_ced_one_machine_knn"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18B_CED_ONE_MACHINE_KNN_AUDIT.md"
MACHINE_DEFAULT = "gearbox"


def parse_args():
    p = argparse.ArgumentParser(description="Stage18B CED one-machine frozen kNN sanity")
    p.add_argument("--mode", required=True, choices=["check", "one_wav_sanity", "one_machine_knn"])
    p.add_argument("--ced_model_path", default=MODEL_DEFAULT)
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
    for name in ["transformers", "torch", "torchaudio", "soundfile", "librosa", "timm", "safetensors", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def ensure_local_class_labels_csv(model_path: Path):
    label_csv = model_path / "class_labels_indices.csv"
    if label_csv.exists():
        return label_csv

    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing CED config.json: {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    id2label = cfg.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        raise RuntimeError("CED config.json missing usable id2label mapping.")

    rows = [["index", "mid", "display_name"]]
    for key, value in sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0]):
        rows.append([str(key), "", str(value)])

    with label_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return label_csv


def load_ced(model_path: Path, device: str):
    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    ensure_local_class_labels_csv(model_path)
    package_parent = str(model_path.parent.resolve())
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    pkg = model_path.name
    cfg_mod = importlib.import_module(f"{pkg}.configuration_ced")
    feat_mod = importlib.import_module(f"{pkg}.feature_extraction_ced")
    model_mod = importlib.import_module(f"{pkg}.modeling_ced")

    CedConfig = getattr(cfg_mod, "CedConfig")
    CedFeatureExtractor = getattr(feat_mod, "CedFeatureExtractor")
    CedModel = getattr(model_mod, "CedModel")

    cfg = CedConfig.from_pretrained(str(model_path))

    preproc_cfg = json.loads((model_path / "preprocessor_config.json").read_text())
    feature_kwargs = {
        "f_min": preproc_cfg.get("f_min", 0),
        "sampling_rate": preproc_cfg.get("sampling_rate", 16000),
        "win_size": preproc_cfg.get("win_size", 512),
        "center": preproc_cfg.get("center", True),
        "n_fft": preproc_cfg.get("n_fft", 512),
        "f_max": preproc_cfg.get("f_max", None),
        "hop_size": preproc_cfg.get("hop_size", 160),
        "feature_size": preproc_cfg.get("feature_size", 64),
        "padding_value": preproc_cfg.get("padding_value", 0.0),
    }
    feature_extractor = CedFeatureExtractor(**feature_kwargs)
    model = CedModel.from_pretrained(str(model_path), config=cfg)
    model.eval()
    model.to(requested)
    for p in model.parameters():
        p.requires_grad_(False)
    return feature_extractor, model, requested


def load_waveform_16k(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != 16000:
        tensor = torchaudio.functional.resample(tensor, int(sr), 16000)
    return tensor.detach().cpu().numpy().astype(np.float32), int(sr)


def ced_embed(feature_extractor, model, waveform_16k: np.ndarray, device):
    import torch

    inputs = feature_extractor(waveform_16k, sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        output = model(input_values=input_values)
        tokens = output.logits if hasattr(output, "logits") else output[0]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected CED token output [B,N,D], got {tuple(tokens.shape)}")
        emb = tokens.mean(dim=1)
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
        "# CTAF Stage18B CED One-Machine KNN Audit",
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
        f"- model path exists: `{result['ced_model_path_exists']}`",
        f"- data root exists: `{result['data_root_exists']}`",
        f"- machine exists: `{result['machine_exists']}`",
        f"- source train-normal count: `{result['source_train_normal_count']}`",
        f"- test count: `{result['test_count']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.ced_model_path)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, args.machine) if items else []
    train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
    result = {
        "mode": "check",
        "ced_model_path": str(model_path),
        "ced_model_path_exists": model_path.exists(),
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
    model_path = Path(args.ced_model_path)
    wav_path = Path(args.wav_path)
    if not model_path.exists():
        raise FileNotFoundError(f"CED model path does not exist: {model_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV path does not exist: {wav_path}")

    waveform_16k, orig_sr = load_waveform_16k(wav_path)
    feature_extractor, model, device = load_ced(model_path, args.device)
    tokens, emb = ced_embed(feature_extractor, model, waveform_16k, device)

    result = {
        "mode": "one_wav_sanity",
        "ced_model_path": str(model_path),
        "wav_path": str(wav_path),
        "original_sample_rate": orig_sr,
        "resampled_sample_rate": 16000,
        "device": str(device),
        "token_shape": list(tokens.shape),
        "embedding_shape": list(emb.shape),
        "embedding_dtype": str(emb.dtype),
        "model_class": model.__class__.__name__,
        "feature_extractor_class": feature_extractor.__class__.__name__,
    }
    write_json(out_root / "one_wav_sanity.json", result)


def run_one_machine_knn(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_dir = out_root / "embeddings"
    scores_dir = out_root / "scores"
    emb_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.ced_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"CED model path does not exist: {model_path}")

    feature_extractor, model, device = load_ced(model_path, args.device)
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
        wav16k, _ = load_waveform_16k(Path(item.path))
        tokens, emb = ced_embed(feature_extractor, model, wav16k, device)
        train_embs.append(emb.squeeze(0).detach().cpu().numpy())
        sample_token_shape = list(tokens.shape)

    for item in test_items:
        wav16k, _ = load_waveform_16k(Path(item.path))
        tokens, emb = ced_embed(feature_extractor, model, wav16k, device)
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
        emb_dir / f"{args.machine}_ced_embeddings.npz",
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test_arr,
        d_test=d_test_arr.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        ced_model_path=str(model_path),
    )
    np.savez(
        scores_dir / f"{args.machine}_ced_knn_scores.npz",
        test_scores=scores,
        y_test=y_test_arr,
        d_test=d_test_arr.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        score_mode="CED_knn",
        ced_model_path=str(model_path),
    )

    write_csv(out_root / "embedding_summary.csv", [{
        "machine": args.machine,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "token_shape": sample_token_shape,
        "utterance_embedding_dim": int(train_emb.shape[1]),
        "device": str(device),
        "model_path": str(model_path),
    }])
    write_csv(out_root / "summary.csv", [{
        "candidate": "CED_knn",
        "hmean": float(metrics["hmean"]),
        "pAUC": float(metrics["pauc"]),
        "KS": float(geometry["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
    }])
    write_csv(out_root / "tail_table.csv", [{**tail}])
    write_csv(out_root / "score_geometry.csv", [{**geometry}])

    lines = [
        "# CTAF Stage18B CED One-Machine KNN Audit",
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
