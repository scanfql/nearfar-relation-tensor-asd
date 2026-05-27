#!/usr/bin/env python
"""Stage 16C-2: clean EAT frozen segment-chunk one-machine sanity.

This script is strictly frozen-EAT only.

Allowed:
- local clean EAT load for chunk-cache extraction only
- frame output extraction from clean EAT [T, 768]
- chunk embedding construction with fixed chunk configs
- source train-normal-only chunk memory
- fixed cosine kNN scoring with fixed aggregation rules

Blocked:
- no training
- no fine-tuning
- no LoRA
- no adapter
- no projection head
- no A1/V63 fusion in this first sanity
- no alpha sweep
- no per-machine route selection
- no oracle
- no 7-machine run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MODEL_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
POOLING_ROOT_DEFAULT = "results/stage16b_clean_eat_pooling_fusion"
OUTPUT_ROOT_DEFAULT = "results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE16C2_CLEAN_EAT_SEGMENT_CHUNK_ONE_MACHINE_GEARBOX_AUDIT.md"
MACHINE_DEFAULT = "gearbox"

CHUNK_CONFIGS = [
    {"name": "C08S04", "chunk_len": 8, "chunk_stride": 4},
    {"name": "C16S08", "chunk_len": 16, "chunk_stride": 8},
    {"name": "C32S16", "chunk_len": 32, "chunk_stride": 16},
]
AGGREGATION_RULES = ["top1", "top5mean", "top10mean"]

REGISTERED_P1_REFERENCE = {
    "candidate": "P1_mean_std_REGISTERED",
    "machine": MACHINE_DEFAULT,
    "hmean": 0.561801,
    "pAUC": 0.493158,
    "TPR@99n": 0.020000,
}
REGISTERED_F_P1_A1_065_REFERENCE = {
    "candidate": "F_P1_A1_065_REGISTERED",
    "machine": MACHINE_DEFAULT,
    "hmean": 0.572219,
    "pAUC": 0.519474,
    "TPR@99n": 0.030000,
}
REGISTERED_F_P1_S3_050_REFERENCE = {
    "candidate": "F_P1_S3_050_REGISTERED",
    "machine": MACHINE_DEFAULT,
    "hmean": 0.575941,
    "pAUC": 0.525789,
    "TPR@99n": 0.080000,
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage16C-2 clean EAT segment-chunk one-machine sanity")
    p.add_argument("--mode", required=True, choices=["check", "extract_chunk_cache", "score_chunk"])
    p.add_argument("--eat_model_path", default=MODEL_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--pooling_root", default=POOLING_ROOT_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--distance_chunk_size", type=int, default=256)
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
    for name in ["torch", "torchaudio", "soundfile", "librosa", "transformers", "timm", "safetensors", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_clean_eat(model_path: Path, device: str):
    import torch

    package_parent = model_path.parent.resolve()
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
    for param in model.parameters():
        param.requires_grad_(False)
    return model, requested


def eat_frames(model, waveform: np.ndarray, orig_sr: int, device):
    import torch
    import torchaudio

    target_sr = 16000
    wav = torch.tensor(waveform, dtype=torch.float32)
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)

    with torch.no_grad():
        mel = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=target_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        target_length = 1024
        if mel.shape[0] < target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
        else:
            mel = mel[:target_length, :]
        norm_mean = -4.268
        norm_std = 4.569
        mel = (mel - norm_mean) / (norm_std * 2)
        x = mel.unsqueeze(0).unsqueeze(0).to(device)
        output = model(x)

        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            feat = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            feat = output
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            feat = output[0]
        else:
            raise RuntimeError(f"Unsupported EAT output type: {type(output)}")

        if feat.ndim != 3:
            raise RuntimeError(f"Unexpected EAT output shape: {tuple(feat.shape)}")

    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32), target_sr


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


def p1_score_path(pooling_root: Path, machine: str) -> Path:
    return pooling_root / "scores" / "P1_mean_std" / f"{machine}_scores.npz"


def load_score_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    if "test_scores" in arr.files:
        scores = arr["test_scores"].astype(np.float64)
    elif "scores" in arr.files:
        scores = arr["scores"].astype(np.float64)
    else:
        raise KeyError(f"{npz_path} missing score array; keys={arr.files}")
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "scores": scores,
        "y": arr["y_test"].astype(np.int64),
        "d": arr[domains_key].astype(str),
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        "path": str(npz_path),
    }


def chunk_embeddings_from_frames(frames: np.ndarray, chunk_len: int, chunk_stride: int) -> np.ndarray:
    t = int(frames.shape[0])
    if t <= 0:
        raise RuntimeError("Empty frame sequence encountered.")

    chunks = []
    if t < chunk_len:
        chunk = frames
        mean = chunk.mean(axis=0)
        std = chunk.std(axis=0)
        chunks.append(np.concatenate([mean, std]).astype(np.float32))
    else:
        for start in range(0, t - chunk_len + 1, chunk_stride):
            chunk = frames[start:start + chunk_len]
            mean = chunk.mean(axis=0)
            std = chunk.std(axis=0)
            chunks.append(np.concatenate([mean, std]).astype(np.float32))
    if not chunks:
        raise RuntimeError(f"Failed to construct chunks for frames shape={frames.shape}, len={chunk_len}, stride={chunk_stride}")
    return np.stack(chunks, axis=0).astype(np.float32)


def flatten_with_offsets(chunks_per_file: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    counts = []
    flat = []
    for chunk_emb in chunks_per_file:
        counts.append(int(chunk_emb.shape[0]))
        flat.append(chunk_emb.astype(np.float32))
        offsets.append(offsets[-1] + int(chunk_emb.shape[0]))
    flat_arr = np.concatenate(flat, axis=0).astype(np.float32) if flat else np.zeros((0, 1536), dtype=np.float32)
    return flat_arr, np.asarray(offsets, dtype=np.int32), np.asarray(counts, dtype=np.int32)


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def chunk_knn_scores(train_memory: np.ndarray, query_chunks: np.ndarray, k: int, batch_size: int) -> np.ndarray:
    train = l2_normalize(train_memory.astype(np.float32), axis=1)
    query = l2_normalize(query_chunks.astype(np.float32), axis=1)
    if train.shape[0] == 0:
        raise RuntimeError("Train segment memory is empty.")
    k = min(k, train.shape[0])
    out = []
    train_norm2 = np.sum(train * train, axis=1)[None, :]
    for start in range(0, query.shape[0], batch_size):
        q = query[start:start + batch_size]
        q_norm2 = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_norm2 + train_norm2 - 2.0 * (q @ train.T)
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
        out.append(np.sqrt(nearest).mean(axis=1).astype(np.float64))
    return np.concatenate(out, axis=0)


def aggregate_chunk_scores(chunk_scores: np.ndarray, rule: str) -> float:
    if chunk_scores.size == 0:
        raise RuntimeError("Cannot aggregate empty chunk scores.")
    sorted_scores = np.sort(chunk_scores)[::-1]
    if rule == "top1":
        return float(sorted_scores[0])
    if rule == "top5mean":
        n = max(1, int(np.ceil(0.05 * chunk_scores.size)))
        return float(np.mean(sorted_scores[:n]))
    if rule == "top10mean":
        n = max(1, int(np.ceil(0.10 * chunk_scores.size)))
        return float(np.mean(sorted_scores[:n]))
    raise ValueError(f"Unknown aggregation rule: {rule}")


def tail_metrics(scores: np.ndarray, y_test: np.ndarray) -> dict:
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95)) if len(normals) else 0.0
    q99 = float(np.quantile(normals, 0.99)) if len(normals) else 0.0
    anomaly_median = float(np.quantile(anomalies, 0.50)) if len(anomalies) else 0.0
    anomaly_q75 = float(np.quantile(anomalies, 0.75)) if len(anomalies) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": anomaly_median,
        "anomaly_q75": anomaly_q75,
        "TPR@95n": float(np.mean(anomalies > q95)) if len(anomalies) else 0.0,
        "TPR@99n": float(np.mean(anomalies > q99)) if len(anomalies) else 0.0,
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    summary = {
        "hmean": float(met["hmean"]),
        "pAUC": float(met["pauc"]),
        "KS": float(geom["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "anomaly_median": float(tail["anomaly_median"]),
        "anomaly_q75": float(tail["anomaly_q75"]),
    }
    return summary, tail, geom


def runtime_p1_reference(pooling_root: Path, machine: str):
    score_path = p1_score_path(pooling_root, machine)
    if not score_path.exists():
        return None
    return load_score_bundle(score_path)


def chunk_cache_path(output_root: Path, config_name: str, machine: str) -> Path:
    return output_root / "chunk_cache" / f"{config_name}_{machine}_chunks.npz"


def build_check_markdown(status: dict) -> list[str]:
    failed = [name for name, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage16C2 Clean EAT Segment-Chunk One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- machine: `{status['machine']}`",
        f"- checkpoint path exists: `{status['eat_model_path_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- source train-normal count: `{status['source_train_normal_count']}`",
        f"- test count: `{status['test_count']}`",
        f"- P1 baseline score exists: `{status['p1_baseline_score_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_extract_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "# CTAF Stage16C2 Clean EAT Segment-Chunk One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `extract_chunk_cache`",
        "- frozen clean EAT only",
        "- no scoring",
        "",
        "## Chunk Cache Summary",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"- `{row['config_name']}` train_chunks=`{row['train_total_chunks']}` test_chunks=`{row['test_total_chunks']}` cache=`{row['cache_path']}`",
            ]
        )
    return lines


def build_score_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage16C2 Clean EAT Segment-Chunk One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `score_chunk`",
        "- no model load",
        "- no forward",
        "",
        "## Best Chunk Candidate",
        "",
        f"- machine: `{verdict['machine']}`",
        f"- best chunk candidate: `{verdict['best_chunk_candidate']}`",
        f"- best hmean: `{verdict['best_hmean']:.6f}`",
        f"- best pAUC: `{verdict['best_pAUC']:.6f}`",
        f"- best TPR@99n: `{verdict['best_TPR@99n']:.6f}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- no A1/V63 fusion",
        "- no alpha sweep",
        "- no per-machine route",
        "- gearbox only",
        "- human review required before any promotion",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    pooling_root = Path(args.pooling_root) if args.pooling_root else None
    items, train_items, test_items = expected_items(data_root, args.machine) if data_root.exists() else ([], [], [])
    p1_exists = p1_score_path(pooling_root, args.machine).exists() if pooling_root is not None else False

    status = {
        "mode": "check",
        "machine": args.machine,
        "eat_model_path": str(model_path),
        "eat_model_path_exists": model_path.exists(),
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "requested_machine_exists": len(items) > 0,
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "p1_baseline_path": str(p1_score_path(pooling_root, args.machine)) if pooling_root is not None else "",
        "p1_baseline_score_exists": bool(p1_exists),
        "imports": import_status(),
        "notes": "No model load, no forward, no extraction, no scoring.",
    }

    write_json(out_root / "check_status.json", status)
    write_csv(
        out_root / "machine_check.csv",
        [
            {
                "machine": args.machine,
                "machine_exists": len(items) > 0,
                "source_train_normal_count": len(train_items),
                "test_count": len(test_items),
                "eat_model_path_exists": model_path.exists(),
                "data_root_exists": data_root.exists(),
                "p1_baseline_score_exists": p1_exists,
            }
        ],
    )
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_extract_chunk_cache(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "chunk_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    model, device = load_clean_eat(Path(args.eat_model_path), args.device)
    _, train_items, test_items = expected_items(Path(args.data_root), args.machine)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)} test={len(test_items)}")

    train_paths = [str(item.path) for item in train_items]
    test_paths = [str(item.path) for item in test_items]
    y_test = np.asarray([int(item.label) for item in test_items], dtype=np.int64)
    d_test = np.asarray([item.domain for item in test_items], dtype="U16")

    train_frames_per_file = []
    test_frames_per_file = []
    train_frame_counts = []
    test_frame_counts = []
    sample_rate = None
    frame_dim = None

    for item in train_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        frames, final_sr = eat_frames(model, wav, int(sr), device)
        train_frames_per_file.append(frames)
        train_frame_counts.append(int(frames.shape[0]))
        sample_rate = final_sr
        frame_dim = int(frames.shape[1])

    for item in test_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        frames, final_sr = eat_frames(model, wav, int(sr), device)
        test_frames_per_file.append(frames)
        test_frame_counts.append(int(frames.shape[0]))
        sample_rate = final_sr
        frame_dim = int(frames.shape[1])

    cache_rows = []
    memory_rows = []
    for cfg in CHUNK_CONFIGS:
        train_chunks_per_file = [
            chunk_embeddings_from_frames(frames, cfg["chunk_len"], cfg["chunk_stride"])
            for frames in train_frames_per_file
        ]
        test_chunks_per_file = [
            chunk_embeddings_from_frames(frames, cfg["chunk_len"], cfg["chunk_stride"])
            for frames in test_frames_per_file
        ]
        train_flat, train_offsets, train_counts = flatten_with_offsets(train_chunks_per_file)
        test_flat, test_offsets, test_counts = flatten_with_offsets(test_chunks_per_file)
        cache_path = chunk_cache_path(out_root, cfg["name"], args.machine)

        np.savez_compressed(
            cache_path,
            train_chunk_embeddings=train_flat.astype(np.float32),
            train_chunk_offsets=train_offsets.astype(np.int32),
            train_chunk_counts=train_counts.astype(np.int32),
            train_frame_counts=np.asarray(train_frame_counts, dtype=np.int32),
            train_paths=np.asarray(train_paths, dtype=object),
            test_chunk_embeddings=test_flat.astype(np.float32),
            test_chunk_offsets=test_offsets.astype(np.int32),
            test_chunk_counts=test_counts.astype(np.int32),
            test_frame_counts=np.asarray(test_frame_counts, dtype=np.int32),
            test_paths=np.asarray(test_paths, dtype=object),
            y_test=y_test.astype(np.int64),
            d_test=d_test.astype("U16"),
            chunk_config_name=cfg["name"],
            chunk_len=int(cfg["chunk_len"]),
            chunk_stride=int(cfg["chunk_stride"]),
            embedding_dim=1536,
            frame_dim=int(frame_dim),
            sample_rate=int(sample_rate),
            eat_model_path=str(args.eat_model_path),
            machine=args.machine,
        )

        cache_rows.append(
            {
                "machine": args.machine,
                "config_name": cfg["name"],
                "chunk_len": int(cfg["chunk_len"]),
                "chunk_stride": int(cfg["chunk_stride"]),
                "train_files": len(train_items),
                "test_files": len(test_items),
                "train_total_chunks": int(train_flat.shape[0]),
                "test_total_chunks": int(test_flat.shape[0]),
                "train_embedding_shape": list(train_flat.shape),
                "test_embedding_shape": list(test_flat.shape),
                "sample_rate": int(sample_rate),
                "frame_dim": int(frame_dim),
                "cache_path": str(cache_path),
            }
        )
        memory_rows.append(
            {
                "machine": args.machine,
                "config_name": cfg["name"],
                "chunk_len": int(cfg["chunk_len"]),
                "chunk_stride": int(cfg["chunk_stride"]),
                "train_total_chunks": int(train_flat.shape[0]),
                "mean_train_chunks_per_file": float(np.mean(train_counts)),
                "median_train_chunks_per_file": float(np.median(train_counts)),
                "max_train_chunks_per_file": int(np.max(train_counts)),
                "mean_test_chunks_per_file": float(np.mean(test_counts)),
                "median_test_chunks_per_file": float(np.median(test_counts)),
                "max_test_chunks_per_file": int(np.max(test_counts)),
            }
        )

    write_csv(out_root / "chunk_cache_summary.csv", cache_rows)
    write_csv(out_root / "chunk_memory_summary.csv", memory_rows)
    write_markdown(Path(args.doc_path), build_extract_markdown(cache_rows))


def run_score_chunk(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    tail_rows = []
    geom_rows = []
    best_chunk_row = None
    y_ref = None
    d_ref = None
    paths_ref = None
    p1_runtime_bundle = runtime_p1_reference(Path(args.pooling_root), args.machine)

    for cfg in CHUNK_CONFIGS:
        cache_path = chunk_cache_path(out_root, cfg["name"], args.machine)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing chunk cache: {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)

        train_memory = arr["train_chunk_embeddings"].astype(np.float32)
        test_chunks = arr["test_chunk_embeddings"].astype(np.float32)
        test_offsets = arr["test_chunk_offsets"].astype(np.int32)
        test_counts = arr["test_chunk_counts"].astype(np.int32)
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype(str)
        test_paths = arr["test_paths"].astype(str)

        if y_ref is None:
            y_ref = y_test
            d_ref = d_test
            paths_ref = test_paths
        else:
            if not np.array_equal(y_ref, y_test) or not np.array_equal(d_ref, d_test) or not np.array_equal(paths_ref, test_paths):
                raise RuntimeError(f"Alignment failed across chunk caches for {cfg['name']}")

        per_candidate_scores = {rule: [] for rule in AGGREGATION_RULES}
        for file_idx in range(len(test_counts)):
            start = int(test_offsets[file_idx])
            end = int(test_offsets[file_idx + 1])
            file_chunks = test_chunks[start:end]
            chunk_scores = chunk_knn_scores(train_memory, file_chunks, k=args.latent_knn_k, batch_size=args.distance_chunk_size)
            for rule in AGGREGATION_RULES:
                per_candidate_scores[rule].append(aggregate_chunk_scores(chunk_scores, rule))

        for rule in AGGREGATION_RULES:
            candidate_name = f"{cfg['name']}_{rule}"
            file_scores = np.asarray(per_candidate_scores[rule], dtype=np.float64)
            summary, tail, geom = evaluate(file_scores, y_test, d_test)
            row = {
                "candidate": candidate_name,
                "machine": args.machine,
                "score_source": "segment_chunk",
                "chunk_config": cfg["name"],
                "aggregation_rule": rule,
                **summary,
            }
            candidate_rows.append(row)
            tail_rows.append({"candidate": candidate_name, "machine": args.machine, **tail})
            geom_rows.append({"candidate": candidate_name, "machine": args.machine, **geom})

            candidate_dir = scores_root / candidate_name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                candidate_dir / f"{args.machine}_scores.npz",
                test_scores=file_scores.astype(np.float64),
                y_test=y_test.astype(np.int64),
                d_test=d_test.astype("U16"),
                test_paths=np.asarray(test_paths, dtype=object),
                score_mode=candidate_name,
                machine=args.machine,
                chunk_config=cfg["name"],
                aggregation_rule=rule,
                source_chunk_cache=str(cache_path),
                latent_knn_k=int(args.latent_knn_k),
            )
            if best_chunk_row is None or row["hmean"] > best_chunk_row["hmean"]:
                best_chunk_row = row

    if best_chunk_row is None:
        raise RuntimeError("No chunk candidates were scored.")

    runtime_p1_row = None
    if p1_runtime_bundle is not None:
        y_match = np.array_equal(p1_runtime_bundle["y"], y_ref)
        d_match = np.array_equal(p1_runtime_bundle["d"], d_ref)
        p_match = np.array_equal(p1_runtime_bundle["paths"], paths_ref) if p1_runtime_bundle["paths"] is not None else bool(y_match and d_match)
        if not (y_match and d_match and p_match):
            raise RuntimeError(f"P1 runtime baseline alignment failed: y={y_match} d={d_match} paths={p_match}")
        p1_summary, p1_tail, p1_geom = evaluate(p1_runtime_bundle["scores"], y_ref, d_ref)
        runtime_p1_row = {
            "candidate": "P1_mean_std",
            "machine": args.machine,
            "score_source": "runtime_baseline",
            "chunk_config": "",
            "aggregation_rule": "",
            **p1_summary,
        }
        candidate_rows.append(runtime_p1_row)
        tail_rows.append({"candidate": "P1_mean_std", "machine": args.machine, **p1_tail})
        geom_rows.append({"candidate": "P1_mean_std", "machine": args.machine, **p1_geom})

    write_csv(out_root / "summary.csv", candidate_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    p1_ref_hmean = runtime_p1_row["hmean"] if runtime_p1_row is not None else REGISTERED_P1_REFERENCE["hmean"]
    p1_ref_pauc = runtime_p1_row["pAUC"] if runtime_p1_row is not None else REGISTERED_P1_REFERENCE["pAUC"]
    p1_ref_tpr99 = runtime_p1_row["TPR@99n"] if runtime_p1_row is not None else REGISTERED_P1_REFERENCE["TPR@99n"]

    promising = bool(
        best_chunk_row["hmean"] >= 0.575941
        and best_chunk_row["pAUC"] >= 0.520000
        and best_chunk_row["TPR@99n"] >= 0.060000
    )
    strong_positive = bool(
        best_chunk_row["hmean"] >= 0.585000
        and best_chunk_row["pAUC"] >= 0.525000
        and best_chunk_row["TPR@99n"] >= 0.080000
    )
    negative_hmean = bool(best_chunk_row["hmean"] < REGISTERED_F_P1_S3_050_REFERENCE["hmean"])
    negative_pauc = bool(best_chunk_row["pAUC"] < p1_ref_pauc)
    negative_tpr99 = bool(best_chunk_row["TPR@99n"] < p1_ref_tpr99)
    negative = bool(negative_hmean or negative_pauc or negative_tpr99)

    comparison_row = {
        "machine": args.machine,
        "best_chunk_candidate": best_chunk_row["candidate"],
        "best_hmean": best_chunk_row["hmean"],
        "best_pAUC": best_chunk_row["pAUC"],
        "best_TPR@99n": best_chunk_row["TPR@99n"],
        "runtime_p1_available": runtime_p1_row is not None,
        "runtime_p1_hmean": runtime_p1_row["hmean"] if runtime_p1_row is not None else "",
        "runtime_p1_pAUC": runtime_p1_row["pAUC"] if runtime_p1_row is not None else "",
        "runtime_p1_TPR@99n": runtime_p1_row["TPR@99n"] if runtime_p1_row is not None else "",
        "registered_P1_hmean": REGISTERED_P1_REFERENCE["hmean"],
        "registered_P1_pAUC": REGISTERED_P1_REFERENCE["pAUC"],
        "registered_P1_TPR@99n": REGISTERED_P1_REFERENCE["TPR@99n"],
        "registered_F_P1_A1_065_hmean": REGISTERED_F_P1_A1_065_REFERENCE["hmean"],
        "registered_F_P1_A1_065_pAUC": REGISTERED_F_P1_A1_065_REFERENCE["pAUC"],
        "registered_F_P1_A1_065_TPR@99n": REGISTERED_F_P1_A1_065_REFERENCE["TPR@99n"],
        "registered_F_P1_S3_050_hmean": REGISTERED_F_P1_S3_050_REFERENCE["hmean"],
        "registered_F_P1_S3_050_pAUC": REGISTERED_F_P1_S3_050_REFERENCE["pAUC"],
        "registered_F_P1_S3_050_TPR@99n": REGISTERED_F_P1_S3_050_REFERENCE["TPR@99n"],
        "delta_hmean_vs_P1_ref": best_chunk_row["hmean"] - p1_ref_hmean,
        "delta_pAUC_vs_P1_ref": best_chunk_row["pAUC"] - p1_ref_pauc,
        "delta_TPR99_vs_P1_ref": best_chunk_row["TPR@99n"] - p1_ref_tpr99,
        "delta_hmean_vs_F_P1_S3_050": best_chunk_row["hmean"] - REGISTERED_F_P1_S3_050_REFERENCE["hmean"],
        "delta_pAUC_vs_F_P1_S3_050": best_chunk_row["pAUC"] - REGISTERED_F_P1_S3_050_REFERENCE["pAUC"],
        "delta_TPR99_vs_F_P1_S3_050": best_chunk_row["TPR@99n"] - REGISTERED_F_P1_S3_050_REFERENCE["TPR@99n"],
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "negative_hmean_below_rawframe_best": negative_hmean,
        "negative_pAUC_below_P1_ref": negative_pauc,
        "negative_TPR99_below_P1_ref": negative_tpr99,
    }
    write_csv(out_root / "reference_comparison.csv", [comparison_row])

    verdict = {
        "machine": args.machine,
        "best_chunk_candidate": best_chunk_row["candidate"],
        "best_hmean": float(best_chunk_row["hmean"]),
        "best_pAUC": float(best_chunk_row["pAUC"]),
        "best_TPR@99n": float(best_chunk_row["TPR@99n"]),
        "runtime_p1_available": bool(runtime_p1_row is not None),
        "p1_reference_used_for_verdict": "runtime_P1_mean_std" if runtime_p1_row is not None else "registered_P1_mean_std",
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "negative_reasons": {
            "hmean_below_previous_rawframe_best": negative_hmean,
            "pAUC_below_P1_reference": negative_pauc,
            "TPR99_below_P1_reference": negative_tpr99,
        },
        "registered_references": {
            "P1_mean_std": REGISTERED_P1_REFERENCE,
            "F_P1_A1_065": REGISTERED_F_P1_A1_065_REFERENCE,
            "F_P1_S3_050": REGISTERED_F_P1_S3_050_REFERENCE,
        },
        "notes": [
            "Gearbox one-machine sanity only.",
            "Do not automatically promote to toycar/fan or 7-machine.",
            "Human review required after gearbox.",
        ],
    }
    write_json(out_root / "segment_chunk_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_score_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_chunk_cache":
        run_extract_chunk_cache(args)
        return
    if args.mode == "score_chunk":
        run_score_chunk(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
