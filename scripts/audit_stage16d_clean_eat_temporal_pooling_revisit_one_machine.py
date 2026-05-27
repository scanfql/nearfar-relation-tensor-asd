#!/usr/bin/env python
"""Stage 16D: clean EAT temporal pooling revisit one-machine audit.

This script is strictly frozen-EAT and training-free.

Allowed:
- local clean EAT load for embedding extraction only
- frame output extraction from clean EAT [T, 768]
- fixed preprocessing on frame embeddings
- fixed temporal pooling candidates
- source train-normal-only memory
- fixed cosine kNN scoring

Blocked:
- no training
- no fine-tuning
- no LoRA
- no adapter
- no projection head
- no Conformer
- no A1/V63 fusion in this first sanity
- no alpha sweep
- no per-machine route
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
P1_SCORE_DEFAULT = "results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz"
OUTPUT_ROOT_DEFAULT = "results/stage16d_clean_eat_temporal_pooling_revisit_one_machine_gearbox"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE16D_CLEAN_EAT_TEMPORAL_POOLING_REVISIT_ONE_MACHINE_GEARBOX_AUDIT.md"
MACHINE_DEFAULT = "gearbox"

PREPROCESS_VARIANTS = [
    "none",
    "min_clamp",
    "spike_suppression",
    "min_clamp_spike_suppression",
]
POOLING_CANDIDATES = [
    "mean",
    "mean_std",
    "gem_p3",
    "rdp_g1",
    "rdp_gem_p3_g1",
]

REGISTERED_REFERENCES = {
    "P1_mean_std": {"hmean": 0.561801, "pAUC": 0.493158, "TPR@99n": 0.020000},
    "F_P1_A1_065": {"hmean": 0.572219, "pAUC": 0.519474, "TPR@99n": 0.030000},
    "F_P1_S3_050": {"hmean": 0.575941, "pAUC": 0.525789, "TPR@99n": 0.080000},
    "C16S08_top1": {"hmean": 0.560879, "pAUC": 0.544211, "TPR@99n": 0.090000},
    "STAGE16C3_best_fixed_fusion": {"hmean": 0.571952, "pAUC": 0.505000, "TPR@99n": 0.030000},
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage16D clean EAT temporal pooling revisit one-machine audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_pooling_embeddings", "score_pooling"])
    p.add_argument("--eat_model_path", default=MODEL_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--p1_score_path", default=P1_SCORE_DEFAULT)
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
    for name in ["torch", "torchaudio", "soundfile", "librosa", "transformers", "timm", "safetensors", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


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


def min_clamp(frames: np.ndarray) -> np.ndarray:
    return np.where(np.abs(frames) < 0.1, 0.0, frames).astype(np.float32)


def spike_suppression(frames: np.ndarray) -> np.ndarray:
    out = frames.astype(np.float32, copy=True)
    pos_mask = out > 0.5
    neg_mask = out < -0.5
    out[pos_mask] = 0.5 + np.tanh(out[pos_mask] - 0.5)
    out[neg_mask] = -0.5 + np.tanh(out[neg_mask] + 0.5)
    return out.astype(np.float32)


def preprocess_frames(frames: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frames.astype(np.float32, copy=True)
    if mode == "min_clamp":
        return min_clamp(frames)
    if mode == "spike_suppression":
        return spike_suppression(frames)
    if mode == "min_clamp_spike_suppression":
        return spike_suppression(min_clamp(frames))
    raise ValueError(f"Unknown preprocess mode: {mode}")


def rdp_weights(frames: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    mean = frames.mean(axis=0, dtype=np.float64)
    dev = np.linalg.norm(frames.astype(np.float64) - mean[None, :], axis=1)
    max_dev = float(np.max(dev)) if dev.size else 0.0
    if max_dev <= 0.0:
        d_hat = np.zeros((frames.shape[0],), dtype=np.float64)
    else:
        d_hat = dev / max_dev
    numer = np.power(1.0 + d_hat, gamma)
    denom = float(np.sum(numer))
    if denom <= 0.0:
        return np.full((frames.shape[0],), 1.0 / max(frames.shape[0], 1), dtype=np.float64)
    return (numer / denom).astype(np.float64)


def gem_pool(frames: np.ndarray, p: float = 3.0) -> np.ndarray:
    nonneg = np.maximum(frames.astype(np.float64), 0.0)
    return np.power(np.mean(np.power(nonneg, p), axis=0), 1.0 / p).astype(np.float64)


def weighted_gem_pool(frames: np.ndarray, weights: np.ndarray, p: float = 3.0) -> np.ndarray:
    nonneg = np.maximum(frames.astype(np.float64), 0.0)
    w = weights.astype(np.float64)
    denom = float(np.sum(w))
    if denom <= 0.0:
        raise RuntimeError("Weighted GeM received non-positive weight sum.")
    pooled = np.sum(w[:, None] * np.power(nonneg, p), axis=0) / denom
    return np.power(pooled, 1.0 / p).astype(np.float64)


def pool_frames(frames: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return frames.mean(axis=0, dtype=np.float64).astype(np.float64)
    if pooling == "mean_std":
        mean = frames.mean(axis=0, dtype=np.float64)
        std = frames.std(axis=0, dtype=np.float64)
        return np.concatenate([mean, std]).astype(np.float64)
    if pooling == "gem_p3":
        return gem_pool(frames, p=3.0)
    if pooling == "rdp_g1":
        weights = rdp_weights(frames, gamma=1.0)
        return np.sum(weights[:, None] * frames.astype(np.float64), axis=0).astype(np.float64)
    if pooling == "rdp_gem_p3_g1":
        weights = rdp_weights(frames, gamma=1.0)
        return weighted_gem_pool(frames, weights, p=3.0)
    raise ValueError(f"Unknown pooling: {pooling}")


def candidate_name(preprocess: str, pooling: str) -> str:
    return f"{preprocess}__{pooling}"


def p1_runtime_score_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    return Path(path_text)


def load_score_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    if "test_scores" in arr.files:
        scores = arr["test_scores"].astype(np.float64)
    elif "scores" in arr.files:
        scores = arr["scores"].astype(np.float64)
    else:
        raise KeyError(f"{npz_path} has no test_scores/scores key; keys={arr.files}")
    if "d_test" in arr.files:
        domains = arr["d_test"].astype(str)
    elif "domains" in arr.files:
        domains = arr["domains"].astype(str)
    else:
        raise KeyError(f"{npz_path} has no d_test/domains key; keys={arr.files}")
    return {
        "path": str(npz_path),
        "scores": scores,
        "y": arr["y_test"].astype(np.int64),
        "d": domains,
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5) -> np.ndarray:
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, train_n.shape[0])
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


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


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage16D Clean EAT Temporal Pooling Revisit One-Machine Audit",
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
        f"- runtime P1 score exists: `{status['p1_score_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_extract_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "# CTAF Stage16D Clean EAT Temporal Pooling Revisit One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `extract_pooling_embeddings`",
        "- frozen clean EAT only",
        "- no scoring",
        "",
        "## Candidate Embedding Summary",
        "",
    ]
    for row in rows[:8]:
        lines.append(
            f"- `{row['candidate']}` dim=`{row['embedding_dim']}` train=`{row['train_count']}` test=`{row['test_count']}`"
        )
    return lines


def build_score_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage16D Clean EAT Temporal Pooling Revisit One-Machine Audit",
        "",
        "## Best Candidate",
        "",
        f"- machine: `{verdict['machine']}`",
        f"- best candidate: `{verdict['best_candidate']}`",
        f"- hmean: `{verdict['best_hmean']:.6f}`",
        f"- pAUC: `{verdict['best_pAUC']:.6f}`",
        f"- TPR@99n: `{verdict['best_TPR@99n']:.6f}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- no A1/V63 fusion",
        "- no alpha sweep",
        "- no per-machine selector",
        "- gearbox only",
        "- human review required before any promotion",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    model_path = Path(args.eat_model_path)
    p1_path = p1_runtime_score_path(args.p1_score_path)
    items, train_items, test_items = expected_items(data_root, args.machine) if data_root.exists() else ([], [], [])
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
        "p1_score_path": str(p1_path) if p1_path is not None else "",
        "p1_score_exists": bool(p1_path.exists()) if p1_path is not None else False,
        "imports": import_status(),
        "notes": "No model load, no forward, no extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", status)
    write_csv(
        out_root / "machine_check.csv",
        [{
            "machine": args.machine,
            "machine_exists": len(items) > 0,
            "source_train_normal_count": len(train_items),
            "test_count": len(test_items),
            "eat_model_path_exists": model_path.exists(),
            "data_root_exists": data_root.exists(),
            "p1_score_exists": bool(p1_path.exists()) if p1_path is not None else False,
        }],
    )
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_extract_pooling_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    model, device = load_clean_eat(Path(args.eat_model_path), args.device)
    _, train_items, test_items = expected_items(Path(args.data_root), args.machine)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)} test={len(test_items)}")

    train_frames = []
    test_frames = []
    y_test = []
    d_test = []
    test_paths = []
    frame_lengths = []
    sample_rate = None

    for item in train_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        frames, final_sr = eat_frames(model, wav, int(sr), device)
        train_frames.append(frames)
        sample_rate = final_sr

    for item in test_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        frames, final_sr = eat_frames(model, wav, int(sr), device)
        test_frames.append(frames)
        frame_lengths.append(int(frames.shape[0]))
        y_test.append(int(item.label))
        d_test.append(item.domain)
        test_paths.append(str(item.path))
        sample_rate = final_sr

    y_test_arr = np.asarray(y_test, dtype=np.int64)
    d_test_arr = np.asarray(d_test, dtype="U16")
    test_paths_arr = np.asarray(test_paths, dtype=object)

    rows = []
    for preprocess in PREPROCESS_VARIANTS:
        proc_train = [preprocess_frames(frames, preprocess) for frames in train_frames]
        proc_test = [preprocess_frames(frames, preprocess) for frames in test_frames]
        for pooling in POOLING_CANDIDATES:
            cand = candidate_name(preprocess, pooling)
            train_emb = np.stack([pool_frames(frames, pooling) for frames in proc_train], axis=0).astype(np.float64)
            test_emb = np.stack([pool_frames(frames, pooling) for frames in proc_test], axis=0).astype(np.float64)
            np.savez_compressed(
                emb_root / f"{cand}_{args.machine}_embeddings.npz",
                train_embeddings=train_emb,
                test_embeddings=test_emb,
                y_test=y_test_arr,
                d_test=d_test_arr,
                test_paths=test_paths_arr,
                frame_lengths=np.asarray(frame_lengths, dtype=np.int32),
                machine=args.machine,
                preprocess=preprocess,
                pooling=pooling,
                embedding_dim=int(train_emb.shape[1]),
                sample_rate=int(sample_rate),
                eat_model_path=str(args.eat_model_path),
            )
            rows.append({
                "machine": args.machine,
                "candidate": cand,
                "preprocess": preprocess,
                "pooling": pooling,
                "embedding_dim": int(train_emb.shape[1]),
                "train_count": int(train_emb.shape[0]),
                "test_count": int(test_emb.shape[0]),
                "sample_rate": int(sample_rate),
                "mean_frame_length_test": float(np.mean(frame_lengths)),
                "embedding_path": str(emb_root / f"{cand}_{args.machine}_embeddings.npz"),
            })

    write_csv(out_root / "pooling_embedding_summary.csv", rows)
    write_markdown(Path(args.doc_path), build_extract_markdown(rows))


def run_score_pooling(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    tail_rows = []
    geom_rows = []
    best_row = None
    y_ref = None
    d_ref = None
    paths_ref = None

    for preprocess in PREPROCESS_VARIANTS:
        for pooling in POOLING_CANDIDATES:
            cand = candidate_name(preprocess, pooling)
            emb_path = emb_root / f"{cand}_{args.machine}_embeddings.npz"
            if not emb_path.exists():
                raise FileNotFoundError(f"Missing embedding file: {emb_path}")
            arr = np.load(emb_path, allow_pickle=True)
            train_emb = arr["train_embeddings"].astype(np.float64)
            test_emb = arr["test_embeddings"].astype(np.float64)
            y_test = arr["y_test"].astype(np.int64)
            d_test = arr["d_test"].astype(str)
            test_paths = arr["test_paths"].astype(str)

            if y_ref is None:
                y_ref = y_test
                d_ref = d_test
                paths_ref = test_paths
            else:
                if not np.array_equal(y_ref, y_test) or not np.array_equal(d_ref, d_test) or not np.array_equal(paths_ref, test_paths):
                    raise RuntimeError(f"Alignment failed across embedding candidates for {cand}")

            scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
            summary, tail, geom = evaluate(scores, y_test, d_test)
            row = {"candidate": cand, "machine": args.machine, **summary}
            summary_rows.append(row)
            tail_rows.append({"candidate": cand, "machine": args.machine, **tail})
            geom_rows.append({"candidate": cand, "machine": args.machine, **geom})

            cand_dir = scores_root / cand
            cand_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cand_dir / f"{args.machine}_scores.npz",
                test_scores=scores.astype(np.float64),
                y_test=y_test.astype(np.int64),
                d_test=d_test.astype("U16"),
                test_paths=np.asarray(test_paths, dtype=object),
                score_mode=cand,
                machine=args.machine,
                latent_knn_k=int(args.latent_knn_k),
            )

            if best_row is None or row["hmean"] > best_row["hmean"]:
                best_row = row

    p1_runtime_row = None
    p1_ref = REGISTERED_REFERENCES["P1_mean_std"]
    p1_path = p1_runtime_score_path(args.p1_score_path)
    if p1_path is not None and p1_path.exists():
        bundle = load_score_bundle(p1_path)
        y_match = np.array_equal(bundle["y"], y_ref)
        d_match = np.array_equal(bundle["d"], d_ref)
        if bundle["paths"] is not None:
            p_match = np.array_equal(bundle["paths"], paths_ref)
        else:
            p_match = bool(y_match and d_match)
        if not (y_match and d_match and p_match):
            raise RuntimeError(f"Runtime P1 score alignment failed: y={y_match} d={d_match} paths={p_match}")
        p1_summary, p1_tail, p1_geom = evaluate(bundle["scores"], y_ref, d_ref)
        p1_runtime_row = {"candidate": "P1_mean_std", "machine": args.machine, **p1_summary}
        summary_rows.append(p1_runtime_row)
        tail_rows.append({"candidate": "P1_mean_std", "machine": args.machine, **p1_tail})
        geom_rows.append({"candidate": "P1_mean_std", "machine": args.machine, **p1_geom})
        p1_ref = {
            "hmean": p1_runtime_row["hmean"],
            "pAUC": p1_runtime_row["pAUC"],
            "TPR@99n": p1_runtime_row["TPR@99n"],
        }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    promising = bool(
        best_row["hmean"] >= 0.575941
        and best_row["pAUC"] >= 0.520000
        and best_row["TPR@99n"] >= 0.060000
    )
    strong_positive = bool(
        best_row["hmean"] >= 0.585000
        and best_row["pAUC"] >= 0.525000
        and best_row["TPR@99n"] >= 0.080000
    )
    negative = bool(
        best_row["hmean"] <= REGISTERED_REFERENCES["F_P1_S3_050"]["hmean"]
        or best_row["pAUC"] < p1_ref["pAUC"]
        or best_row["TPR@99n"] < p1_ref["TPR@99n"]
    )

    comparison_row = {
        "machine": args.machine,
        "best_candidate": best_row["candidate"],
        "best_hmean": best_row["hmean"],
        "best_pAUC": best_row["pAUC"],
        "best_TPR@99n": best_row["TPR@99n"],
        "runtime_P1_loaded": bool(p1_runtime_row is not None),
        "P1_reference_hmean": p1_ref["hmean"],
        "P1_reference_pAUC": p1_ref["pAUC"],
        "P1_reference_TPR@99n": p1_ref["TPR@99n"],
        "registered_F_P1_A1_065_hmean": REGISTERED_REFERENCES["F_P1_A1_065"]["hmean"],
        "registered_F_P1_A1_065_pAUC": REGISTERED_REFERENCES["F_P1_A1_065"]["pAUC"],
        "registered_F_P1_A1_065_TPR@99n": REGISTERED_REFERENCES["F_P1_A1_065"]["TPR@99n"],
        "registered_F_P1_S3_050_hmean": REGISTERED_REFERENCES["F_P1_S3_050"]["hmean"],
        "registered_F_P1_S3_050_pAUC": REGISTERED_REFERENCES["F_P1_S3_050"]["pAUC"],
        "registered_F_P1_S3_050_TPR@99n": REGISTERED_REFERENCES["F_P1_S3_050"]["TPR@99n"],
        "registered_C16S08_top1_hmean": REGISTERED_REFERENCES["C16S08_top1"]["hmean"],
        "registered_C16S08_top1_pAUC": REGISTERED_REFERENCES["C16S08_top1"]["pAUC"],
        "registered_C16S08_top1_TPR@99n": REGISTERED_REFERENCES["C16S08_top1"]["TPR@99n"],
        "registered_STAGE16C3_best_fixed_fusion_hmean": REGISTERED_REFERENCES["STAGE16C3_best_fixed_fusion"]["hmean"],
        "registered_STAGE16C3_best_fixed_fusion_pAUC": REGISTERED_REFERENCES["STAGE16C3_best_fixed_fusion"]["pAUC"],
        "registered_STAGE16C3_best_fixed_fusion_TPR@99n": REGISTERED_REFERENCES["STAGE16C3_best_fixed_fusion"]["TPR@99n"],
        "delta_hmean_vs_F_P1_S3_050": best_row["hmean"] - REGISTERED_REFERENCES["F_P1_S3_050"]["hmean"],
        "delta_pAUC_vs_F_P1_S3_050": best_row["pAUC"] - REGISTERED_REFERENCES["F_P1_S3_050"]["pAUC"],
        "delta_TPR99_vs_F_P1_S3_050": best_row["TPR@99n"] - REGISTERED_REFERENCES["F_P1_S3_050"]["TPR@99n"],
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
    }
    write_csv(out_root / "reference_comparison.csv", [comparison_row])

    verdict = {
        "machine": args.machine,
        "best_candidate": best_row["candidate"],
        "best_hmean": float(best_row["hmean"]),
        "best_pAUC": float(best_row["pAUC"]),
        "best_TPR@99n": float(best_row["TPR@99n"]),
        "runtime_P1_loaded": bool(p1_runtime_row is not None),
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "registered_references": REGISTERED_REFERENCES,
        "notes": [
            "Gearbox one-machine sanity only.",
            "Do not automatically promote to toycar/fan or 7-machine.",
            "Human review required after gearbox.",
        ],
    }
    write_json(out_root / "temporal_pooling_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_score_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_pooling_embeddings":
        run_extract_pooling_embeddings(args)
        return
    if args.mode == "score_pooling":
        run_score_pooling(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
