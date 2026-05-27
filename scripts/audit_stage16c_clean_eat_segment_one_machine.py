#!/usr/bin/env python
"""Stage 16C-1A: clean EAT segment-level one-machine sanity.

This script tests whether frame/segment-level clean EAT scores improve a bottleneck
machine, especially toycar.

It does NOT:
- train models
- fine-tune EAT
- use LoRA / adapters
- search weights
- use per-machine oracle routing
- use anomaly labels for model selection

It DOES:
- load clean EAT locally
- extract frame-level EAT outputs [B, T, 768] for one machine
- build source train-normal frame memory
- compute frame-level kNN anomaly scores
- aggregate frame scores with fixed rules:
  S1_frame_max
  S2_frame_top5mean
  S3_frame_top10mean
- optionally fuse fixed segment scores with existing P1_mean_std:
  F_P1_S1_050
  F_P1_S2_050
  F_P1_S3_050
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


def parse_args():
    p = argparse.ArgumentParser(description="Stage16C clean EAT segment-level one-machine sanity")
    p.add_argument("--mode", required=True, choices=["check", "extract_frame_cache", "segment_score"])
    p.add_argument("--eat_model_path", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--machine", default="toycar")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--pooling_root", default="results/stage16b_clean_eat_pooling_fusion")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--stage14a_root", default="results/stage14a_frozen_beats_v63_fusion")
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--train_frame_stride", type=int, default=8)
    p.add_argument("--test_frame_stride", type=int, default=1)
    p.add_argument("--frame_chunk_size", type=int, default=64)
    return p.parse_args()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def import_status():
    status = {}
    for name in ["torch", "torchaudio", "soundfile", "librosa", "transformers", "timm", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray):
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95))
    q99 = float(np.quantile(normals, 0.99))
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "TPR@95n": float(np.mean(anomalies > q95)),
        "TPR@99n": float(np.mean(anomalies > q99)),
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray):
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    return {
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


def load_clean_eat(model_path: Path, device: str):
    import torch

    package_parent = model_path.parent.resolve()
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception:
        from checkpoints.configuration_eat import EATConfig
        from checkpoints.modeling_eat import EATModel
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


def load_score_bundle(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing score file: {npz_path}")
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "scores": arr["test_scores"].astype(np.float64),
        "y": arr["y_test"].astype(np.int64),
        "d": arr[domains_key].astype(str),
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        "path": str(npz_path),
    }


def p1_score_path(pooling_root: Path, machine: str) -> Path:
    return pooling_root / "scores" / "P1_mean_std" / f"{machine}_scores.npz"


def v63_score_path(v63_root: Path, seed: int, machine: str) -> Path:
    return v63_root / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"


def a1_score_path(stage14a_root: Path, seed: int, machine: str) -> Path:
    return (
        stage14a_root
        / "a1_v63_beats_fusion"
        / f"seed_{seed}"
        / "scores"
        / f"{machine}_v63_beats_rankfusion_scores.npz"
    )


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def frame_knn_scores(
    train_frames: np.ndarray,
    test_frames: np.ndarray,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    """Compute per-test-frame kNN scores against train frame memory.

    train_frames: [N_train_frames, D]
    test_frames: [N_test_frames, D]
    return: [N_test_frames]
    """
    train = l2_normalize(train_frames.astype(np.float32), axis=1)
    test = l2_normalize(test_frames.astype(np.float32), axis=1)

    k = min(k, train.shape[0])
    out = []

    train_norm2 = np.sum(train * train, axis=1)[None, :]

    for start in range(0, test.shape[0], chunk_size):
        chunk = test[start:start + chunk_size]
        chunk_norm2 = np.sum(chunk * chunk, axis=1, keepdims=True)
        d2 = chunk_norm2 + train_norm2 - 2.0 * (chunk @ train.T)
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
        out.append(np.sqrt(nearest).mean(axis=1).astype(np.float64))

    return np.concatenate(out, axis=0)


def aggregate_frame_scores(frame_scores: np.ndarray):
    n = frame_scores.size
    top5_n = max(1, int(np.ceil(0.05 * n)))
    top10_n = max(1, int(np.ceil(0.10 * n)))
    sorted_scores = np.sort(frame_scores)[::-1]
    return {
        "S1_frame_max": float(sorted_scores[0]),
        "S2_frame_top5mean": float(np.mean(sorted_scores[:top5_n])),
        "S3_frame_top10mean": float(np.mean(sorted_scores[:top10_n])),
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    model_path = Path(args.eat_model_path)
    machine = args.machine

    items, train_items, test_items = expected_items(data_root, machine) if data_root.exists() else ([], [], [])

    paths = {
        "P1_mean_std": p1_score_path(Path(args.pooling_root), machine),
        "A1": a1_score_path(Path(args.stage14a_root), args.seed, machine),
        "V63": v63_score_path(Path(args.v63_root), args.seed, machine),
    }

    status = {
        "mode": "check",
        "machine": machine,
        "eat_model_path": str(model_path),
        "eat_model_path_exists": model_path.exists(),
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "num_machine_items": len(items),
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "train_frame_stride": args.train_frame_stride,
        "test_frame_stride": args.test_frame_stride,
        "latent_knn_k": args.latent_knn_k,
        "frame_chunk_size": args.frame_chunk_size,
        "imports": import_status(),
        "existing_score_files": {k: {"path": str(v), "exists": v.exists()} for k, v in paths.items()},
        "notes": "No model load, no forward, no embedding extraction, no scoring.",
    }

    write_json(out_root / "check_status.json", status)

    write_csv(out_root / "machine_check.csv", [{
        "machine": machine,
        "num_items": len(items),
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "eat_model_path_exists": model_path.exists(),
        "P1_score_exists": paths["P1_mean_std"].exists(),
        "A1_score_exists": paths["A1"].exists(),
        "V63_score_exists": paths["V63"].exists(),
    }])


def run_extract_frame_cache(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "frame_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    machine = args.machine
    data_root = Path(args.data_root)
    model, device = load_clean_eat(Path(args.eat_model_path), args.device)

    _, train_items, test_items = expected_items(data_root, machine)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")

    train_frames_all = []
    train_frame_file_index = []
    test_frames_all = []
    y_test = []
    d_test = []
    test_paths = []
    frame_shape = None
    sample_rate = None

    for file_idx, item in enumerate(train_items):
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        frames, final_sr = eat_frames(model, wav, int(sr), device)
        frame_shape = list(frames.shape)
        sample_rate = final_sr
        frames = frames[::args.train_frame_stride]
        train_frames_all.append(frames)
        train_frame_file_index.extend([file_idx] * frames.shape[0])

    for item in test_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        frames, final_sr = eat_frames(model, wav, int(sr), device)
        frame_shape = list(frames.shape)
        sample_rate = final_sr
        frames = frames[::args.test_frame_stride]
        test_frames_all.append(frames)
        y_test.append(item.label)
        d_test.append(item.domain)
        test_paths.append(str(item.path))

    train_frames = np.concatenate(train_frames_all, axis=0).astype(np.float32)
    test_frames = np.stack(test_frames_all, axis=0).astype(np.float32)

    y_test = np.asarray(y_test, dtype=np.int64)
    d_test = np.asarray(d_test, dtype="U16")
    test_paths = np.asarray(test_paths, dtype=object)
    train_frame_file_index = np.asarray(train_frame_file_index, dtype=np.int32)

    np.savez_compressed(
        cache_dir / f"{machine}_eat_frame_cache.npz",
        train_frames=train_frames,
        test_frames=test_frames,
        train_frame_file_index=train_frame_file_index,
        y_test=y_test,
        d_test=d_test,
        test_paths=test_paths,
        original_frame_shape=frame_shape,
        sample_rate=sample_rate,
        train_frame_stride=args.train_frame_stride,
        test_frame_stride=args.test_frame_stride,
        eat_model_path=str(args.eat_model_path),
    )

    write_csv(out_root / "frame_cache_summary.csv", [{
        "machine": machine,
        "train_files": len(train_items),
        "test_files": len(test_items),
        "original_frame_shape": frame_shape,
        "sample_rate": sample_rate,
        "train_frame_stride": args.train_frame_stride,
        "test_frame_stride": args.test_frame_stride,
        "train_frame_memory_shape": list(train_frames.shape),
        "test_frame_tensor_shape": list(test_frames.shape),
        "cache_path": str(cache_dir / f"{machine}_eat_frame_cache.npz"),
    }])


def run_segment_score(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    machine = args.machine
    cache_path = out_root / "frame_cache" / f"{machine}_eat_frame_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing frame cache: {cache_path}")

    arr = np.load(cache_path, allow_pickle=True)
    train_frames = arr["train_frames"].astype(np.float32)
    test_frames = arr["test_frames"].astype(np.float32)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    test_paths = arr["test_paths"].astype(str)

    p1 = load_score_bundle(p1_score_path(Path(args.pooling_root), machine))
    a1 = load_score_bundle(a1_score_path(Path(args.stage14a_root), args.seed, machine))
    v63 = load_score_bundle(v63_score_path(Path(args.v63_root), args.seed, machine))

    expected_paths = test_paths

    for name, bundle in {"P1_mean_std": p1, "A1": a1, "V63": v63}.items():
        y_match = np.array_equal(bundle["y"], y_test)
        d_match = np.array_equal(bundle["d"], d_test)
        p_match = np.array_equal(bundle["paths"], expected_paths) if bundle["paths"] is not None else bool(y_match and d_match)
        if not (y_match and d_match and p_match):
            raise RuntimeError(f"Alignment failed for {name}: y={y_match}, d={d_match}, paths={p_match}")

    s1 = []
    s2 = []
    s3 = []

    per_file_rows = []

    for i in range(test_frames.shape[0]):
        fs = frame_knn_scores(
            train_frames=train_frames,
            test_frames=test_frames[i],
            k=args.latent_knn_k,
            chunk_size=args.frame_chunk_size,
        )
        ag = aggregate_frame_scores(fs)
        s1.append(ag["S1_frame_max"])
        s2.append(ag["S2_frame_top5mean"])
        s3.append(ag["S3_frame_top10mean"])
        per_file_rows.append({
            "machine": machine,
            "file_index": i,
            "test_path": test_paths[i],
            "label": int(y_test[i]),
            "domain": str(d_test[i]),
            "num_frames": int(fs.size),
            "frame_score_mean": float(np.mean(fs)),
            "frame_score_std": float(np.std(fs)),
            "S1_frame_max": ag["S1_frame_max"],
            "S2_frame_top5mean": ag["S2_frame_top5mean"],
            "S3_frame_top10mean": ag["S3_frame_top10mean"],
        })

    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)

    p1_scores = p1["scores"].astype(np.float64)
    a1_scores = a1["scores"].astype(np.float64)
    v63_scores = v63["scores"].astype(np.float64)

    candidates = {
        "P1_mean_std": p1_scores,
        "A1": a1_scores,
        "V63": v63_scores,
        "F_P1_A1_065": 0.65 * rank_norm(p1_scores) + 0.35 * rank_norm(a1_scores),
        "S1_frame_max": s1,
        "S2_frame_top5mean": s2,
        "S3_frame_top10mean": s3,
        "F_P1_S1_050": 0.5 * rank_norm(p1_scores) + 0.5 * rank_norm(s1),
        "F_P1_S2_050": 0.5 * rank_norm(p1_scores) + 0.5 * rank_norm(s2),
        "F_P1_S3_050": 0.5 * rank_norm(p1_scores) + 0.5 * rank_norm(s3),
        "F_S2_A1_065": 0.65 * rank_norm(s2) + 0.35 * rank_norm(a1_scores),
        "F_S3_A1_065": 0.65 * rank_norm(s3) + 0.35 * rank_norm(a1_scores),
    }

    summary_rows = []
    tail_rows = []

    for cand, scores in candidates.items():
        ev = evaluate(scores, y_test, d_test)
        summary_rows.append({
            "machine": machine,
            "candidate": cand,
            "hmean": ev["hmean"],
            "pAUC": ev["pAUC"],
            "KS": ev["KS"],
            "TPR@95n": ev["TPR@95n"],
            "TPR@99n": ev["TPR@99n"],
        })
        tail_rows.append({
            "machine": machine,
            "candidate": cand,
            "q95": ev["q95"],
            "q99": ev["q99"],
            "anomaly_median": ev["anomaly_median"],
            "anomaly_q75": ev["anomaly_q75"],
            "TPR@95n": ev["TPR@95n"],
            "TPR@99n": ev["TPR@99n"],
        })

        score_dir = scores_root / cand
        score_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            score_dir / f"{machine}_scores.npz",
            test_scores=scores.astype(np.float64),
            y_test=y_test.astype(np.int64),
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            score_mode=cand,
            machine=machine,
            seed=args.seed,
        )

    write_csv(out_root / "per_file_segment_scores.csv", per_file_rows)
    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)

    p1_h = next(r["hmean"] for r in summary_rows if r["candidate"] == "P1_mean_std")
    best = max(summary_rows, key=lambda r: r["hmean"])

    verdict = {
        "machine": machine,
        "baseline_P1_hmean": p1_h,
        "best_candidate": best["candidate"],
        "best_hmean": best["hmean"],
        "best_pAUC": best["pAUC"],
        "best_TPR@99n": best["TPR@99n"],
        "hmean_gain_vs_P1": float(best["hmean"] - p1_h),
        "segment_promising": bool(best["hmean"] - p1_h >= 0.015 or best["TPR@99n"] >= 0.08),
        "notes": "One-machine sanity only. Do not freeze route based on this result.",
    }
    write_json(out_root / "segment_verdict.json", verdict)

    doc_lines = [
        "# CTAF Stage16C Clean EAT Segment One-Machine Audit",
        "",
        f"- machine: `{machine}`",
        f"- seed reference: `{args.seed}`",
        f"- train frame stride: `{args.train_frame_stride}`",
        f"- test frame stride: `{args.test_frame_stride}`",
        f"- frame kNN k: `{args.latent_knn_k}`",
        "",
        "## Best Candidate",
        "",
        f"- candidate: `{best['candidate']}`",
        f"- hmean: `{best['hmean']:.6f}`",
        f"- pAUC: `{best['pAUC']:.6f}`",
        f"- TPR@99n: `{best['TPR@99n']:.6f}`",
        f"- hmean gain vs P1: `{best['hmean'] - p1_h:.6f}`",
        "",
        "## Confirmation",
        "",
        "- no training",
        "- no fine-tuning",
        "- no LoRA",
        "- no adapter",
        "- no per-machine oracle route",
        "- one-machine sanity only",
    ]
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(doc_lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_frame_cache":
        run_extract_frame_cache(args)
        return
    if args.mode == "segment_score":
        run_segment_score(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()