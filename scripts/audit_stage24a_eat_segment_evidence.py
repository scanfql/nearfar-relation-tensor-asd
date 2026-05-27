#!/usr/bin/env python
"""Stage24A: clean EAT segment-level evidence audit.

This is a no-training architecture audit. It tests whether clean EAT frame
outputs contain local anomaly evidence that clip-level min_clamp__mean pooling
averages away.

Fixed protocol:
- clean EAT only
- min_clamp frame preprocessing
- fixed frame chunks: length=16, stride=8
- source train-normal segment memory only
- segment kNN score_k=5
- file score = mean of top 5% segment scores
- no LDN, no selector, no fusion, no tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21e_adapted_eval import (
    evaluate,
    extract_frame_output,
    load_clean_eat_local,
    load_waveform_16k,
    min_clamp,
    waveform_to_eat_tensor,
)


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage24a_eat_segment_evidence"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE24A_EAT_SEGMENT_EVIDENCE.md"
CLIP_REFERENCE_DEFAULT = "results/stage18e_local_density_protocol_full7/per_machine.csv"

CANDIDATE = "eat_segment_knn_top5pct_scorek5"
CLIP_REFERENCE_CANDIDATE = "ldn_diff_K1_scorek5"
CONFIRMED_STAGE22_HMEAN = 0.6166917954459288
CONFIRMED_CLEAN_LDN_HMEAN = 0.6159467288425517
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Stage24A clean EAT segment evidence audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_segment_embeddings", "score_segment_evidence"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--clip_reference_per_machine", default=CLIP_REFERENCE_DEFAULT)
    p.add_argument("--segment_len", type=int, default=16)
    p.add_argument("--segment_stride", type=int, default=8)
    p.add_argument("--score_k", type=int, default=5)
    p.add_argument("--top_percent", type=float, default=5.0)
    p.add_argument("--query_batch_size", type=int, default=256)
    return p.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
        for key in row:
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
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def embedding_path(output_root: Path, machine: str) -> Path:
    return output_root / "embeddings" / f"{machine}_eat_segment_embeddings.npz"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_clip_reference(path: Path) -> dict[str, dict]:
    rows = load_csv(path)
    return {
        row["machine"]: row
        for row in rows
        if row.get("candidate") == CLIP_REFERENCE_CANDIDATE
    }


def chunk_frame_embeddings(frames: np.ndarray, segment_len: int, stride: int) -> np.ndarray:
    if frames.ndim != 2:
        raise RuntimeError(f"Expected [T,D] frames, got {frames.shape}")
    t, d = frames.shape
    if t <= segment_len:
        return frames.mean(axis=0, dtype=np.float64)[None, :].astype(np.float32)
    chunks = []
    for start in range(0, t - segment_len + 1, stride):
        chunk = frames[start:start + segment_len]
        chunks.append(chunk.mean(axis=0, dtype=np.float64).astype(np.float32))
    if (t - segment_len) % stride != 0:
        chunks.append(frames[-segment_len:].mean(axis=0, dtype=np.float64).astype(np.float32))
    return np.stack(chunks, axis=0).astype(np.float32)


def extract_file_segments(model, path: Path, device, segment_len: int, stride: int) -> np.ndarray:
    wav = load_waveform_16k(path)
    mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        frames = extract_frame_output(model, mel, device)
    frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
    frame_np = min_clamp(frame_np)
    return chunk_frame_embeddings(frame_np, segment_len, stride)


def append_segments(
    segment_blocks: list[np.ndarray],
    offsets: list[dict],
    file_index: int,
    path: str,
    label: int | None,
    domain: str | None,
    segments: np.ndarray,
):
    start = sum(block.shape[0] for block in segment_blocks)
    segment_blocks.append(segments)
    offsets.append({
        "file_index": file_index,
        "path": path,
        "label": "" if label is None else int(label),
        "domain": "" if domain is None else domain,
        "segment_start": start,
        "segment_end": start + int(segments.shape[0]),
        "segment_count": int(segments.shape[0]),
    })


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def segment_knn_scores(train_segments: np.ndarray, query_segments: np.ndarray, k: int, batch_size: int) -> np.ndarray:
    train = l2_normalize(train_segments.astype(np.float64))
    query = l2_normalize(query_segments.astype(np.float64))
    k = min(k, train.shape[0])
    scores = []
    train_norm2 = np.sum(train * train, axis=1)[None, :]
    for start in range(0, query.shape[0], batch_size):
        q = query[start:start + batch_size]
        d2 = np.sum(q * q, axis=1, keepdims=True) + train_norm2 - 2.0 * (q @ train.T)
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
        scores.append(np.sqrt(nearest).mean(axis=1).astype(np.float64))
    return np.concatenate(scores, axis=0)


def aggregate_top_percent(segment_scores: np.ndarray, offsets: list[dict], top_percent: float) -> np.ndarray:
    file_scores = []
    for row in offsets:
        start = int(row["segment_start"])
        end = int(row["segment_end"])
        scores = segment_scores[start:end]
        if scores.size == 0:
            file_scores.append(0.0)
            continue
        top_n = max(1, int(np.ceil(scores.size * top_percent / 100.0)))
        top = np.partition(scores, kth=max(0, scores.size - top_n))[-top_n:]
        file_scores.append(float(np.mean(top)))
    return np.asarray(file_scores, dtype=np.float64)


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


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine) if items else []
        train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(train_items),
            "test_count": len(test_items),
            "segment_embedding_exists": embedding_path(out_root, machine).exists(),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "clip_reference_exists": Path(args.clip_reference_per_machine).exists(),
        "machines": machines,
        "segment_len": args.segment_len,
        "segment_stride": args.segment_stride,
        "score_k": args.score_k,
        "top_percent": args.top_percent,
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no scoring.",
            "This stage tests clean EAT segment evidence without training, LDN, selectors, or fusion.",
        ],
    })


def run_extract_segment_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    eat_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    if not eat_path.exists():
        raise FileNotFoundError(f"Missing EAT checkpoint path: {eat_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machine in {machines}; known={MACHINES_ALL}")

    model, device = load_clean_eat_local(eat_path, args.device)
    rows = []
    for machine in machines:
        items = filter_items(scan_dataset(data_root), machine)
        train_items, test_items = split_train_test(items, use_target_train=False)
        if not train_items or not test_items:
            raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")

        train_blocks: list[np.ndarray] = []
        test_blocks: list[np.ndarray] = []
        train_offsets: list[dict] = []
        test_offsets: list[dict] = []
        y_test = []
        d_test = []
        test_paths = []

        for idx, item in enumerate(train_items):
            segments = extract_file_segments(model, Path(item.path), device, args.segment_len, args.segment_stride)
            append_segments(train_blocks, train_offsets, idx, str(item.path), None, None, segments)

        for idx, item in enumerate(test_items):
            segments = extract_file_segments(model, Path(item.path), device, args.segment_len, args.segment_stride)
            append_segments(test_blocks, test_offsets, idx, str(item.path), int(item.label), item.domain, segments)
            y_test.append(int(item.label))
            d_test.append(item.domain)
            test_paths.append(str(item.path))

        train_segments = np.concatenate(train_blocks, axis=0).astype(np.float32)
        test_segments = np.concatenate(test_blocks, axis=0).astype(np.float32)
        path = embedding_path(out_root, machine)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_segments=train_segments,
            test_segments=test_segments,
            train_offsets=np.asarray(train_offsets, dtype=object),
            test_offsets=np.asarray(test_offsets, dtype=object),
            y_test=np.asarray(y_test, dtype=np.int64),
            d_test=np.asarray(d_test, dtype="U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            machine=machine,
            segment_len=int(args.segment_len),
            segment_stride=int(args.segment_stride),
            preprocess="min_clamp",
            segment_pooling="mean",
            embedding_source="clean_eat_frame_chunks",
        )
        rows.append({
            "machine": machine,
            "train_file_count": len(train_items),
            "test_file_count": len(test_items),
            "train_segment_count": int(train_segments.shape[0]),
            "test_segment_count": int(test_segments.shape[0]),
            "embedding_dim": int(train_segments.shape[1]),
            "avg_train_segments_per_file": float(train_segments.shape[0] / len(train_items)),
            "avg_test_segments_per_file": float(test_segments.shape[0] / len(test_items)),
            "embedding_path": str(path),
        })
    write_csv(out_root / "embedding_summary.csv", rows)


def run_score_segment_evidence(args):
    out_root = Path(args.output_root)
    machines = parse_csv_list(args.machines)
    missing = [str(embedding_path(out_root, m)) for m in machines if not embedding_path(out_root, m).exists()]
    if missing:
        raise FileNotFoundError(f"Missing segment embeddings: {missing}")

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    score_root = out_root / "scores" / CANDIDATE
    score_root.mkdir(parents=True, exist_ok=True)
    clip_refs = get_clip_reference(Path(args.clip_reference_per_machine))
    comparison_rows = []

    for machine in machines:
        arr = np.load(embedding_path(out_root, machine), allow_pickle=True)
        train_segments = arr["train_segments"].astype(np.float64)
        test_segments = arr["test_segments"].astype(np.float64)
        test_offsets = list(arr["test_offsets"])
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype(str)
        test_paths = arr["test_paths"].astype(str)

        segment_scores = segment_knn_scores(train_segments, test_segments, args.score_k, args.query_batch_size)
        file_scores = aggregate_top_percent(segment_scores, test_offsets, args.top_percent)
        summary, tail, geom = evaluate(file_scores, y_test, d_test)
        per_machine_rows.append({"candidate": CANDIDATE, "machine": machine, **summary})
        tail_rows.append({"candidate": CANDIDATE, "machine": machine, **tail})
        geom_rows.append({"candidate": CANDIDATE, "machine": machine, **geom})

        np.savez(
            score_root / f"{machine}_scores.npz",
            test_scores=file_scores.astype(np.float64),
            segment_scores=segment_scores.astype(np.float64),
            y_test=y_test,
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            candidate=CANDIDATE,
            machine=machine,
            top_percent=float(args.top_percent),
            score_k=int(args.score_k),
        )

        ref = clip_refs.get(machine)
        comparison_rows.append({
            "machine": machine,
            "segment_hmean": summary["hmean"],
            "segment_pAUC": summary["pAUC"],
            "segment_TPR99": summary["TPR@99n"],
            "clip_ldn_hmean": float(ref["hmean"]) if ref else None,
            "clip_ldn_pAUC": float(ref["pAUC"]) if ref else None,
            "clip_ldn_TPR99": float(ref["TPR@99n"]) if ref else None,
            "delta_hmean_vs_clean_clip_ldn": summary["hmean"] - float(ref["hmean"]) if ref else None,
            "delta_pAUC_vs_clean_clip_ldn": summary["pAUC"] - float(ref["pAUC"]) if ref else None,
            "delta_TPR99_vs_clean_clip_ldn": summary["TPR@99n"] - float(ref["TPR@99n"]) if ref else None,
        })

    mean_h = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_p = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_t95 = float(np.mean([r["TPR@95n"] for r in per_machine_rows]))
    mean_t99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))
    mean_ks = float(np.mean([r["KS"] for r in per_machine_rows]))
    summary_rows = [{
        "candidate": CANDIDATE,
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_KS": mean_ks,
        "mean_TPR@95n": mean_t95,
        "mean_TPR@99n": mean_t99,
    }]
    improved_vs_clean_clip = sum(
        1 for row in comparison_rows
        if row["delta_hmean_vs_clean_clip_ldn"] is not None and row["delta_hmean_vs_clean_clip_ldn"] > 0
    )
    bottleneck_improved_t99 = sum(
        1 for row in comparison_rows
        if row["machine"] in {"fan", "gearbox", "slider", "toycar", "toytrain"}
        and row["delta_TPR99_vs_clean_clip_ldn"] is not None
        and row["delta_TPR99_vs_clean_clip_ldn"] > 0
    )
    strong_collapse = any(
        row["machine"] in {"bearing", "toytrain", "valve"}
        and row["delta_hmean_vs_clean_clip_ldn"] is not None
        and row["delta_hmean_vs_clean_clip_ldn"] < -0.02
        for row in comparison_rows
    )
    verdict = {
        "candidate": CANDIDATE,
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_TPR99": mean_t99,
        "delta_hmean_vs_clean_ldn_best": mean_h - CONFIRMED_CLEAN_LDN_HMEAN,
        "delta_hmean_vs_stage22_hmean_best": mean_h - CONFIRMED_STAGE22_HMEAN,
        "improved_machine_count_vs_clean_clip_ldn": improved_vs_clean_clip,
        "bottleneck_TPR99_improved_count_vs_clean_clip_ldn": bottleneck_improved_t99,
        "strong_machine_severe_collapse": strong_collapse,
        "segment_evidence_positive": bool(
            mean_h > CONFIRMED_CLEAN_LDN_HMEAN
            or bottleneck_improved_t99 >= 2
            or improved_vs_clean_clip >= 4
        ),
        "promotable": bool(mean_h > CONFIRMED_STAGE22_HMEAN and improved_vs_clean_clip >= 4 and not strong_collapse),
        "notes": [
            "No training.",
            "No LDN.",
            "No selector or fusion.",
            "Fixed segment_len=16, stride=8, score_k=5, top_percent=5 unless explicitly changed.",
        ],
    }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "comparison_vs_clean_clip_ldn.csv", comparison_rows)
    write_json(out_root / "stage24a_segment_evidence_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage24A EAT Segment Evidence Audit",
        "",
        f"- candidate: `{CANDIDATE}`",
        f"- mean_hmean: `{mean_h:.12f}`",
        f"- mean_pAUC: `{mean_p:.12f}`",
        f"- mean_TPR@99n: `{mean_t99:.12f}`",
        f"- delta hmean vs clean LDN best: `{verdict['delta_hmean_vs_clean_ldn_best']:.12f}`",
        f"- delta hmean vs Stage22 hmean best: `{verdict['delta_hmean_vs_stage22_hmean_best']:.12f}`",
        f"- improved machines vs clean clip LDN: `{improved_vs_clean_clip}`",
        f"- bottleneck TPR@99n improved count: `{bottleneck_improved_t99}`",
        f"- strong-machine severe collapse: `{strong_collapse}`",
        f"- segment evidence positive: `{verdict['segment_evidence_positive']}`",
        f"- promotable: `{verdict['promotable']}`",
        "",
        "This stage tests scoring architecture only: clean EAT segment chunks with fixed top-5% segment kNN aggregation. No training, LDN, selector, fusion, or parameter search is introduced.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_segment_embeddings":
        run_extract_segment_embeddings(args)
        return
    if args.mode == "score_segment_evidence":
        run_score_segment_evidence(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
