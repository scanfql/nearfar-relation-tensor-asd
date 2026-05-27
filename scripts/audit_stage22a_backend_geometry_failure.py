#!/usr/bin/env python
"""Stage22A: backend-geometry failure analysis for Stage21D adapted EAT.

This audit compares clean EAT and Stage21D-adapted embeddings on normal train
audio for one machine. It does not use anomaly labels, tune LDN parameters, or
select checkpoints by test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21e_adapted_eval import (
    baseline_knn_scores,
    chord_distance_matrix,
    extract_frame_output,
    load_clean_eat_local,
    load_stage21d_adapted,
    load_waveform_16k,
    local_density_knn,
    mean_pool,
    min_clamp,
    waveform_to_eat_tensor,
)


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ADAPTED_CKPT_DEFAULT = "results/stage21d_global_dap_ssl_sanity/checkpoints/stage21d_global_best.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE21D_ROOT_DEFAULT = "results/stage21d_global_dap_ssl_sanity"
OUTPUT_ROOT_DEFAULT = "results/stage22a_backend_geometry_failure"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE22A_BACKEND_GEOMETRY_FAILURE_AUDIT.md"
MACHINE_DEFAULT = "fan"
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Stage22A backend-geometry failure analysis")
    p.add_argument("--mode", required=True, choices=["check", "extract_train_embeddings", "analyze_train_geometry"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--adapted_ckpt_path", default=ADAPTED_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--stage21d_root", default=STAGE21D_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT, choices=["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--max_train_items", type=int, default=0, help="Optional smoke-test cap. 0 means all train items.")
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--ldn_K", type=int, default=1)
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
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def machine_out_root(base_root: Path, machine: str) -> Path:
    return base_root / machine


def expected_train_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, _ = split_train_test(items, use_target_train=False)
    return train_items


def extract_minclamp_mean_embedding(model, path: Path, device) -> np.ndarray:
    import torch

    wav = load_waveform_16k(path)
    mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        frames = extract_frame_output(model, mel, device)
    frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return mean_pool(min_clamp(frame_np)).astype(np.float64)


def quantile_summary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_q05": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q95": 0.0,
            f"{prefix}_q99": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_q99": float(np.quantile(values, 0.99)),
    }


def topk_indices(dist: np.ndarray, k: int) -> np.ndarray:
    k = min(k, max(1, dist.shape[1] - 1))
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    row = np.arange(d.shape[0])[:, None]
    order = np.argsort(d[row, idx], axis=1)
    return idx[row, order]


def neighbor_retention(clean_dist: np.ndarray, adapted_dist: np.ndarray, ks: list[int]) -> list[dict]:
    rows = []
    n = clean_dist.shape[0]
    for k in ks:
        if n <= 1:
            rows.append({"k": k, "mean_retention": 0.0, "median_retention": 0.0})
            continue
        kk = min(k, n - 1)
        clean_nn = topk_indices(clean_dist, kk)
        adapted_nn = topk_indices(adapted_dist, kk)
        overlaps = []
        for i in range(n):
            overlaps.append(len(set(clean_nn[i]).intersection(set(adapted_nn[i]))) / float(kk))
        rows.append({
            "k": kk,
            "mean_retention": float(np.mean(overlaps)),
            "median_retention": float(np.median(overlaps)),
            "q10_retention": float(np.quantile(overlaps, 0.10)),
            "q90_retention": float(np.quantile(overlaps, 0.90)),
        })
    return rows


def leave_one_out_knn_scores(dist: np.ndarray, score_k: int) -> np.ndarray:
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    score_k = min(score_k, max(1, d.shape[1] - 1))
    nearest = np.partition(d, kth=score_k - 1, axis=1)[:, :score_k]
    return nearest.mean(axis=1).astype(np.float64)


def leave_one_out_ldn_scores(dist: np.ndarray, rho: np.ndarray, score_k: int) -> np.ndarray:
    adjusted = dist - rho[None, :]
    np.fill_diagonal(adjusted, np.inf)
    score_k = min(score_k, max(1, adjusted.shape[1] - 1))
    nearest = np.partition(adjusted, kth=score_k - 1, axis=1)[:, :score_k]
    return nearest.mean(axis=1).astype(np.float64)


def embedding_dispersion(emb: np.ndarray, dist: np.ndarray, prefix: str) -> dict:
    centered = emb - emb.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, emb.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    eig_sum = float(eig.sum())
    if eig_sum > EPS:
        p = eig / eig_sum
        p = p[p > EPS]
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))
        top1_ratio = float(eig[-1] / eig_sum)
    else:
        effective_rank = 0.0
        top1_ratio = 0.0

    tri = dist[np.triu_indices(dist.shape[0], k=1)] if dist.shape[0] > 1 else np.asarray([], dtype=np.float64)
    nn = leave_one_out_knn_scores(dist, score_k=1) if dist.shape[0] > 1 else np.asarray([], dtype=np.float64)
    return {
        f"{prefix}_variance_trace": eig_sum,
        f"{prefix}_effective_rank": effective_rank,
        f"{prefix}_top1_eigen_ratio": top1_ratio,
        **quantile_summary(tri, f"{prefix}_pairwise_chord"),
        **quantile_summary(nn, f"{prefix}_nearest_chord"),
    }


def paired_drift(clean_emb: np.ndarray, adapted_emb: np.ndarray) -> dict:
    clean_norm = clean_emb / np.maximum(np.linalg.norm(clean_emb, axis=1, keepdims=True), EPS)
    adapted_norm = adapted_emb / np.maximum(np.linalg.norm(adapted_emb, axis=1, keepdims=True), EPS)
    cos = np.sum(clean_norm * adapted_norm, axis=1)
    chord = np.sqrt(np.maximum(2.0 - 2.0 * cos, 0.0))
    return {
        **quantile_summary(cos, "paired_cosine_clean_adapted"),
        **quantile_summary(chord, "paired_chord_clean_adapted"),
    }


def rho_summary(dist: np.ndarray, K: int, prefix: str) -> tuple[np.ndarray, dict]:
    rho = local_density_knn(dist, K=K)
    return rho, quantile_summary(rho, f"{prefix}_rho_K{K}")


def ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > EPS else 0.0


def run_check(args):
    out_root = machine_out_root(Path(args.output_root), args.machine)
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    train_items = expected_train_items(data_root, args.machine) if data_root.exists() else []
    status = {
        "mode": "check",
        "machine": args.machine,
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "adapted_ckpt_exists": Path(args.adapted_ckpt_path).exists(),
        "data_root_exists": data_root.exists(),
        "stage21d_root_exists": Path(args.stage21d_root).exists(),
        "source_train_normal_count": len(train_items),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no scoring.",
            "This audit uses train-normal geometry only and never uses anomaly labels.",
        ],
    }
    write_json(out_root / "check_status.json", status)


def run_extract_train_embeddings(args):
    out_root = machine_out_root(Path(args.output_root), args.machine)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    eat_path = Path(args.eat_model_path)
    adapted_path = Path(args.adapted_ckpt_path)
    data_root = Path(args.data_root)
    if not eat_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {eat_path}")
    if not adapted_path.exists():
        raise FileNotFoundError(f"Missing adapted checkpoint: {adapted_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    train_items = expected_train_items(data_root, args.machine)
    if args.max_train_items > 0:
        train_items = train_items[:args.max_train_items]
    if len(train_items) < 2:
        raise RuntimeError(f"Need at least two train items for geometry audit, got {len(train_items)}")

    clean_model, device = load_clean_eat_local(eat_path, args.device)
    adapted_model, _, adapted_device, blob, _ = load_stage21d_adapted(eat_path, adapted_path, str(device))
    if str(adapted_device) != str(device):
        raise RuntimeError(f"Device mismatch: clean={device} adapted={adapted_device}")

    clean_embs = []
    adapted_embs = []
    rows = []
    for idx, item in enumerate(train_items):
        path = Path(item.path)
        clean_emb = extract_minclamp_mean_embedding(clean_model, path, device)
        adapted_emb = extract_minclamp_mean_embedding(adapted_model, path, device)
        clean_embs.append(clean_emb)
        adapted_embs.append(adapted_emb)
        rows.append({
            "row_index": idx,
            "path": str(path),
            "machine": args.machine,
            "label": int(item.label),
            "domain": item.domain,
        })

    clean_arr = np.stack(clean_embs, axis=0).astype(np.float64)
    adapted_arr = np.stack(adapted_embs, axis=0).astype(np.float64)
    paths = np.asarray([r["path"] for r in rows], dtype=object)
    domains = np.asarray([r["domain"] for r in rows], dtype="U16")
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    emb_path = emb_root / f"{args.machine}_clean_vs_stage21d_train_minclamp_mean_embeddings.npz"
    np.savez_compressed(
        emb_path,
        clean_train_embeddings=clean_arr,
        adapted_train_embeddings=adapted_arr,
        train_paths=paths,
        train_domains=domains,
        train_labels=labels,
        machine=args.machine,
        clean_checkpoint_path=str(eat_path),
        adapted_checkpoint_path=str(adapted_path),
        stage21d_best_epoch=blob.get("best_epoch"),
        stage21d_best_val_loss=blob.get("best_val_loss"),
        preprocess="min_clamp",
        pooling="mean",
    )
    write_csv(out_root / "train_embedding_inventory.csv", rows)
    write_json(out_root / "embedding_summary.json", {
        "machine": args.machine,
        "train_count": int(clean_arr.shape[0]),
        "embedding_dim": int(clean_arr.shape[1]),
        "embedding_path": str(emb_path),
        "stage21d_best_epoch": blob.get("best_epoch"),
        "stage21d_best_val_loss": blob.get("best_val_loss"),
        "max_train_items": int(args.max_train_items),
        "notes": "Clean and adapted embeddings are paired by the same train-normal files.",
    })


def run_analyze_train_geometry(args):
    out_root = machine_out_root(Path(args.output_root), args.machine)
    emb_path = out_root / "embeddings" / f"{args.machine}_clean_vs_stage21d_train_minclamp_mean_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {emb_path}")

    arr = np.load(emb_path, allow_pickle=True)
    clean_emb = arr["clean_train_embeddings"].astype(np.float64)
    adapted_emb = arr["adapted_train_embeddings"].astype(np.float64)
    paths = arr["train_paths"].astype(str)
    labels = arr["train_labels"].astype(np.int64)
    if not np.all(labels == 0):
        raise RuntimeError("Stage22A expected train-normal only embeddings, found non-zero labels.")

    clean_dist = chord_distance_matrix(clean_emb, clean_emb)
    adapted_dist = chord_distance_matrix(adapted_emb, adapted_emb)
    clean_rho, clean_rho_stats = rho_summary(clean_dist, args.ldn_K, "clean")
    adapted_rho, adapted_rho_stats = rho_summary(adapted_dist, args.ldn_K, "adapted")

    clean_base = leave_one_out_knn_scores(clean_dist, args.latent_knn_k)
    adapted_base = leave_one_out_knn_scores(adapted_dist, args.latent_knn_k)
    clean_ldn = leave_one_out_ldn_scores(clean_dist, clean_rho, args.latent_knn_k)
    adapted_ldn = leave_one_out_ldn_scores(adapted_dist, adapted_rho, args.latent_knn_k)

    retention_rows = neighbor_retention(clean_dist, adapted_dist, ks=[1, 5, 10])
    write_csv(out_root / "neighbor_retention.csv", retention_rows)

    clean_disp = embedding_dispersion(clean_emb, clean_dist, "clean")
    adapted_disp = embedding_dispersion(adapted_emb, adapted_dist, "adapted")
    score_stats = {
        **quantile_summary(clean_base, "clean_train_loo_baseline_k5"),
        **quantile_summary(adapted_base, "adapted_train_loo_baseline_k5"),
        **quantile_summary(clean_ldn, "clean_train_loo_ldn_diff_K1_scorek5"),
        **quantile_summary(adapted_ldn, "adapted_train_loo_ldn_diff_K1_scorek5"),
    }
    drift_stats = paired_drift(clean_emb, adapted_emb)

    clean_eff = clean_disp["clean_effective_rank"]
    adapted_eff = adapted_disp["adapted_effective_rank"]
    clean_pair = clean_disp["clean_pairwise_chord_mean"]
    adapted_pair = adapted_disp["adapted_pairwise_chord_mean"]
    clean_rho_mean = clean_rho_stats[f"clean_rho_K{args.ldn_K}_mean"]
    adapted_rho_mean = adapted_rho_stats[f"adapted_rho_K{args.ldn_K}_mean"]
    retention_at5 = next(r for r in retention_rows if r["k"] == min(5, clean_emb.shape[0] - 1))["mean_retention"]

    summary = {
        "mode": "analyze_train_geometry",
        "machine": args.machine,
        "train_count": int(clean_emb.shape[0]),
        "embedding_dim": int(clean_emb.shape[1]),
        "latent_knn_k": int(args.latent_knn_k),
        "ldn_K": int(args.ldn_K),
        "neighbor_retention_at5": float(retention_at5),
        "effective_rank_ratio_adapted_over_clean": ratio(adapted_eff, clean_eff),
        "mean_pairwise_chord_ratio_adapted_over_clean": ratio(adapted_pair, clean_pair),
        "rho_mean_ratio_adapted_over_clean": ratio(adapted_rho_mean, clean_rho_mean),
        "ldn_proxy_std_ratio_adapted_over_clean": ratio(
            score_stats["adapted_train_loo_ldn_diff_K1_scorek5_std"],
            score_stats["clean_train_loo_ldn_diff_K1_scorek5_std"],
        ),
        **clean_rho_stats,
        **adapted_rho_stats,
        **clean_disp,
        **adapted_disp,
        **score_stats,
        **drift_stats,
    }

    finding_flags = {
        "low_neighbor_retention_at5": bool(summary["neighbor_retention_at5"] < 0.50),
        "possible_dispersion_collapse": bool(summary["effective_rank_ratio_adapted_over_clean"] < 0.80 or summary["mean_pairwise_chord_ratio_adapted_over_clean"] < 0.80),
        "large_rho_shift": bool(summary["rho_mean_ratio_adapted_over_clean"] < 0.80 or summary["rho_mean_ratio_adapted_over_clean"] > 1.25),
        "ldn_proxy_tail_widened": bool(summary["ldn_proxy_std_ratio_adapted_over_clean"] > 1.20),
    }
    recommendations = []
    if finding_flags["low_neighbor_retention_at5"]:
        recommendations.append("Prioritize Stage22 neighborhood-preservation loss against clean EAT top-k neighbors.")
    if finding_flags["possible_dispersion_collapse"]:
        recommendations.append("Add dispersion/effective-rank guard before any local-density proxy loss.")
    if finding_flags["large_rho_shift"] or finding_flags["ldn_proxy_tail_widened"]:
        recommendations.append("Use an LDN-proxy consistency term so train-normal local-density scores remain stable.")
    if not recommendations:
        recommendations.append("Stage21D did not show obvious train-normal geometry damage; inspect fan test score geometry before training a new objective.")

    verdict = {
        "machine": args.machine,
        "stage22a_failure_analysis_complete": True,
        "uses_anomaly_labels": False,
        "tuned_ldn_parameters": False,
        "checkpoint_selected_by_test_labels": False,
        "finding_flags": finding_flags,
        "recommended_stage22_objective_components": recommendations,
    }

    write_json(out_root / "geometry_failure_summary.json", summary)
    write_json(out_root / "stage22a_geometry_failure_verdict.json", verdict)

    per_file_rows = []
    rho_delta = adapted_rho - clean_rho
    ldn_delta = adapted_ldn - clean_ldn
    paired_chord = np.sqrt(np.maximum(2.0 - 2.0 * np.sum(
        clean_emb / np.maximum(np.linalg.norm(clean_emb, axis=1, keepdims=True), EPS)
        * adapted_emb / np.maximum(np.linalg.norm(adapted_emb, axis=1, keepdims=True), EPS),
        axis=1,
    ), 0.0))
    for idx, path in enumerate(paths):
        per_file_rows.append({
            "row_index": idx,
            "path": path,
            "clean_rho_K1": float(clean_rho[idx]),
            "adapted_rho_K1": float(adapted_rho[idx]),
            "delta_rho_adapted_minus_clean": float(rho_delta[idx]),
            "clean_train_loo_ldn_diff_K1_scorek5": float(clean_ldn[idx]),
            "adapted_train_loo_ldn_diff_K1_scorek5": float(adapted_ldn[idx]),
            "delta_ldn_proxy_adapted_minus_clean": float(ldn_delta[idx]),
            "paired_chord_clean_adapted": float(paired_chord[idx]),
        })
    write_csv(out_root / "per_file_geometry_drift.csv", per_file_rows)

    write_markdown(Path(args.doc_path), [
        "# CTAF Stage22A Backend Geometry Failure Audit",
        "",
        f"- machine: `{args.machine}`",
        f"- train-normal count: `{clean_emb.shape[0]}`",
        f"- neighbor retention @5: `{summary['neighbor_retention_at5']:.6f}`",
        f"- effective-rank ratio adapted/clean: `{summary['effective_rank_ratio_adapted_over_clean']:.6f}`",
        f"- mean pairwise chord ratio adapted/clean: `{summary['mean_pairwise_chord_ratio_adapted_over_clean']:.6f}`",
        f"- rho mean ratio adapted/clean: `{summary['rho_mean_ratio_adapted_over_clean']:.6f}`",
        f"- LDN proxy std ratio adapted/clean: `{summary['ldn_proxy_std_ratio_adapted_over_clean']:.6f}`",
        "",
        "## Verdict",
        "",
        f"- low neighbor retention @5: `{finding_flags['low_neighbor_retention_at5']}`",
        f"- possible dispersion collapse: `{finding_flags['possible_dispersion_collapse']}`",
        f"- large rho shift: `{finding_flags['large_rho_shift']}`",
        f"- LDN proxy tail widened: `{finding_flags['ldn_proxy_tail_widened']}`",
        "",
        "## Stage22 Objective Implication",
        "",
        *[f"- {line}" for line in recommendations],
        "",
        "No anomaly labels, selector, checkpoint selection by test labels, or LDN parameter tuning were used.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_train_embeddings":
        run_extract_train_embeddings(args)
        return
    if args.mode == "analyze_train_geometry":
        run_analyze_train_geometry(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
