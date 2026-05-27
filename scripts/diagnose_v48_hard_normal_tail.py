#!/usr/bin/env python
"""V48 no-training diagnostic for hard-normal / normal-tail low-FPR bottlenecks."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import extract_logmel, load_audio
from src.features.tensor_builder import TensorNormalizer, build_tensor
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder


DEFAULT_SEEDS = "13711,13721,13731"
DEFAULT_MACHINES = "gearbox,slider,toycar,toytrain,fan"
PROXY_COMPRESS_RATIO = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose hard-normal low-FPR bottlenecks")
    p.add_argument("--data_root", required=True)
    p.add_argument("--score_seed_dirs", required=True,
                   help="Comma-separated seed result dirs or one root containing seed_<SEED> dirs")
    p.add_argument("--checkpoint_dirs", required=True,
                   help="Comma-separated checkpoint model dirs or one root containing seed_<SEED>/models dirs")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--machines", default=DEFAULT_MACHINES)
    p.add_argument("--experiment", default="full_5view")
    p.add_argument("--score_mode", default="latent_summary_knn")
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--hard_normal_top_ratio", type=float, default=0.10)
    p.add_argument("--easy_normal_bottom_ratio", type=float, default=0.50)
    p.add_argument("--normal_tail_q_low", type=float, default=0.95)
    p.add_argument("--normal_tail_q_high", type=float, default=0.99)
    p.add_argument("--max_samples", type=int, default=20)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def parse_seed_list(value: str):
    return [s.strip() for s in value.split(",") if s.strip()]


def resolve_seed_dirs(spec: str, seeds: list[str], expect_models: bool = False):
    parts = [Path(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) == 1:
        if len(seeds) == 1:
            return parts
        root = parts[0]
        resolved = []
        for seed in seeds:
            cand = root / f"seed_{seed}"
            if expect_models:
                cand = cand / "models"
            resolved.append(cand)
        return resolved
    if len(parts) != len(seeds):
        raise ValueError("seed dir specification must be one root or one path per seed")
    return parts


def load_score_npz(seed_dir: Path, machine: str, experiment: str, score_mode: str):
    return seed_dir / "scores" / f"{machine}_{experiment}_{score_mode}_scores.npz"


def safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else 0.0


def latent_summary_from_tensor(model, tensor: np.ndarray, device: str):
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    model = model.to(requested)
    model.eval()
    x = torch.from_numpy(tensor[None].astype(np.float32)).to(requested)
    with torch.no_grad():
        latent = model.encode(x)
        mean = latent.mean(dim=(2, 3))
        std = latent.std(dim=(2, 3), unbiased=False)
        max_vals = latent.amax(dim=(2, 3))
        summary = torch.cat([mean, std, max_vals], dim=1)
    return summary.squeeze(0).cpu().numpy().astype(np.float64)


def augmentation_time_mask(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    _, _, t = out.shape
    w = max(1, int(np.ceil(t * 0.10)))
    t0 = max(0, (t - w) // 2)
    out[:, :, t0:t0 + w] = 0.0
    return out


def augmentation_freq_mask(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    _, f, _ = out.shape
    w = max(1, int(np.ceil(f * 0.10)))
    f0 = max(0, (f - w) // 2)
    out[:, f0:f0 + w, :] = 0.0
    return out


def augmentation_light_noise(x: np.ndarray, seed: int = 13711) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (x + rng.normal(0.0, 0.05, size=x.shape)).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(1.0 - np.dot(an, bn))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def build_train_tensors(train_items):
    tensors = []
    for item in train_items:
        audio = load_audio(item.path)
        logmel = extract_logmel(audio)
        tensors.append(build_tensor("full_5view", logmel, audio=audio))
    return tensors


def load_model_from_checkpoint(checkpoint_path: Path):
    model = ViewwiseUnsharedLateFusionAutoencoder(
        in_channels=5,
        latent_channels=64,
        use_batch_norm=True,
        per_view_channels=16,
        stem_channels=8,
        dropout_p=0.0,
    )
    blob = torch.load(checkpoint_path, map_location="cpu")
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state)
    return model


def mean_sq_radius(features: np.ndarray):
    center = features.mean(axis=0, keepdims=True)
    d2 = np.sum((features - center) ** 2, axis=1)
    return float(np.mean(d2))


def nearest_group_flags(train_features, hard_idx, easy_idx, query_features):
    train_norm = train_features / np.maximum(np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12)
    q_norm = query_features / np.maximum(np.linalg.norm(query_features, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(q_norm * q_norm, axis=1, keepdims=True)
        + np.sum(train_norm * train_norm, axis=1)[None, :]
        - 2.0 * (q_norm @ train_norm.T)
    )
    d2 = np.maximum(d2, 0.0)
    nn_idx = np.argmin(d2, axis=1)
    hard_set = set(map(int, hard_idx))
    return np.array([idx in hard_set for idx in nn_idx], dtype=bool), nn_idx


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    machines = parse_seed_list(args.machines)
    score_seed_dirs = resolve_seed_dirs(args.score_seed_dirs, seeds, expect_models=False)
    checkpoint_dirs = resolve_seed_dirs(args.checkpoint_dirs, seeds, expect_models=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    items = scan_dataset(Path(args.data_root))

    hard_rows = []
    tail_rows = []
    overlap_rows = []
    augment_rows = []
    proxy_rows = []
    missing_rows = []

    for seed, seed_dir, ckpt_dir in zip(seeds, score_seed_dirs, checkpoint_dirs):
        for machine in machines:
            npz_path = load_score_npz(seed_dir, machine, args.experiment, args.score_mode)
            if not npz_path.exists():
                missing_rows.append({"seed": seed, "machine": machine, "kind": "score_file", "path": str(npz_path)})
                continue

            arr = np.load(npz_path, allow_pickle=True)
            train_scores = arr["train_scores"].astype(np.float64)
            test_scores = arr["test_scores"].astype(np.float64)
            train_features = arr["train_latent_summaries"].astype(np.float64)
            test_features = arr["test_latent_summaries"].astype(np.float64)
            y_test = arr["y_test"].astype(np.int64)

            n_train = len(train_scores)
            hard_n = max(1, int(np.ceil(n_train * args.hard_normal_top_ratio)))
            easy_n = max(1, int(np.floor(n_train * args.easy_normal_bottom_ratio)))
            order = np.argsort(train_scores)
            hard_idx = order[-hard_n:]
            easy_idx = order[:easy_n]
            hard_feat = train_features[hard_idx]
            easy_feat = train_features[easy_idx]

            all_center = train_features.mean(axis=0, keepdims=True)
            hard_rows.append({
                "seed": seed,
                "machine": machine,
                "train_count": n_train,
                "hard_ratio": args.hard_normal_top_ratio,
                "hard_count": int(hard_n),
                "easy_definition": f"lowest_{int(args.easy_normal_bottom_ratio*100)}pct",
                "easy_count": int(easy_n),
                "hard_centroid_dist": float(np.mean(np.linalg.norm(hard_feat - all_center, axis=1))),
                "easy_centroid_dist": float(np.mean(np.linalg.norm(easy_feat - all_center, axis=1))),
                "hard_knn_score_mean": float(np.mean(train_scores[hard_idx])),
                "easy_knn_score_mean": float(np.mean(train_scores[easy_idx])),
                "hard_intra_var": mean_sq_radius(hard_feat),
                "easy_intra_var": mean_sq_radius(easy_feat),
                "domain_comp": "source_only",
            })

            normal_idx = np.where(y_test == 0)[0]
            anomaly_idx = np.where(y_test == 1)[0]
            normal_scores = test_scores[normal_idx]
            q95 = float(np.quantile(normal_scores, args.normal_tail_q_low))
            q99 = float(np.quantile(normal_scores, args.normal_tail_q_high))
            tail_idx = normal_idx[(test_scores[normal_idx] > q95) & (test_scores[normal_idx] <= q99)]
            near_q99_cut = q95 + 0.75 * (q99 - q95)
            near_q99_idx = tail_idx[test_scores[tail_idx] >= near_q99_cut]

            hard_center = hard_feat.mean(axis=0, keepdims=True)
            easy_center = easy_feat.mean(axis=0, keepdims=True)
            if len(tail_idx):
                tail_dist_h = np.linalg.norm(test_features[tail_idx] - hard_center, axis=1)
                tail_dist_e = np.linalg.norm(test_features[tail_idx] - easy_center, axis=1)
                hard_like = tail_dist_h < tail_dist_e
            else:
                hard_like = np.array([], dtype=bool)
            if len(near_q99_idx):
                near_h = np.linalg.norm(test_features[near_q99_idx] - hard_center, axis=1)
                near_e = np.linalg.norm(test_features[near_q99_idx] - easy_center, axis=1)
                hard_like_near = near_h < near_e
            else:
                hard_like_near = np.array([], dtype=bool)
            tail_rows.append({
                "seed": seed,
                "machine": machine,
                "q95": q95,
                "q99": q99,
                "normal_tail_count": int(len(tail_idx)),
                "hard_like_ratio_in_tail": float(hard_like.mean()) if len(hard_like) else float("nan"),
                "hard_like_ratio_near_q99": float(hard_like_near.mean()) if len(hard_like_near) else float("nan"),
            })

            missed_idx = anomaly_idx[test_scores[anomaly_idx] <= q99]
            missed_feat = test_features[missed_idx]
            hard_flags, _ = nearest_group_flags(train_features, hard_idx, easy_idx, missed_feat) if len(missed_feat) else (np.array([], dtype=bool), np.array([], dtype=int))
            dist_h = np.linalg.norm(missed_feat - hard_center, axis=1) if len(missed_feat) else np.array([])
            dist_e = np.linalg.norm(missed_feat - easy_center, axis=1) if len(missed_feat) else np.array([])
            overlap_rows.append({
                "seed": seed,
                "machine": machine,
                "missed_count": int(len(missed_idx)),
                "hard_nn_overlap_ratio": float(hard_flags.mean()) if len(hard_flags) else float("nan"),
                "dist_to_hard_centroid": float(np.mean(dist_h)) if len(dist_h) else float("nan"),
                "dist_to_easy_centroid": float(np.mean(dist_e)) if len(dist_e) else float("nan"),
                "closer_to_hard_ratio": float((dist_h < dist_e).mean()) if len(dist_h) else float("nan"),
            })

            # Theoretical proxy using hard-normal-like normal-tail points.
            if len(tail_idx):
                tail_scores = test_scores[tail_idx].copy()
                tail_scores[hard_like] *= (1.0 - PROXY_COMPRESS_RATIO)
                proxy_scores = test_scores.copy()
                proxy_scores[tail_idx] = tail_scores
                proxy_normals = proxy_scores[normal_idx]
                proxy_q99 = float(np.quantile(proxy_normals, args.normal_tail_q_high))
                proxy_tpr99 = float((proxy_scores[anomaly_idx] > proxy_q99).mean()) if len(anomaly_idx) else float("nan")
            else:
                proxy_q99 = float("nan")
                proxy_tpr99 = float("nan")
            base_tpr99 = float((test_scores[anomaly_idx] > q99).mean()) if len(anomaly_idx) else float("nan")
            proxy_rows.append({
                "seed": seed,
                "machine": machine,
                "base_q99": q99,
                "proxy_q99": proxy_q99,
                "proxy_q99_delta": proxy_q99 - q99 if np.isfinite(proxy_q99) else float("nan"),
                "base_tpr99": base_tpr99,
                "proxy_tpr99": proxy_tpr99,
                "proxy_tpr99_delta": proxy_tpr99 - base_tpr99 if np.isfinite(proxy_tpr99) else float("nan"),
                "proxy_compress_ratio": PROXY_COMPRESS_RATIO,
            })

            checkpoint_path = ckpt_dir / f"{machine}_{args.experiment}_seed{seed}.pt"
            if not checkpoint_path.exists():
                missing_rows.append({"seed": seed, "machine": machine, "kind": "checkpoint", "path": str(checkpoint_path)})
                continue

            # Augmentation consistency
            train_items, _ = split_train_test(filter_items(items, machine), use_target_train=False)
            train_tensors = build_train_tensors(train_items)
            normalizer = TensorNormalizer()
            train_norm = normalizer.fit_transform(train_tensors)

            model = load_model_from_checkpoint(checkpoint_path)
            hard_pick = hard_idx[np.argsort(train_scores[hard_idx])[::-1][:args.max_samples]]
            easy_pick = easy_idx[np.argsort(train_scores[easy_idx])[:args.max_samples]]
            groups = {
                "hard_normal_top10": hard_pick,
                "easy_normal_bottom50": easy_pick,
            }
            for group_name, idxs in groups.items():
                if len(idxs) == 0:
                    continue
                originals = [train_norm[i] for i in idxs]
                orig_feats = [latent_summary_from_tensor(model, x, args.device) for x in originals]
                for aug_name, aug_fn in [
                    ("time_mask", augmentation_time_mask),
                    ("frequency_mask", augmentation_freq_mask),
                    ("light_noise", augmentation_light_noise),
                ]:
                    aug_feats = [latent_summary_from_tensor(model, aug_fn(x), args.device) for x in originals]
                    cos = [cosine_distance(a, b) for a, b in zip(orig_feats, aug_feats)]
                    l2s = [l2_distance(a, b) for a, b in zip(orig_feats, aug_feats)]
                    augment_rows.append({
                        "seed": seed,
                        "machine": machine,
                        "group": group_name,
                        "augmentation": aug_name,
                        "mean_cosine_distance": float(np.mean(cos)),
                        "mean_l2_distance": float(np.mean(l2s)),
                        "n_samples": int(len(idxs)),
                        "checkpoint_path": str(checkpoint_path),
                    })

    # Write CSVs
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "hard_normal_stats.csv": hard_rows,
        "normal_tail_composition.csv": tail_rows,
        "missed_anomaly_overlap.csv": overlap_rows,
        "augmentation_consistency.csv": augment_rows,
        "tail_compression_proxy.csv": proxy_rows,
        "missing_paths.csv": missing_rows,
    }
    for name, rows in tables.items():
        path = output_root / name
        if rows:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # Markdown report
    lines = [
        "# CTAF V48 Hard-Normal Tail Audit",
        "",
        "## 1. Stage 5 Summary",
        "",
        "- mechanism 3-view frontends cannot replace `full_5view`",
        "- current mainline remains `full_5view + latent_summary_knn`",
        "",
        "## 2. Hard-Normal Definition",
        "",
        f"- hard-normal main ratio: `{args.hard_normal_top_ratio:.2f}` of train-normal by highest `{args.score_mode}` score",
        f"- easy-normal main definition: `lowest {int(args.easy_normal_bottom_ratio * 100)}%` of train-normal by `{args.score_mode}` score",
        "- anomaly labels are not used in hard-normal definition",
        "",
    ]

    by_machine_hard = defaultdict(list)
    for row in hard_rows:
        by_machine_hard[row["machine"]].append(row)
    lines.append("| machine | mean train count | mean hard-normal count | mean easy-normal count |")
    lines.append("|---|---:|---:|---:|")
    for machine in machines:
        rows = by_machine_hard.get(machine, [])
        if not rows:
            continue
        lines.append(
            f"| {machine} | {np.mean([r['train_count'] for r in rows]):.1f} | "
            f"{np.mean([r['hard_count'] for r in rows]):.1f} | {np.mean([r['easy_count'] for r in rows]):.1f} |"
        )

    lines.extend([
        "",
        "## 3. Normal-Tail Composition Table",
        "",
        "| machine | mean q95 | mean q99 | mean normal-tail count | mean hard-normal-like ratio in q95-q99 | mean hard-normal-like ratio near q99 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_machine_tail = defaultdict(list)
    for row in tail_rows:
        by_machine_tail[row["machine"]].append(row)
    for machine in machines:
        rows = by_machine_tail.get(machine, [])
        if not rows:
            continue
        lines.append(
            f"| {machine} | {safe_mean([r['q95'] for r in rows]):.6f} | {safe_mean([r['q99'] for r in rows]):.6f} | "
            f"{safe_mean([r['normal_tail_count'] for r in rows]):.1f} | "
            f"{safe_mean([r['hard_like_ratio_in_tail'] for r in rows]):.4f} | "
            f"{safe_mean([r['hard_like_ratio_near_q99'] for r in rows]):.4f} |"
        )

    lines.extend([
        "",
        "## 4. Hard-Normal vs Easy-Normal Latent Statistics",
        "",
        "| machine | hard centroid dist | easy centroid dist | hard kNN score | easy kNN score | hard intra-var | easy intra-var | domain composition |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for machine in machines:
        rows = by_machine_hard.get(machine, [])
        if not rows:
            continue
        lines.append(
            f"| {machine} | {safe_mean([r['hard_centroid_dist'] for r in rows]):.6f} | "
            f"{safe_mean([r['easy_centroid_dist'] for r in rows]):.6f} | "
            f"{safe_mean([r['hard_knn_score_mean'] for r in rows]):.6f} | "
            f"{safe_mean([r['easy_knn_score_mean'] for r in rows]):.6f} | "
            f"{safe_mean([r['hard_intra_var'] for r in rows]):.6f} | "
            f"{safe_mean([r['easy_intra_var'] for r in rows]):.6f} | source_only |"
        )

    lines.extend([
        "",
        "## 5. Missed Anomaly Overlap Table",
        "",
        "| machine | mean missed count | mean hard-NN overlap ratio | mean dist to hard centroid | mean dist to easy centroid | mean closer-to-hard ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_machine_overlap = defaultdict(list)
    for row in overlap_rows:
        by_machine_overlap[row["machine"]].append(row)
    for machine in machines:
        rows = by_machine_overlap.get(machine, [])
        if not rows:
            continue
        lines.append(
            f"| {machine} | {safe_mean([r['missed_count'] for r in rows]):.1f} | "
            f"{safe_mean([r['hard_nn_overlap_ratio'] for r in rows]):.4f} | "
            f"{safe_mean([r['dist_to_hard_centroid'] for r in rows]):.6f} | "
            f"{safe_mean([r['dist_to_easy_centroid'] for r in rows]):.6f} | "
            f"{safe_mean([r['closer_to_hard_ratio'] for r in rows]):.4f} |"
        )

    lines.extend([
        "",
        "## 6. Augmentation Consistency Table",
        "",
    ])
    if augment_rows:
        lines.extend([
            "| machine | group | augmentation | mean cosine distance | mean L2 distance | n_samples |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in augment_rows:
            lines.append(
                f"| {row['machine']} | {row['group']} | {row['augmentation']} | "
                f"{row['mean_cosine_distance']:.6f} | {row['mean_l2_distance']:.6f} | {row['n_samples']} |"
            )
    else:
        lines.append("Checkpoint-based augmentation consistency could not be computed because checkpoint paths were missing.")

    lines.extend([
        "",
        "## 7. Theoretical Tail Compression Proxy",
        "",
        f"Fixed proxy compression ratio on hard-normal-like tail samples: `{PROXY_COMPRESS_RATIO:.2f}`",
        "",
        "| machine | base q99 | proxy q99 | delta q99 | base TPR@99n | proxy TPR@99n | delta TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    by_machine_proxy = defaultdict(list)
    for row in proxy_rows:
        by_machine_proxy[row["machine"]].append(row)
    for machine in machines:
        rows = by_machine_proxy.get(machine, [])
        if not rows:
            continue
        lines.append(
            f"| {machine} | {safe_mean([r['base_q99'] for r in rows]):.6f} | "
            f"{safe_mean([r['proxy_q99'] for r in rows]):.6f} | "
            f"{safe_mean([r['proxy_q99_delta'] for r in rows]):+.6f} | "
            f"{safe_mean([r['base_tpr99'] for r in rows]):.4f} | "
            f"{safe_mean([r['proxy_tpr99'] for r in rows]):.4f} | "
            f"{safe_mean([r['proxy_tpr99_delta'] for r in rows]):+.4f} |"
        )

    # Final conclusion heuristic
    closer = safe_mean([r["closer_to_hard_ratio"] for r in overlap_rows])
    hard_like = safe_mean([r["hard_like_ratio_in_tail"] for r in tail_rows])
    aug_gap = float("nan")
    if augment_rows:
        hard_aug = [r["mean_cosine_distance"] for r in augment_rows if r["group"] == "hard_normal_top10"]
        easy_aug = [r["mean_cosine_distance"] for r in augment_rows if r["group"] == "easy_normal_bottom50"]
        if hard_aug and easy_aug:
            aug_gap = safe_mean(hard_aug) - safe_mean(easy_aug)

    if closer >= 0.60 and hard_like >= 0.60 and (not np.isfinite(aug_gap) or aug_gap > 0.0):
        conclusion = "hard-normal tail likely bottleneck"
    elif closer <= 0.40 and hard_like <= 0.40:
        conclusion = "hard-normal tail unlikely bottleneck"
    else:
        conclusion = "unclear"

    lines.extend([
        "",
        "## 8. Final Diagnostic Conclusion",
        "",
        f"`{conclusion}`",
        "",
        "Missing-path policy:",
        "",
    ])
    if missing_rows:
        lines.append("| seed | machine | kind | path |")
        lines.append("|---|---|---|---|")
        for row in missing_rows:
            lines.append(f"| {row['seed']} | {row['machine']} | {row['kind']} | {row['path']} |")
    else:
        lines.append("- no missing score files")
        lines.append("- no missing checkpoint paths for the executed sanity configuration")

    lines.append("")
    lines.append("This script performs no training, no optimizer step, no score tuning, and no per-machine method selection.")

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
