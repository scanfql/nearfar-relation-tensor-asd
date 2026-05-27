#!/usr/bin/env python
"""V50 no-training audit for patch-level spatial latent summaries."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
import matplotlib
matplotlib.use("Agg")
import torch

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.evaluation.scorer import compute_latent_knn_scores
from src.features.logmel import extract_logmel, load_audio
from src.features.tensor_builder import TensorNormalizer, build_tensor
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder


SEEDS = [13711, 13721, 13731]
MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="V50 patch spatial summary audit")
    p.add_argument("--data_root", required=True)
    p.add_argument("--v37_score_root", required=True)
    p.add_argument("--v37_checkpoint_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--out_md", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def load_model(ckpt_path: Path, device: str):
    model = ViewwiseUnsharedLateFusionAutoencoder(
        in_channels=5,
        latent_channels=64,
        use_batch_norm=True,
        per_view_channels=16,
        stem_channels=8,
        dropout_p=0.0,
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    model = model.to(requested)
    model.eval()
    return model, requested


def build_machine_arrays(data_root: Path, machine: str):
    items = scan_dataset(data_root)
    items = filter_items(items, machine)
    train_items, test_items = split_train_test(items, use_target_train=False)

    train_tensors = []
    test_tensors = []
    test_labels = []
    test_domains = []
    for item in train_items:
        audio = load_audio(item.path)
        logmel = extract_logmel(audio)
        train_tensors.append(build_tensor("full_5view", logmel, audio=audio))
    for item in test_items:
        audio = load_audio(item.path)
        logmel = extract_logmel(audio)
        test_tensors.append(build_tensor("full_5view", logmel, audio=audio))
        test_labels.append(item.label)
        test_domains.append(item.domain)
    normalizer = TensorNormalizer()
    train_norm = normalizer.fit_transform(train_tensors)
    test_norm = [normalizer.transform(t) for t in test_tensors]
    return (
        np.stack(train_norm, axis=0).astype(np.float32),
        np.stack(test_norm, axis=0).astype(np.float32),
        np.asarray(test_labels, dtype=np.int64),
        np.asarray(test_domains),
    )


def compute_patch_2x2_summary_vectors(model, data: np.ndarray, device: str, batch_size: int = 32):
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    model = model.to(requested)
    out = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = torch.from_numpy(data[start:start + batch_size]).to(requested)
            latent = model.encode(x)  # [B, C, H, W]
            b, c, h, w = latent.shape
            h_mid = h // 2
            w_mid = w // 2
            patches = [
                latent[:, :, :h_mid, :w_mid],
                latent[:, :, :h_mid, w_mid:],
                latent[:, :, h_mid:, :w_mid],
                latent[:, :, h_mid:, w_mid:],
            ]
            patch_summaries = []
            for p in patches:
                mean = p.mean(dim=(2, 3))
                std = p.std(dim=(2, 3), unbiased=False)
                max_vals = p.amax(dim=(2, 3))
                patch_summaries.append(torch.cat([mean, std, max_vals], dim=1))
            summary = torch.cat(patch_summaries, dim=1)
            out.append(summary.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def zscore_fit(train_features: np.ndarray):
    mean = train_features.mean(axis=0).astype(np.float64)
    std = train_features.std(axis=0).astype(np.float64)
    std = np.maximum(std, 1e-8)
    return mean, std


def zscore_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((features - mean) / std).astype(np.float64)


def tail_metrics(npz_path: Path):
    arr = np.load(npz_path, allow_pickle=True)
    scores = arr["test_scores"].astype(np.float64)
    y = arr["y_test"].astype(np.int64)
    normals = scores[y == 0]
    anomalies = scores[y == 1]
    q95 = float(np.quantile(normals, 0.95))
    q99 = float(np.quantile(normals, 0.99))
    return {
        "normal_q95": q95,
        "normal_q99": q99,
        "anomaly_median": float(np.median(anomalies)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "tpr95n": float(np.mean(anomalies > q95)),
        "tpr99n": float(np.mean(anomalies > q99)),
    }


def fmt(x): return f"{x:.6f}"
def fmt4(x): return f"{x:.4f}"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    v37_score_root = Path(args.v37_score_root)
    v37_ckpt_root = Path(args.v37_checkpoint_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_md = Path(args.out_md)

    seed_rows = []
    per_machine_rows = []
    geometry_rows = []

    machine_cache = {}

    for seed in SEEDS:
        seed_dir = out_root / f"seed_{seed}"
        (seed_dir / "scores").mkdir(parents=True, exist_ok=True)
        score_dir = v37_score_root / f"seed_{seed}" / "scores"
        baseline_summary = list(csv.DictReader((v37_score_root / f"seed_{seed}" / "summary.csv").open()))
        baseline_summary_row = next(
            r for r in baseline_summary
            if r["experiment"] == "full_5view" and r["score_mode"] == "latent_summary_knn"
        )
        for machine in MACHINES:
            if machine not in machine_cache:
                machine_cache[machine] = build_machine_arrays(data_root, machine)
            train_arr, test_arr, y_test, d_test = machine_cache[machine]

            baseline_npz = score_dir / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            baseline = np.load(baseline_npz, allow_pickle=True)
            baseline_scores = baseline["test_scores"].astype(np.float64)
            baseline_metrics = compute_metrics(y_test, baseline_scores, d_test)
            baseline_geom = compute_score_geometry(baseline_scores, y_test, d_test)
            per_machine_rows.append({
                "seed": seed, "machine": machine, "score_mode": "global_latent_summary_knn",
                "hmean": baseline_metrics["hmean"], "pauc": baseline_metrics["pauc"],
                "source_auc": baseline_metrics["auc_source"], "target_auc": baseline_metrics["auc_target"],
                "ks": baseline_geom["source_target_ks"],
            })
            geometry_rows.append({"seed": seed, "machine": machine, "score_mode": "global_latent_summary_knn", **baseline_geom})

            ckpt = v37_ckpt_root / "v37_full5view_ts" / "models" / f"{machine}_full_5view_seed{seed}.pt"
            model, device = load_model(ckpt, args.device)
            train_patch = compute_patch_2x2_summary_vectors(model, train_arr, str(device))
            test_patch = compute_patch_2x2_summary_vectors(model, test_arr, str(device))
            mean, std = zscore_fit(train_patch)
            train_patch_z = zscore_apply(train_patch, mean, std)
            test_patch_z = zscore_apply(test_patch, mean, std)
            patch_scores = compute_latent_knn_scores(train_patch_z, test_patch_z, k=args.latent_knn_k)
            patch_metrics = compute_metrics(y_test, patch_scores, d_test)
            patch_geom = compute_score_geometry(patch_scores, y_test, d_test)

            score_path = seed_dir / "scores" / f"{machine}_full_5view_patch_2x2_summary_knn_scores.npz"
            np.savez_compressed(
                score_path,
                test_scores=patch_scores,
                y_test=y_test,
                d_test=d_test.astype("U16"),
                train_patch_summaries=train_patch,
                test_patch_summaries=test_patch,
                zscore_mean=mean,
                zscore_std=std,
                score_mode="patch_2x2_summary_knn",
            )

            per_machine_rows.append({
                "seed": seed, "machine": machine, "score_mode": "patch_2x2_summary_knn",
                "hmean": patch_metrics["hmean"], "pauc": patch_metrics["pauc"],
                "source_auc": patch_metrics["auc_source"], "target_auc": patch_metrics["auc_target"],
                "ks": patch_geom["source_target_ks"],
            })
            geometry_rows.append({"seed": seed, "machine": machine, "score_mode": "patch_2x2_summary_knn", **patch_geom})

        # add one aggregate baseline row from V37 summary and one aggregate patch row from current audit
        patch_rows_seed = [r for r in per_machine_rows if r["seed"] == seed and r["score_mode"] == "patch_2x2_summary_knn"]
        patch_geo_seed = [g for g in geometry_rows if g["seed"] == seed and g["score_mode"] == "patch_2x2_summary_knn"]
        seed_rows.append({
            "seed": seed,
            "score_mode": "global_latent_summary_knn",
            "hmean": float(baseline_summary_row["mean_hmean"]),
            "pauc": float(baseline_summary_row["mean_pauc"]),
            "ks": float(baseline_summary_row["mean_source_target_ks"]),
        })
        seed_rows.append({
            "seed": seed,
            "score_mode": "patch_2x2_summary_knn",
            "hmean": float(np.mean([r["hmean"] for r in patch_rows_seed])),
            "pauc": float(np.mean([r["pauc"] for r in patch_rows_seed])),
            "ks": float(np.mean([g["source_target_ks"] for g in patch_geo_seed])),
        })

    # write csvs
    def write_csv(path: Path, rows):
        if not rows:
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_root / "seed_summary.csv", seed_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "score_geometry.csv", geometry_rows)

    # tail table averaged across seeds
    tail_rows = []
    for machine in BOTTLENECK + ["bearing", "valve"]:
        for mode in ["global_latent_summary_knn", "patch_2x2_summary_knn"]:
            vals = []
            for seed in SEEDS:
                if mode == "global_latent_summary_knn":
                    npz = v37_score_root / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
                else:
                    npz = out_root / f"seed_{seed}" / "scores" / f"{machine}_full_5view_patch_2x2_summary_knn_scores.npz"
                vals.append(tail_metrics(npz))
            tail_rows.append({
                "machine": machine,
                "score_mode": mode,
                "normal_q95": np.mean([v["normal_q95"] for v in vals]),
                "normal_q99": np.mean([v["normal_q99"] for v in vals]),
                "anomaly_median": np.mean([v["anomaly_median"] for v in vals]),
                "anomaly_q75": np.mean([v["anomaly_q75"] for v in vals]),
                "tpr95n": np.mean([v["tpr95n"] for v in vals]),
                "tpr99n": np.mean([v["tpr99n"] for v in vals]),
            })
    write_csv(out_root / "tail_table.csv", tail_rows)

    # markdown
    def collect(mode):
        rows = [r for r in seed_rows if r["score_mode"] == mode]
        h = [r["hmean"] for r in rows]; p = [r["pauc"] for r in rows]; ks = [r["ks"] for r in rows]
        return np.mean(h), np.std(h, ddof=1), np.mean(p), np.std(p, ddof=1), np.mean(ks), np.std(ks, ddof=1)

    g_h, g_hs, g_p, g_ps, g_ks, g_kss = collect("global_latent_summary_knn")
    p_h, p_hs, p_p, p_ps, p_ks, p_kss = collect("patch_2x2_summary_knn")
    improved_tpr = 0
    for machine in BOTTLENECK:
        base = next(r["tpr99n"] for r in tail_rows if r["machine"] == machine and r["score_mode"] == "global_latent_summary_knn")
        patch = next(r["tpr99n"] for r in tail_rows if r["machine"] == machine and r["score_mode"] == "patch_2x2_summary_knn")
        if patch > base:
            improved_tpr += 1
    q99_bad = 0
    for machine in BOTTLENECK + ["bearing", "valve"]:
        base_q95 = next(r["normal_q95"] for r in tail_rows if r["machine"] == machine and r["score_mode"] == "global_latent_summary_knn")
        base_q99 = next(r["normal_q99"] for r in tail_rows if r["machine"] == machine and r["score_mode"] == "global_latent_summary_knn")
        patch_q95 = next(r["normal_q95"] for r in tail_rows if r["machine"] == machine and r["score_mode"] == "patch_2x2_summary_knn")
        patch_q99 = next(r["normal_q99"] for r in tail_rows if r["machine"] == machine and r["score_mode"] == "patch_2x2_summary_knn")
        base_tail_width = (base_q99 - base_q95) / max(abs(base_q95), 1e-8)
        patch_tail_width = (patch_q99 - patch_q95) / max(abs(patch_q95), 1e-8)
        if patch_tail_width > base_tail_width * 1.10:
            q99_bad += 1
    bv_collapse = False
    for machine in ["bearing", "valve"]:
        base_h = np.mean([r["hmean"] for r in per_machine_rows if r["machine"] == machine and r["score_mode"] == "global_latent_summary_knn"])
        patch_h = np.mean([r["hmean"] for r in per_machine_rows if r["machine"] == machine and r["score_mode"] == "patch_2x2_summary_knn"])
        if patch_h < base_h - 0.03:
            bv_collapse = True
    toycar_base = np.mean([r["pauc"] for r in per_machine_rows if r["machine"] == "toycar" and r["score_mode"] == "global_latent_summary_knn"])
    toycar_patch = np.mean([r["pauc"] for r in per_machine_rows if r["machine"] == "toycar" and r["score_mode"] == "patch_2x2_summary_knn"])
    toycar_degrade = toycar_patch < toycar_base - 0.01

    positive = (
        p_h >= g_h - 0.003
        and p_p >= g_p
        and improved_tpr >= 2
        and q99_bad == 0
        and not bv_collapse
        and not toycar_degrade
    )
    if positive:
        verdict = "patch_2x2 positive"
    elif (p_h < g_h - 0.003 and p_p < g_p) or bv_collapse:
        verdict = "patch_2x2 negative"
    else:
        verdict = "unclear"

    lines = [
        "# CTAF V50 Patch Spatial Summary Audit",
        "",
        "## 1. Stage 5 / V49 Summary",
        "",
        "- mechanism 3-view frontends cannot replace full_5view",
        "- hard-normal consistency compresses q99 but fails as deployable method",
        "- current mainline remains full_5view + latent_summary_knn",
        "",
        "## 2. Hypothesis",
        "",
        "- global pooling may erase spatial local evidence",
        "- patch 2x2 summary preserves coarse spatial structure",
        "",
        "## 3. Overall Comparison",
        "",
        "| score_mode | mean_hmean | mean_pAUC | mean_KS | std_hmean | std_pAUC | std_KS |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| global_latent_summary_knn | {fmt(g_h)} | {fmt(g_p)} | {fmt(g_ks)} | {fmt(g_hs)} | {fmt(g_ps)} | {fmt(g_kss)} |",
        f"| patch_2x2_summary_knn | {fmt(p_h)} | {fmt(p_p)} | {fmt(p_ks)} | {fmt(p_hs)} | {fmt(p_ps)} | {fmt(p_kss)} |",
        "",
        "## 4. Seed-Panel Table",
        "",
        "| seed | score_mode | hmean | pAUC | KS |",
        "|---|---|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for mode in ["global_latent_summary_knn", "patch_2x2_summary_knn"]:
            row = next(r for r in seed_rows if r["seed"] == seed and r["score_mode"] == mode)
            lines.append(f"| {seed} | {mode} | {fmt(row['hmean'])} | {fmt(row['pauc'])} | {fmt(row['ks'])} |")

    lines.extend([
        "",
        "## 5. Per-Machine hmean / pAUC / KS",
        "",
        "| machine | score_mode | mean hmean | mean pAUC | mean KS |",
        "|---|---|---:|---:|---:|",
    ])
    for machine in MACHINES:
        for mode in ["global_latent_summary_knn", "patch_2x2_summary_knn"]:
            rows = [r for r in per_machine_rows if r["machine"] == machine and r["score_mode"] == mode]
            lines.append(
                f"| {machine} | {mode} | {fmt(np.mean([r['hmean'] for r in rows]))} | "
                f"{fmt(np.mean([r['pauc'] for r in rows]))} | {fmt(np.mean([r['ks'] for r in rows]))} |"
            )

    lines.extend([
        "",
        "## 6. Tail Table",
        "",
        "| machine | score_mode | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tail_rows:
        lines.append(
            f"| {row['machine']} | {row['score_mode']} | {fmt4(row['normal_q95'])} | {fmt4(row['normal_q99'])} | "
            f"{fmt4(row['anomaly_median'])} | {fmt4(row['anomaly_q75'])} | {fmt4(row['tpr95n'])} | {fmt4(row['tpr99n'])} |"
        )

    lines.extend([
        "",
        "## 7. q99 Inflation Check",
        "",
        "- q99 inflation is checked by relative tail-width inflation:",
        "  `(q99 - q95) / q95` under patch_2x2 versus global summary.",
        f"- machines with >10% relative tail-width inflation under patch_2x2: `{q99_bad}`",
        "",
        "## 8. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "Success criteria check:",
        "",
        f"- mean hmean >= global - 0.003: `{'PASS' if p_h >= g_h - 0.003 else 'FAIL'}`",
        f"- mean pAUC >= global: `{'PASS' if p_p >= g_p else 'FAIL'}`",
        f"- >= 2 bottleneck machines improve TPR@99n: `{improved_tpr}/5`",
        f"- normal q99 does not inflate clearly: `{'PASS' if q99_bad == 0 else 'FAIL'}`",
        f"- bearing/valve do not collapse: `{'PASS' if not bv_collapse else 'FAIL'}`",
        f"- toycar does not clearly degrade: `{'PASS' if not toycar_degrade else 'FAIL'}`",
        "",
        "This is a no-training audit. No checkpoint was modified. No grid search, no score fusion, and no per-machine selection were used.",
    ])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
