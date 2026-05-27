#!/usr/bin/env python
"""Rescore saved CTAF latent-summary outputs with fixed label-free rules.

This avoids retraining when only the score function changes. The intended use
is to take a frozen experiment directory that already contains
train_latent_summaries/test_latent_summaries in each score NPZ and generate a
new result directory with a new fixed score mode.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Rescore saved latent-summary outputs")
    p.add_argument("--input_dir", required=True, help="Frozen results dir with per_machine.csv and scores/")
    p.add_argument("--output_dir", required=True, help="New results dir for rescored outputs")
    p.add_argument(
        "--score_modes",
        default="latent_summary_robust_knn",
        help=(
            "Comma-separated fixed score modes. Supported: "
            "latent_summary_robust_knn,latent_summary_blockscale_knn,"
            "latent_summary_whiten_knn"
        ),
    )
    p.add_argument("--latent_knn_k", type=int, default=5, help="Fixed k for latent kNN scoring")
    return p.parse_args()


def _resolve_score_path(project_root: Path, raw_path: str, input_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    candidate = project_root / p
    if candidate.exists():
        return candidate
    return input_dir / "scores" / p.name


def _write_summary_csv(path, results, geometry_rows):
    from collections import defaultdict

    agg = defaultdict(list)
    for r in results:
        agg[(r["experiment"], r["score_mode"])].append(r)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "score_mode", "num_views", "mean_source_auc",
            "mean_target_auc", "mean_pauc", "mean_hmean", "mean_dcase_auc",
            "mean_source_target_ks", "num_machines_improved_vs_logmel", "notes",
        ])

        experiments = ["logmel_only", "dynamic_3view", "edge_3view", "full_5view"]
        score_modes = []
        for r in results:
            if r["score_mode"] not in score_modes:
                score_modes.append(r["score_mode"])

        for exp_name in experiments:
            for score_mode in score_modes:
                rows = agg.get((exp_name, score_mode), [])
                if not rows:
                    continue
                mean_src = np.mean([r["source_auc"] for r in rows])
                mean_tgt = np.mean([r["target_auc"] for r in rows])
                mean_p = np.mean([r["pauc"] for r in rows])
                mean_h = np.mean([r["hmean"] for r in rows])
                mean_dcase = np.mean([r["dcase_auc"] for r in rows])
                ks_vals = [
                    g["source_target_ks"]
                    for g in geometry_rows
                    if g["experiment"] == exp_name and g["score_mode"] == score_mode
                ]
                baseline_rows = [
                    r for r in results
                    if r["experiment"] == "logmel_only" and r["score_mode"] == score_mode
                ]
                baseline_mean = (
                    float(np.mean([r["hmean"] for r in baseline_rows])) if baseline_rows else 0.0
                )
                logmel_map = {r["machine"]: r["hmean"] for r in baseline_rows}
                mean_ks = float(np.mean(ks_vals)) if ks_vals else 0.0
                n_improved = sum(
                    1 for r in rows
                    if r["machine"] in logmel_map and r["hmean"] > logmel_map[r["machine"]]
                )
                notes = ""
                if exp_name == "logmel_only":
                    notes = f"self_baseline_mean_hmean={baseline_mean:.6f}"
                w.writerow([
                    exp_name, score_mode, rows[0]["num_views"],
                    f"{mean_src:.6f}", f"{mean_tgt:.6f}", f"{mean_p:.6f}",
                    f"{mean_h:.6f}", f"{mean_dcase:.6f}", f"{mean_ks:.4f}",
                    n_improved, notes,
                ])


def _write_per_machine_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_score_geometry_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_results_md(path, args, results, geometry_rows):
    from collections import defaultdict

    by_group = defaultdict(list)
    for r in results:
        by_group[(r["experiment"], r["score_mode"])].append(r)

    with open(path, "w") as f:
        f.write("# CTAF Rescored Latent-Summary Results\n\n")
        f.write("## Protocol Confirmation\n\n")
        f.write("- **Model/features frozen**: scores are recomputed from saved latent summaries only.\n")
        f.write("- **Train-normal stats only**: any normalization/scaling is fit on train-normal latent summaries.\n")
        f.write("- **No labels in scoring**: labels are used only after scores are fixed for final metrics.\n")
        f.write("- **No machine-specific tuning**: one fixed rule for every machine and experiment.\n\n")
        f.write(f"**Input dir**: `{args.input_dir}`\n\n")
        f.write(f"**Output dir**: `{args.output_dir}`\n\n")
        f.write(f"**Score modes**: `{args.score_modes}`  with `k={args.latent_knn_k}`\n\n")

        f.write("## Overall Results\n\n")
        f.write("| Experiment | Score mode | Views | Mean src_AUC | Mean tgt_AUC | Mean pAUC | Mean hmean | Mean KS |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        experiments = ["logmel_only", "dynamic_3view", "edge_3view", "full_5view"]
        score_modes = []
        for r in results:
            if r["score_mode"] not in score_modes:
                score_modes.append(r["score_mode"])
        for exp_name in experiments:
            for score_mode in score_modes:
                rows = by_group.get((exp_name, score_mode), [])
                if not rows:
                    continue
                ks_vals = [
                    g["source_target_ks"]
                    for g in geometry_rows
                    if g["experiment"] == exp_name and g["score_mode"] == score_mode
                ]
                f.write(
                    f"| {exp_name} | {score_mode} | {rows[0]['num_views']} | "
                    f"{np.mean([r['source_auc'] for r in rows]):.4f} | "
                    f"{np.mean([r['target_auc'] for r in rows]):.4f} | "
                    f"{np.mean([r['pauc'] for r in rows]):.4f} | "
                    f"{np.mean([r['hmean'] for r in rows]):.4f} | "
                    f"{np.mean(ks_vals):.4f} |\n"
                )


def main():
    args = parse_args()

    from src.evaluation.metrics import compute_metrics
    from src.evaluation.plots import plot_score_histogram
    from src.evaluation.score_geometry import compute_score_geometry
    from src.evaluation.scorer import (
        apply_feature_block_scales,
        apply_feature_whitener,
        compute_latent_knn_scores,
        fit_feature_block_scales,
        fit_feature_robust_stats,
        fit_feature_whitener,
    )

    project_root = Path(__file__).resolve().parent.parent
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    score_modes = [s.strip() for s in args.score_modes.split(",") if s.strip()]
    allowed_modes = {
        "latent_summary_robust_knn",
        "latent_summary_blockscale_knn",
        "latent_summary_whiten_knn",
    }
    unknown_modes = sorted(set(score_modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"Unknown rescore mode(s): {','.join(unknown_modes)}")

    with open(input_dir / "per_machine.csv", newline="") as f:
        base_rows = list(csv.DictReader(f))

    latent_rows = [r for r in base_rows if r["score_mode"] == "latent_summary_knn"]
    if not latent_rows:
        raise ValueError("Input per_machine.csv does not contain latent_summary_knn rows")

    results = []
    geometry_rows = []

    for base_row in latent_rows:
        npz_path = _resolve_score_path(project_root, base_row["score_file"], input_dir)
        arr = np.load(npz_path)
        train_summaries = arr["train_latent_summaries"]
        test_summaries = arr["test_latent_summaries"]
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype("U16")
        feature_stats = fit_feature_robust_stats(train_summaries)
        block_stats = fit_feature_block_scales(train_summaries)
        whiten_stats = fit_feature_whitener(train_summaries)

        for score_mode in score_modes:
            if score_mode == "latent_summary_robust_knn":
                train_scores = compute_latent_knn_scores(
                    train_summaries,
                    train_summaries,
                    k=args.latent_knn_k,
                    train_feature_stats=feature_stats,
                )
                test_scores = compute_latent_knn_scores(
                    train_summaries,
                    test_summaries,
                    k=args.latent_knn_k,
                    train_feature_stats=feature_stats,
                )
            elif score_mode == "latent_summary_blockscale_knn":
                train_block = apply_feature_block_scales(train_summaries, block_stats)
                test_block = apply_feature_block_scales(test_summaries, block_stats)
                train_scores = compute_latent_knn_scores(
                    train_block, train_block, k=args.latent_knn_k
                )
                test_scores = compute_latent_knn_scores(
                    train_block, test_block, k=args.latent_knn_k
                )
            elif score_mode == "latent_summary_whiten_knn":
                train_white = apply_feature_whitener(train_summaries, whiten_stats)
                test_white = apply_feature_whitener(test_summaries, whiten_stats)
                train_scores = compute_latent_knn_scores(
                    train_white, train_white, k=args.latent_knn_k
                )
                test_scores = compute_latent_knn_scores(
                    train_white, test_white, k=args.latent_knn_k
                )
            else:
                raise ValueError(f"Unsupported score mode: {score_mode}")

            metrics = compute_metrics(y_test, test_scores, d_test)
            geometry = compute_score_geometry(test_scores, y_test, d_test)

            machine = base_row["machine"]
            exp_name = base_row["experiment"]
            score_label = f"{exp_name}_{score_mode}"
            hist_path = plot_score_histogram(
                test_scores, y_test, d_test, machine, score_label, str(figs_dir)
            )
            score_path = scores_dir / f"{machine}_{score_label}_scores.npz"
            np.savez_compressed(
                score_path,
                train_scores=train_scores,
                test_scores=test_scores,
                train_latent_summaries=train_summaries,
                test_latent_summaries=test_summaries,
                latent_summary_feature_median=feature_stats["median"],
                latent_summary_feature_iqr=feature_stats["iqr"],
                latent_summary_block_scales=block_stats["scales"],
                latent_summary_whiten_mean=whiten_stats["mean"],
                latent_summary_whiten_inv_sqrt=whiten_stats["inv_sqrt"],
                latent_knn_k=args.latent_knn_k,
                y_test=y_test,
                d_test=d_test,
                score_mode=score_mode,
                source_score_file=str(npz_path),
            )

            row = dict(base_row)
            row.update({
                "score_mode": score_mode,
                "source_auc": metrics["auc_source"],
                "target_auc": metrics["auc_target"],
                "pauc": metrics["pauc"],
                "dcase_auc": metrics["dcase_auc"],
                "hmean": metrics["hmean"],
                "latent_knn_k": args.latent_knn_k,
                "score_file": str(score_path),
                "hist_file": str(Path(hist_path)),
                "source_score_file": str(npz_path.relative_to(project_root)),
            })
            results.append(row)

            geometry_rows.append({
                "machine": machine,
                "experiment": exp_name,
                "score_mode": score_mode,
                "source_normal_mean": geometry["source_normal_mean"],
                "target_normal_mean": geometry["target_normal_mean"],
                "source_normal_std": geometry["source_normal_std"],
                "target_normal_std": geometry["target_normal_std"],
                "source_target_ks": geometry["source_target_ks"],
                "median_gap": geometry["median_gap"],
                "iqr_gap": geometry["iqr_gap"],
                "score_skewness": geometry["score_skewness"],
                "score_kurtosis": geometry["score_kurtosis"],
            })

    _write_summary_csv(str(output_dir / "summary.csv"), results, geometry_rows)
    _write_per_machine_csv(str(output_dir / "per_machine.csv"), results)
    _write_score_geometry_csv(str(output_dir / "score_geometry.csv"), geometry_rows)
    _write_results_md(str(docs_dir / "RESULTS_SUMMARY.md"), args, results, geometry_rows)

    print(f"summary       : {output_dir / 'summary.csv'}")
    print(f"per_machine   : {output_dir / 'per_machine.csv'}")
    print(f"score_geometry: {output_dir / 'score_geometry.csv'}")
    print(f"report        : {docs_dir / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
