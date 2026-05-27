#!/usr/bin/env python
"""V51 audit: learned one-class boundary on frozen V37 latent summaries."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SEEDS_DEFAULT = "13711,13721,13731"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="Audit V51 learned one-class boundary")
    p.add_argument("--data_root", required=True)
    p.add_argument("--score_seed_dirs", required=True)
    p.add_argument("--checkpoint_dirs", required=True)
    p.add_argument("--hard_normal_index_dir", required=True)
    p.add_argument("--seeds", default=SEEDS_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--experiment", default="full_5view")
    p.add_argument("--baseline_score_mode", default="latent_summary_knn")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def parse_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_seed_dirs(spec: str, seeds: list[str], suffix: str = "", static_shared: bool = False):
    parts = [Path(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) == 1:
        root = parts[0]
        if static_shared:
            return [root for _ in seeds]
        out = []
        for seed in seeds:
            cand = root / f"seed_{seed}"
            if suffix:
                cand = cand / suffix
            out.append(cand)
        return out
    if len(parts) != len(seeds):
        raise ValueError("directory spec must be one root or one path per seed")
    return parts


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(xs):
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [float(x) for x in xs]
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else 0.0


def tail_metrics(scores: np.ndarray, y_test: np.ndarray):
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95))
    q99 = float(np.quantile(normals, 0.99))
    return {
        "normal_q95": q95,
        "normal_q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "TPR_at_95n": float(np.mean(anomalies > q95)),
        "TPR_at_99n": float(np.mean(anomalies > q99)),
    }


def fmt(x): return f"{x:.6f}"
def fmt4(x): return f"{x:.4f}"


def main():
    args = parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    score_seed_dirs = parse_seed_dirs(args.score_seed_dirs, seeds)
    checkpoint_seed_dirs = parse_seed_dirs(args.checkpoint_dirs, seeds, static_shared=True)  # one shared models dir
    hard_seed_dirs = parse_seed_dirs(args.hard_normal_index_dir, seeds)

    from src.evaluation.metrics import compute_metrics
    from src.evaluation.score_geometry import compute_score_geometry

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    missing = []
    seed_table = []
    per_machine = []
    tail_rows = []
    train_outlier_rows = []

    # V37 baseline numbers (from summaries)
    v37_baseline = []

    for seed, score_dir, ckpt_dir, hard_dir in zip(seeds, score_seed_dirs, checkpoint_seed_dirs, hard_seed_dirs):
        summary_csv = score_dir / "summary.csv"
        if not summary_csv.exists():
            missing.append({"seed": seed, "machine": "--", "kind": "summary_csv", "path": str(summary_csv)})
            continue
        summary_rows = parse_csv(summary_csv)
        baseline_summary = next(
            (r for r in summary_rows if r["experiment"] == args.experiment and r["score_mode"] == args.baseline_score_mode),
            None,
        )
        if baseline_summary is None:
            missing.append({"seed": seed, "machine": "--", "kind": "baseline_summary_row", "path": str(summary_csv)})
            continue
        v37_baseline.append({
            "seed": seed,
            "hmean": float(baseline_summary["mean_hmean"]),
            "pauc": float(baseline_summary["mean_pauc"]),
            "ks": float(baseline_summary["mean_source_target_ks"]),
        })

        seed_h, seed_p, seed_ks = [], [], []
        for machine in machines:
            npz_path = score_dir / "scores" / f"{machine}_{args.experiment}_{args.baseline_score_mode}_scores.npz"
            if not npz_path.exists():
                missing.append({"seed": seed, "machine": machine, "kind": "score_npz", "path": str(npz_path)})
                continue

            arr = np.load(npz_path, allow_pickle=True)
            required = ["memory_latent_summaries", "test_latent_summaries", "y_test", "d_test"]
            if not all(k in arr.files for k in required):
                missing.append({"seed": seed, "machine": machine, "kind": "missing_latent_summary_arrays", "path": str(npz_path)})
                continue

            # checkpoint existence check only; do not load if not needed
            ckpt_path = ckpt_dir / f"{machine}_{args.experiment}_seed{seed}.pt"
            if not ckpt_path.exists():
                missing.append({"seed": seed, "machine": machine, "kind": "checkpoint", "path": str(ckpt_path)})

            train_features = arr["memory_latent_summaries"].astype(np.float64)
            test_features = arr["test_latent_summaries"].astype(np.float64)
            y_test = arr["y_test"].astype(np.int64)
            d_test = arr["d_test"].astype("U16")

            scaler = StandardScaler(with_mean=True, with_std=True)
            train_z = scaler.fit_transform(train_features)
            test_z = scaler.transform(test_features)

            oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
            oc.fit(train_z)
            test_scores = -oc.decision_function(test_z).astype(np.float64).ravel()
            metrics = compute_metrics(y_test, test_scores, d_test)
            geometry = compute_score_geometry(test_scores, y_test, d_test)

            seed_h.append(metrics["hmean"])
            seed_p.append(metrics["pauc"])
            seed_ks.append(geometry["source_target_ks"])
            per_machine.append({
                "seed": seed,
                "machine": machine,
                "score_mode": "v51_ocsvm_rbf",
                "hmean": metrics["hmean"],
                "pauc": metrics["pauc"],
                "source_auc": metrics["auc_source"],
                "target_auc": metrics["auc_target"],
                "ks": geometry["source_target_ks"],
            })

            tt = tail_metrics(test_scores, y_test)
            tail_rows.append({
                "seed": seed,
                "machine": machine,
                "score_mode": "v51_ocsvm_rbf",
                **tt,
            })

            # Diagnostic: OC-SVM boundary vs V48/V49 hard-normal tail
            hard_csv = hard_dir / f"{machine}_{args.experiment}_{args.baseline_score_mode}_hard_normal.csv"
            if not hard_csv.exists():
                missing.append({"seed": seed, "machine": machine, "kind": "hard_normal_index", "path": str(hard_csv)})
            else:
                rows = parse_csv(hard_csv)
                hard_flags = np.asarray([int(r["hard_normal_flag"]) for r in rows], dtype=np.int64)
                decision = oc.decision_function(train_z).astype(np.float64).ravel()
                train_outlier = decision < 0.0
                n_out = int(np.sum(train_outlier))
                overlap = int(np.sum(train_outlier & (hard_flags > 0)))
                train_outlier_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "n_train_normal": int(len(train_z)),
                    "n_train_outliers": n_out,
                    "outlier_rate": float(n_out / max(len(train_z), 1)),
                    "hard_normal_count": int(np.sum(hard_flags > 0)),
                    "overlap_count": overlap,
                    "outlier_hard_normal_overlap": float(overlap / max(n_out, 1)),
                    "hard_normal_recall": float(overlap / max(int(np.sum(hard_flags > 0)), 1)),
                })

        if seed_h:
            seed_table.append({
                "seed": seed,
                "score_mode": "v51_ocsvm_rbf",
                "hmean": float(np.mean(seed_h)),
                "pauc": float(np.mean(seed_p)),
                "ks": float(np.mean(seed_ks)),
            })

    write_csv(output_root / "seed_summary.csv", seed_table)
    write_csv(output_root / "per_machine.csv", per_machine)
    write_csv(output_root / "tail_table.csv", tail_rows)
    write_csv(output_root / "train_outlier_diagnostics.csv", train_outlier_rows)
    write_csv(output_root / "missing_paths.csv", missing)

    # Aggregate and report
    def collect_seed(mode_rows):
        return (
            safe_mean([r["hmean"] for r in mode_rows]),
            safe_std([r["hmean"] for r in mode_rows]),
            safe_mean([r["pauc"] for r in mode_rows]),
            safe_std([r["pauc"] for r in mode_rows]),
            safe_mean([r["ks"] for r in mode_rows]),
            safe_std([r["ks"] for r in mode_rows]),
        )

    v37_h, v37_hs, v37_p, v37_ps, v37_ks, v37_kss = collect_seed(v37_baseline)
    v51_h, v51_hs, v51_p, v51_ps, v51_ks, v51_kss = collect_seed(seed_table)

    # Success gates
    matched_pass = 0
    for seed in seeds:
        logmel_summary = next(
            r for r in parse_csv(score_seed_dirs[seeds.index(seed)] / "summary.csv")
            if r["experiment"] == "logmel_only" and r["score_mode"] == args.baseline_score_mode
        )
        v51_row = next(r for r in seed_table if r["seed"] == seed)
        if v51_row["hmean"] > float(logmel_summary["mean_hmean"]) and v51_row["pauc"] >= float(logmel_summary["mean_pauc"]):
            matched_pass += 1

    # per-machine aggregates for gates
    per_machine_agg = defaultdict(lambda: defaultdict(list))
    for row in per_machine:
        per_machine_agg[row["machine"]]["h"].append(row["hmean"])
        per_machine_agg[row["machine"]]["p"].append(row["pauc"])
        per_machine_agg[row["machine"]]["ks"].append(row["ks"])

    # baseline tail from V37 latent_summary
    baseline_tail = {}
    for seed, score_dir in zip(seeds, score_seed_dirs):
        for machine in machines:
            arr = np.load(score_dir / "scores" / f"{machine}_{args.experiment}_{args.baseline_score_mode}_scores.npz", allow_pickle=True)
            s = arr["test_scores"].astype(np.float64)
            y = arr["y_test"].astype(np.int64)
            tt = tail_metrics(s, y)
            baseline_tail.setdefault(machine, []).append(tt)

    baseline_tail_mean = {
        m: {k: safe_mean([v[k] for v in vals]) for k in vals[0].keys()}
        for m, vals in baseline_tail.items()
    }
    v51_tail_mean = defaultdict(dict)
    for row in tail_rows:
        v51_tail_mean[row["machine"]] = {k: row[k] for k in ["normal_q95", "normal_q99", "anomaly_median", "anomaly_q75", "TPR_at_95n", "TPR_at_99n"]}

    improved_tpr = 0
    for machine in BOTTLENECK:
        if v51_tail_mean[machine]["TPR_at_99n"] > baseline_tail_mean[machine]["TPR_at_99n"]:
            improved_tpr += 1

    q99_inflate = []
    for machine in machines:
        if v51_tail_mean[machine]["normal_q99"] > baseline_tail_mean[machine]["normal_q99"] * 1.05:
            q99_inflate.append(machine)

    bearing_collapse = safe_mean(per_machine_agg["bearing"]["h"]) < safe_mean([r["hmean"] for r in per_machine if r["machine"] == "bearing"])  # placeholder false
    # compare against V37 per-machine baselines
    v37_pm = defaultdict(lambda: defaultdict(list))
    for seed, score_dir in zip(seeds, score_seed_dirs):
        rows = parse_csv(score_dir / "per_machine.csv")
        for r in rows:
            if r["experiment"] == args.experiment and r["score_mode"] == args.baseline_score_mode:
                v37_pm[r["machine"]]["h"].append(float(r["hmean"]))
                v37_pm[r["machine"]]["p"].append(float(r["pauc"]))
                v37_pm[r["machine"]]["ks"].append(float(parse_csv(score_dir / "score_geometry.csv")[0]["source_target_ks"]) if False else 0.0)

    def mean_pm(machine, field):
        return safe_mean(v37_pm[machine][field])

    bearing_collapse = safe_mean(per_machine_agg["bearing"]["h"]) < mean_pm("bearing", "h") - 0.03
    valve_collapse = safe_mean(per_machine_agg["valve"]["h"]) < mean_pm("valve", "h") - 0.03
    toycar_degrade = safe_mean(per_machine_agg["toycar"]["p"]) < mean_pm("toycar", "p") - 0.01

    positive = (
        v51_h >= v37_h - 0.003
        and v51_p >= v37_p
        and improved_tpr >= 2
        and not bearing_collapse
        and not valve_collapse
        and not toycar_degrade
        and not q99_inflate
    )
    if positive:
        verdict = "V51 positive"
    else:
        # distinguish unclear from clear negative
        if v51_h < v37_h and v51_p < v37_p:
            verdict = "V51 negative"
        else:
            verdict = "unclear"

    lines = [
        "# CTAF V51 Learned One-Class Boundary Audit",
        "",
        "## 1. V51 prereg summary",
        "",
        "- V51 is a new Stage-8 hypothesis",
        "- it is not a continuation of V39-V50 parameter search",
        "- encoder / frontend / pooling remain frozen",
        "- only the low-FPR boundary scoring is replaced",
        "",
        "## 2. OC-SVM fixed protocol",
        "",
        "- model: One-Class SVM",
        "- kernel: `RBF`",
        "- `nu = 0.05`",
        "- `gamma = \"scale\"`",
        "- feature: frozen V37 full_5view latent_summary (192-dim)",
        "- preprocessing: train-normal feature-wise z-score",
        "- score = `-decision_function(x)`",
        "- one model per `seed / machine`",
        "- no sweep",
        "- no anomaly label",
        "",
        "## 3. Overall comparison",
        "",
        "| score_mode | mean hmean | mean pAUC | mean KS | std hmean | std pAUC | std KS |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V37 latent_summary_knn | {fmt(v37_h)} | {fmt(v37_p)} | {fmt(v37_ks)} | {fmt(v37_hs)} | {fmt(v37_ps)} | {fmt(v37_kss)} |",
        f"| V51 ocsvm_rbf | {fmt(v51_h)} | {fmt(v51_p)} | {fmt(v51_ks)} | {fmt(v51_hs)} | {fmt(v51_ps)} | {fmt(v51_kss)} |",
        "",
        "## 4. Seed-level table",
        "",
        "| seed | hmean | pAUC | KS |",
        "|---|---:|---:|---:|",
    ]
    for row in seed_table:
        lines.append(f"| {row['seed']} | {fmt(row['hmean'])} | {fmt(row['pauc'])} | {fmt(row['ks'])} |")

    lines.extend([
        "",
        "## 5. Per-machine table",
        "",
        "| machine | mean hmean | mean pAUC | mean KS |",
        "|---|---:|---:|---:|",
    ])
    for machine in machines:
        lines.append(
            f"| {machine} | {fmt(safe_mean(per_machine_agg[machine]['h']))} | "
            f"{fmt(safe_mean(per_machine_agg[machine]['p']))} | {fmt(safe_mean(per_machine_agg[machine]['ks']))} |"
        )

    lines.extend([
        "",
        "## 6. Tail table",
        "",
        "| machine | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for machine in machines:
        row = v51_tail_mean[machine]
        lines.append(
            f"| {machine} | {fmt4(row['normal_q95'])} | {fmt4(row['normal_q99'])} | "
            f"{fmt4(row['anomaly_median'])} | {fmt4(row['anomaly_q75'])} | "
            f"{fmt4(row['TPR_at_95n'])} | {fmt4(row['TPR_at_99n'])} |"
        )

    lines.extend([
        "",
        "## 7. q99 inflation check",
        "",
        f"- machines with >5% q99 inflation relative to V37 baseline: `{', '.join(q99_inflate) if q99_inflate else 'none'}`",
        "",
        "## 8. OC-SVM Boundary vs V48 Hard-Normal Tail",
        "",
        "| machine | n_train_outliers | outlier_rate | hard_normal_count | overlap_count | outlier_hard_normal_overlap | hard_normal_recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in train_outlier_rows:
        lines.append(
            f"| {row['machine']} | {row['n_train_outliers']} | {fmt4(row['outlier_rate'])} | "
            f"{row['hard_normal_count']} | {row['overlap_count']} | "
            f"{fmt4(row['outlier_hard_normal_overlap'])} | {fmt4(row['hard_normal_recall'])} |"
        )

    lines.extend([
        "",
        "Diagnostic interpretation rule:",
        "",
        "- high overlap suggests OC-SVM and V48 hard-normal tail are aligned",
        "- low overlap suggests OC-SVM is learning a different boundary structure",
        "- if train outlier rate deviates strongly from `nu=0.05`, the learned boundary behavior is not matching the nominal expectation exactly",
        "",
        "## 9. Final verdict",
        "",
        f"`{verdict}`",
        "",
        "Success gates:",
        "",
        f"- mean hmean >= V37 - 0.003: `{'PASS' if v51_h >= v37_h - 0.003 else 'FAIL'}`",
        f"- mean pAUC >= V37: `{'PASS' if v51_p >= v37_p else 'FAIL'}`",
        f"- >=2 bottleneck machines improve TPR@99n: `{improved_tpr}/5`",
        f"- bearing / valve do not collapse: `{'PASS' if not (bearing_collapse or valve_collapse) else 'FAIL'}`",
        f"- toycar does not clearly degrade: `{'PASS' if not toycar_degrade else 'FAIL'}`",
        f"- normal q99 does not clearly inflate: `{'PASS' if not q99_inflate else 'FAIL'}`",
        f"- matched baseline majority gate: `{matched_pass}/3` seeds",
        "",
        "Missing path policy:",
    ])
    if missing:
        lines.extend(["", "| seed | machine | kind | path |", "|---|---|---|---|"])
        for row in missing:
            lines.append(f"| {row['seed']} | {row['machine']} | {row['kind']} | {row['path']} |")
    else:
        lines.extend(["", "- no missing required score files", "- no missing required hard-normal indices"])

    lines.extend([
        "",
        "This audit trains only fixed OC-SVM boundaries on frozen latent summaries. No encoder/checkpoint training occurred. No protocol changes were made.",
    ])

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
