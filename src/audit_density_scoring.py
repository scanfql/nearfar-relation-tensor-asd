#!/usr/bin/env python
"""Audit fixed density scoring rules on frozen latent-summary representations."""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Density scoring audit on saved latent summaries")
    p.add_argument("--input_seedpanel_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--seed_list", default="13711,13721,13731")
    return p.parse_args()


def read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, scores, domains):
    from src.evaluation.metrics import compute_metrics as _compute_metrics
    return _compute_metrics(y_true, scores, domains)


def compute_score_geometry(scores, y_true, domains):
    from src.evaluation.score_geometry import compute_score_geometry as _compute_score_geometry
    return _compute_score_geometry(scores, y_true, domains)


def gaussian_shrinkage_nll(train_features: np.ndarray, test_features: np.ndarray):
    lw = LedoitWolf().fit(train_features)
    mean = lw.location_.astype(np.float64)
    precision = lw.precision_.astype(np.float64)
    sign, logdet_precision = np.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("precision matrix is not positive definite")
    d = train_features.shape[1]
    centered = test_features.astype(np.float64) - mean
    mahal = np.einsum("bi,ij,bj->b", centered, precision, centered)
    logdet_cov = -logdet_precision
    return 0.5 * (mahal + logdet_cov + d * np.log(2.0 * np.pi))


def gmm_diag_nll(train_features: np.ndarray, test_features: np.ndarray, k: int = 4):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=0,
        reg_covar=1e-6,
        max_iter=200,
    )
    gmm.fit(train_features.astype(np.float64))
    return -gmm.score_samples(test_features.astype(np.float64))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_tail(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    s = z["test_scores"]
    y = z["y_test"]
    normals = s[y == 0]
    anomalies = s[y == 1]
    return {
        "normal_q95": float(np.quantile(normals, 0.95)),
        "normal_q99": float(np.quantile(normals, 0.99)),
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "tpr95n": float((anomalies > np.quantile(normals, 0.95)).mean()),
        "tpr99n": float((anomalies > np.quantile(normals, 0.99)).mean()),
    }


def fmt(x):
    return f"{x:.6f}"


def fmt4(x):
    return f"{x:.4f}"


def main():
    args = parse_args()
    from src.evaluation.plots import plot_score_histogram

    input_root = Path(args.input_seedpanel_dir)
    output_root = Path(args.output_dir)
    output_md = Path(args.output_md)
    ensure_dir(output_root)

    seeds = [s.strip() for s in args.seed_list.split(",") if s.strip()]
    score_modes = ["latent_summary_knn", "density_gaussian_shrinkage", "density_gmm_diag_k4"]
    experiments = ["logmel_only", "full_5view"]

    all_seed_rows = []
    all_per_machine_rows = []
    all_geometry_rows = []

    for seed in seeds:
        seed_in = input_root / f"seed_{seed}"
        seed_out = output_root / f"seed_{seed}"
        ensure_dir(seed_out / "scores")
        ensure_dir(seed_out / "figures")

        base_rows = read_csv(seed_in / "per_machine.csv")
        latent_rows = [
            r for r in base_rows
            if r["score_mode"] == "latent_summary_knn" and r["experiment"] in experiments
        ]

        per_machine_rows = []
        geometry_rows = []
        for row in latent_rows:
            npz_path = seed_in / "scores" / Path(row["score_file"]).name
            arr = np.load(npz_path, allow_pickle=True)
            train_summaries = arr["memory_latent_summaries"]
            test_summaries = arr["test_latent_summaries"]
            y_test = arr["y_test"].astype(np.int64)
            d_test = arr["d_test"].astype("U16")

            mode_to_scores = {
                "latent_summary_knn": arr["test_scores"].astype(np.float64),
                "density_gaussian_shrinkage": gaussian_shrinkage_nll(train_summaries, test_summaries),
                "density_gmm_diag_k4": gmm_diag_nll(train_summaries, test_summaries, k=4),
            }

            for score_mode, test_scores in mode_to_scores.items():
                metrics = compute_metrics(y_test, test_scores, d_test)
                geometry = compute_score_geometry(test_scores, y_test, d_test)
                score_label = f"{row['experiment']}_{score_mode}"
                hist_path = plot_score_histogram(
                    test_scores, y_test, d_test, row["machine"], score_label, str(seed_out / "figures")
                )
                score_path = seed_out / "scores" / f"{row['machine']}_{score_label}_scores.npz"
                np.savez_compressed(
                    score_path,
                    test_scores=test_scores,
                    y_test=y_test,
                    d_test=d_test,
                    train_latent_summaries=train_summaries,
                    test_latent_summaries=test_summaries,
                    score_mode=score_mode,
                    source_score_file=str(npz_path),
                )

                out_row = dict(row)
                out_row.update({
                    "score_mode": score_mode,
                    "source_auc": metrics["auc_source"],
                    "target_auc": metrics["auc_target"],
                    "pauc": metrics["pauc"],
                    "dcase_auc": metrics["dcase_auc"],
                    "hmean": metrics["hmean"],
                    "score_file": str(score_path),
                    "hist_file": hist_path,
                })
                per_machine_rows.append(out_row)
                all_per_machine_rows.append({"seed": seed, **out_row})

                geo_row = {
                    "seed": seed,
                    "machine": row["machine"],
                    "experiment": row["experiment"],
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
                }
                geometry_rows.append(geo_row)
                all_geometry_rows.append(geo_row)

        # seed summary
        seed_summary_rows = []
        for exp in experiments:
            for mode in score_modes:
                rows = [r for r in per_machine_rows if r["experiment"] == exp and r["score_mode"] == mode]
                if not rows:
                    continue
                ks_rows = [g["source_target_ks"] for g in geometry_rows if g["experiment"] == exp and g["score_mode"] == mode]
                seed_summary_rows.append({
                    "experiment": exp,
                    "score_mode": mode,
                    "num_views": rows[0]["num_views"],
                    "mean_source_auc": fmt(np.mean([float(r["source_auc"]) for r in rows])),
                    "mean_target_auc": fmt(np.mean([float(r["target_auc"]) for r in rows])),
                    "mean_pauc": fmt(np.mean([float(r["pauc"]) for r in rows])),
                    "mean_hmean": fmt(np.mean([float(r["hmean"]) for r in rows])),
                    "mean_dcase_auc": fmt(np.mean([float(r["dcase_auc"]) for r in rows])),
                    "mean_source_target_ks": f"{np.mean(ks_rows):.4f}",
                    "num_machines_improved_vs_logmel": "",
                    "notes": "",
                })
                all_seed_rows.append({"seed": seed, **seed_summary_rows[-1]})

        write_csv(seed_out / "per_machine.csv", per_machine_rows)
        write_csv(seed_out / "score_geometry.csv", geometry_rows)
        write_csv(seed_out / "summary.csv", seed_summary_rows)

    # report
    v16_h = 0.549547
    v16_p = 0.542331
    v16_ks = 0.294880
    v37_h = 0.553898
    v37_p = 0.545689
    v37_ks = 0.305733

    summary_map = defaultdict(dict)
    for row in all_seed_rows:
        summary_map[row["seed"]][(row["experiment"], row["score_mode"])] = row

    per_machine_map = defaultdict(lambda: defaultdict(list))
    for row in all_per_machine_rows:
        if row["experiment"] == "full_5view":
            per_machine_map[(row["machine"], row["score_mode"])]["h"].append(float(row["hmean"]))
            per_machine_map[(row["machine"], row["score_mode"])]["p"].append(float(row["pauc"]))
    geom_map = defaultdict(list)
    for row in all_geometry_rows:
        if row["experiment"] == "full_5view":
            geom_map[(row["machine"], row["score_mode"])].append(float(row["source_target_ks"]))

    tail_modes = ["latent_summary_knn", "density_gaussian_shrinkage", "density_gmm_diag_k4"]
    bottlenecks = ["gearbox", "slider", "toycar", "toytrain", "fan"]
    tail_table = []
    for machine in bottlenecks:
        for mode in tail_modes:
            vals = []
            for seed in seeds:
                npz = output_root / f"seed_{seed}" / "scores" / f"{machine}_full_5view_{mode}_scores.npz"
                vals.append(summarize_tail(npz))
            tail_table.append({
                "machine": machine,
                "score_mode": mode,
                "normal_q95": np.mean([v["normal_q95"] for v in vals]),
                "normal_q99": np.mean([v["normal_q99"] for v in vals]),
                "anomaly_median": np.mean([v["anomaly_median"] for v in vals]),
                "anomaly_q75": np.mean([v["anomaly_q75"] for v in vals]),
                "tpr95n": np.mean([v["tpr95n"] for v in vals]),
                "tpr99n": np.mean([v["tpr99n"] for v in vals]),
            })

    lines = []
    lines.append("# CTAF V38 Density Scoring Audit")
    lines.append("")
    lines.append("## 1. V16 Seed-Panel Mean Baseline")
    lines.append("")
    lines.append("```text")
    lines.append(f"hmean = {v16_h:.6f}")
    lines.append(f"pAUC  = {v16_p:.6f}")
    lines.append(f"KS    = {v16_ks:.6f}")
    lines.append("```")
    lines.append("")
    lines.append("## 2. V37 latent_summary_knn Baseline")
    lines.append("")
    lines.append("```text")
    lines.append(f"hmean = {v37_h:.6f}")
    lines.append(f"pAUC  = {v37_p:.6f}")
    lines.append(f"KS    = {v37_ks:.6f}")
    lines.append("bottleneck TPR@99n improvement = 0/5")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Seed-Panel Table")
    lines.append("")
    lines.append("| seed | score_mode | hmean | pAUC | KS |")
    lines.append("|---|---|---:|---:|---:|")
    for seed in seeds:
        for mode in score_modes:
            row = summary_map[seed][("full_5view", mode)]
            lines.append(f"| {seed} | {mode} | {row['mean_hmean']} | {row['mean_pauc']} | {row['mean_source_target_ks']} |")
    lines.append("")
    lines.append("## 4. Aggregate Mean/Std")
    lines.append("")
    lines.append("| score_mode | mean_hmean | mean_pAUC | mean_KS | std_hmean | std_pAUC | std_KS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in score_modes:
        h = [float(summary_map[s][("full_5view", mode)]["mean_hmean"]) for s in seeds]
        p = [float(summary_map[s][("full_5view", mode)]["mean_pauc"]) for s in seeds]
        ks = [float(summary_map[s][("full_5view", mode)]["mean_source_target_ks"]) for s in seeds]
        lines.append(
            f"| {mode} | {fmt(np.mean(h))} | {fmt(np.mean(p))} | {fmt(np.mean(ks))} | "
            f"{fmt(np.std(h, ddof=1))} | {fmt(np.std(p, ddof=1))} | {fmt(np.std(ks, ddof=1))} |"
        )
    lines.append("")
    lines.append("## 5. Per-Machine hmean / pAUC / KS")
    lines.append("")
    lines.append("| machine | score_mode | mean hmean | mean pAUC | mean KS |")
    lines.append("|---|---|---:|---:|---:|")
    for machine in sorted({k[0] for k in per_machine_map}):
        for mode in score_modes:
            h = np.mean(per_machine_map[(machine, mode)]["h"])
            p = np.mean(per_machine_map[(machine, mode)]["p"])
            ks = np.mean(geom_map[(machine, mode)])
            lines.append(f"| {machine} | {mode} | {fmt(h)} | {fmt(p)} | {fmt(ks)} |")
    lines.append("")
    lines.append("## 6. Bottleneck-Machine Tail Table")
    lines.append("")
    lines.append("| machine | score_mode | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in tail_table:
        lines.append(
            f"| {row['machine']} | {row['score_mode']} | {fmt4(row['normal_q95'])} | "
            f"{fmt4(row['normal_q99'])} | {fmt4(row['anomaly_median'])} | {fmt4(row['anomaly_q75'])} | "
            f"{fmt4(row['tpr95n'])} | {fmt4(row['tpr99n'])} |"
        )
    lines.append("")
    lines.append("## 7. Final Verdict")
    lines.append("")
    lines.append("Success criteria check:")
    lines.append("")
    for mode in ["density_gaussian_shrinkage", "density_gmm_diag_k4"]:
        h = [float(summary_map[s][("full_5view", mode)]["mean_hmean"]) for s in seeds]
        p = [float(summary_map[s][("full_5view", mode)]["mean_pauc"]) for s in seeds]
        ks = [float(summary_map[s][("full_5view", mode)]["mean_source_target_ks"]) for s in seeds]
        matched = sum(
            float(summary_map[s][("full_5view", mode)]["mean_hmean"]) > float(summary_map[s][("logmel_only", mode)]["mean_hmean"])
            and float(summary_map[s][("full_5view", mode)]["mean_pauc"]) >= float(summary_map[s][("logmel_only", mode)]["mean_pauc"])
            for s in seeds
        )
        improved_tpr = 0
        for machine in bottlenecks:
            base_tpr = next(r["tpr99n"] for r in tail_table if r["machine"] == machine and r["score_mode"] == "latent_summary_knn")
            mode_tpr = next(r["tpr99n"] for r in tail_table if r["machine"] == machine and r["score_mode"] == mode)
            if mode_tpr > base_tpr:
                improved_tpr += 1
        lines.append(f"- `{mode}` matched-baseline gate: `{matched}/3` seeds")
        lines.append(
            f"- `{mode}` seed-panel mean: hmean={fmt(np.mean(h))}, pAUC={fmt(np.mean(p))}, KS={fmt(np.mean(ks))}"
        )
        lines.append(f"- `{mode}` bottleneck TPR@99n improvements: `{improved_tpr}/5`")
    lines.append("")
    lines.append(
        "Density scoring does not solve the low-FPR bottleneck if it fails to improve "
        "bottleneck-machine `TPR@99n`, even when overall hmean/pAUC remain competitive."
    )
    lines.append("")
    lines.append("```text")
    lines.append("If both density scores fail the low-FPR gate, record V38 as a scoring-rule audit failure.")
    lines.append("```")

    output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
