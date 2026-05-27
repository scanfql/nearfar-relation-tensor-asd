#!/usr/bin/env python
"""V63A: V37/V61 fixed score complementarity audit (no training)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


BOTTLENECKS = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="V63A V37/V61 score complementarity audit")
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--v37_root", required=True)
    p.add_argument("--v61_root", required=True)
    p.add_argument("--seeds", default="13711,13721,13731")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def tail_stats(scores: np.ndarray, y_true: np.ndarray) -> dict:
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    q95 = float(np.percentile(normal_scores, 95)) if len(normal_scores) else 0.0
    q99 = float(np.percentile(normal_scores, 99)) if len(normal_scores) else 0.0
    tpr95 = float(np.mean(anomaly_scores > q95)) if len(anomaly_scores) else 0.0
    tpr99 = float(np.mean(anomaly_scores > q99)) if len(anomaly_scores) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "TPR@95n": tpr95,
        "TPR@99n": tpr99,
        "anomaly_median": float(np.median(anomaly_scores)) if len(anomaly_scores) else 0.0,
        "anomaly_q75": float(np.percentile(anomaly_scores, 75)) if len(anomaly_scores) else 0.0,
    }


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    bundle = {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        "score_mode": str(arr["score_mode"]) if "score_mode" in arr.files else "latent_summary_knn",
    }
    return bundle


def expected_test_reference(data_root: Path, machine: str) -> dict:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return {
        "paths": np.asarray([str(x.path) for x in test_items], dtype=str),
        "y_test": np.asarray([x.label for x in test_items], dtype=np.int64),
        "domains": np.asarray([x.domain for x in test_items], dtype=str),
    }


def main():
    args = parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    alignment_rows = []
    metric_rows = []
    complement_rows = []
    alignment_issue = False

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        for seed in seeds:
            v37_path = Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v61_path = Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            exists_v37 = v37_path.exists()
            exists_v61 = v61_path.exists()
            row = {
                "seed": seed,
                "machine": machine,
                "v37_exists": exists_v37,
                "v61_exists": exists_v61,
                "test_filenames_match": False,
                "y_test_match": False,
                "domains_match": False,
                "sample_order_match": False,
                "alignment_ok": False,
                "issue": "",
            }
            if not exists_v37 or not exists_v61:
                row["issue"] = "missing_score_file"
                alignment_issue = True
                alignment_rows.append(row)
                continue

            b37 = load_bundle(v37_path)
            b61 = load_bundle(v61_path)
            y_match = np.array_equal(b37["y_test"], b61["y_test"]) and np.array_equal(b37["y_test"], expected["y_test"])
            d_match = np.array_equal(b37["domains"], b61["domains"]) and np.array_equal(b37["domains"], expected["domains"])
            if b61["test_paths"] is not None:
                filenames_match = np.array_equal(b61["test_paths"], expected["paths"])
                sample_order_match = filenames_match and y_match and d_match
            else:
                filenames_match = False
                sample_order_match = y_match and d_match

            row["test_filenames_match"] = filenames_match
            row["y_test_match"] = y_match
            row["domains_match"] = d_match
            row["sample_order_match"] = sample_order_match
            row["alignment_ok"] = y_match and d_match and sample_order_match
            if not row["alignment_ok"]:
                row["issue"] = "alignment_mismatch"
                alignment_issue = True
                alignment_rows.append(row)
                continue

            alignment_rows.append(row)

            c0 = b37["test_scores"]
            c1 = b61["test_scores"]
            c2 = 0.5 * rank_norm(c0) + 0.5 * rank_norm(c1)
            y_test = b37["y_test"]
            d_test = b37["domains"]

            for name, scores in [("C0_V37_raw", c0), ("C1_V61_raw", c1), ("C2_rank_avg_fixed", c2)]:
                met = compute_metrics(y_test, scores, d_test)
                geom = compute_score_geometry(scores, y_test, d_test)
                tail = tail_stats(scores, y_test)
                metric_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "candidate": name,
                    "hmean": met["hmean"],
                    "pAUC": met["pauc"],
                    "KS": geom["source_target_ks"],
                    "TPR@95n": tail["TPR@95n"],
                    "TPR@99n": tail["TPR@99n"],
                    "q95": tail["q95"],
                    "q99": tail["q99"],
                    "anomaly_median": tail["anomaly_median"],
                    "anomaly_q75": tail["anomaly_q75"],
                })

            q99_v37 = tail_stats(c0, y_test)["q99"]
            q99_v61 = tail_stats(c1, y_test)["q99"]
            normal_mask = y_test == 0
            anomaly_mask = y_test == 1
            top_v37 = set(np.where(normal_mask & (c0 >= q99_v37))[0].tolist())
            top_v61 = set(np.where(normal_mask & (c1 >= q99_v61))[0].tolist())
            missed_v37 = set(np.where(anomaly_mask & (c0 <= q99_v37))[0].tolist())
            missed_v61 = set(np.where(anomaly_mask & (c1 <= q99_v61))[0].tolist())
            top_union = len(top_v37 | top_v61)
            miss_union = len(missed_v37 | missed_v61)
            detect37_miss61 = int(np.sum(anomaly_mask & (c0 > q99_v37) & (c1 <= q99_v61)))
            detect61_miss37 = int(np.sum(anomaly_mask & (c1 > q99_v61) & (c0 <= q99_v37)))
            sp = spearmanr(c0, c1)
            complement_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_r": float(sp.correlation) if sp.correlation == sp.correlation else 0.0,
                "spearman_p": float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0,
                "top_q99_normal_overlap_count": len(top_v37 & top_v61),
                "top_q99_normal_union_count": top_union,
                "top_q99_normal_jaccard": float(len(top_v37 & top_v61) / top_union) if top_union else 1.0,
                "missed_anomaly_overlap_count": len(missed_v37 & missed_v61),
                "missed_anomaly_union_count": miss_union,
                "missed_anomaly_jaccard": float(len(missed_v37 & missed_v61) / miss_union) if miss_union else 1.0,
                "v37_detect_v61_miss_count": detect37_miss61,
                "v61_detect_v37_miss_count": detect61_miss37,
            })

    write_csv(output_root / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        write_csv(output_root / "candidate_metrics.csv", metric_rows)
        write_csv(output_root / "complementarity_diagnostics.csv", complement_rows)
        lines = [
            "# CTAF V63A V37/V61 Score Complementarity Audit",
            "",
            "## 1. Alignment Check",
            "",
            "Alignment failed on at least one seed/machine pair.",
            "Stop here. Do not evaluate score complementarity under misaligned samples.",
            "",
            "## 2. Final Verdict",
            "",
            "`unclear: alignment issue`",
            "",
            "## 3. Confirmation",
            "",
            "- no training occurred",
            "- no score files were modified",
            "- no fusion weight tuning was performed",
            "- no oracle candidate selection was used",
        ]
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text("\n".join(lines) + "\n")
        return

    write_csv(output_root / "candidate_metrics.csv", metric_rows)
    write_csv(output_root / "complementarity_diagnostics.csv", complement_rows)

    # Seed-level overall summaries (mean across machines).
    seed_rows = []
    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    for row in metric_rows:
        by_seed_cand[(row["seed"], row["candidate"])].append(row)
        by_machine_cand[(row["machine"], row["candidate"])].append(row)

    for (seed, cand), rows in sorted(by_seed_cand.items()):
        seed_rows.append({
            "seed": seed,
            "candidate": cand,
            "hmean": float(np.mean([r["hmean"] for r in rows])),
            "pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "KS": float(np.mean([r["KS"] for r in rows])),
            "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "q95": float(np.mean([r["q95"] for r in rows])),
            "q99": float(np.mean([r["q99"] for r in rows])),
        })
    write_csv(output_root / "seed_summary.csv", seed_rows)

    aggregate_rows = []
    for cand in ["C0_V37_raw", "C1_V61_raw", "C2_rank_avg_fixed"]:
        rows = [r for r in seed_rows if r["candidate"] == cand]
        aggregate_rows.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "mean_q95": float(np.mean([r["q95"] for r in rows])),
            "mean_q99": float(np.mean([r["q99"] for r in rows])),
        })
    write_csv(output_root / "aggregate_summary.csv", aggregate_rows)

    per_machine_rows = []
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "std_TPR@99n": float(np.std([r["TPR@99n"] for r in rows], ddof=0)),
            "mean_q99": float(np.mean([r["q99"] for r in rows])),
            "std_q99": float(np.std([r["q99"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
        })
    write_csv(output_root / "per_machine_summary.csv", per_machine_rows)

    # Verdict for C2
    agg = {r["candidate"]: r for r in aggregate_rows}
    pm = {(r["machine"], r["candidate"]): r for r in per_machine_rows}
    c2 = agg["C2_rank_avg_fixed"]
    c0 = agg["C0_V37_raw"]
    c1 = agg["C1_V61_raw"]
    best_h = max(c0["mean_hmean"], c1["mean_hmean"])
    best_p = max(c0["mean_pAUC"], c1["mean_pAUC"])
    bottleneck_improved = sum(
        pm[(m, "C2_rank_avg_fixed")]["mean_TPR@99n"] > pm[(m, "C1_V61_raw")]["mean_TPR@99n"]
        for m in BOTTLENECKS
    )
    toycar_ok = (
        pm[("toycar", "C2_rank_avg_fixed")]["mean_hmean"] >= pm[("toycar", "C1_V61_raw")]["mean_hmean"] - 1e-3
        and pm[("toycar", "C2_rank_avg_fixed")]["mean_pAUC"] >= pm[("toycar", "C1_V61_raw")]["mean_pAUC"] - 1e-3
    )
    bearing_ok = pm[("bearing", "C2_rank_avg_fixed")]["mean_hmean"] >= 0.50
    valve_ok = pm[("valve", "C2_rank_avg_fixed")]["mean_hmean"] >= 0.50
    verdict = "V63A fail: do not pursue score fusion"
    pass_flags = {
        "mean hmean >= max(V37,V61)-0.001": c2["mean_hmean"] >= best_h - 0.001,
        "mean pAUC >= max(V37,V61)-0.001": c2["mean_pAUC"] >= best_p - 0.001,
        "TPR@99n >= V61": c2["mean_TPR@99n"] >= c1["mean_TPR@99n"],
        ">=2 bottleneck TPR@99n improve over V61": bottleneck_improved >= 2,
        "toycar not degrade further than V61": toycar_ok,
        "bearing not collapse": bearing_ok,
        "valve not collapse": valve_ok,
        "no alignment issue": True,
        "no oracle candidate selection": True,
    }
    if all(pass_flags.values()):
        verdict = "V63A pass: fixed V37/V61 rank-fusion eligible for 3-seed panel summary"

    # doc
    lines = [
        "# CTAF V63A V37/V61 Score Complementarity Audit",
        "",
        "## 1. Score Alignment Check",
        "",
        f"- checked seeds: `{','.join(seeds)}`",
        f"- checked machines: `{','.join(machines)}`",
        "- V37 files: present",
        "- V61 files: present",
        "- y_test alignment: PASS",
        "- domain alignment: PASS",
        "- sample order alignment: PASS",
        "- no alignment stop condition triggered",
        "",
        "## 2. Candidate Definition",
        "",
        "- `C0 = V37 raw score`",
        "- `C1 = V61 raw score`",
        "- `C2 = 0.5 * rank_norm(V37) + 0.5 * rank_norm(V61)`",
        "- rank normalization is done independently per machine, per seed",
        "- no weight tuning, no oracle candidate selection",
        "",
        "## 3. 3-Seed Mean ± Std",
        "",
        "| candidate | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n | mean q95 | mean q99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        lines.append(
            f"| {r['candidate']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} | "
            f"{r['mean_TPR@95n']:.6f} | {r['mean_TPR@99n']:.6f} | {r['mean_q95']:.6f} | {r['mean_q99']:.6f} |"
        )
    lines.extend([
        "",
        "## 4. Per-Machine Mean ± Std",
        "",
        "| machine | candidate | mean hmean | std hmean | mean pAUC | std pAUC | mean TPR@99n | std TPR@99n | mean q99 | std q99 | mean KS | std KS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in per_machine_rows:
        lines.append(
            f"| {r['machine']} | {r['candidate']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_TPR@99n']:.6f} | {r['std_TPR@99n']:.6f} | "
            f"{r['mean_q99']:.6f} | {r['std_q99']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} |"
        )
    lines.extend([
        "",
        "## 5. Complementarity Diagnostics",
        "",
        "| seed | machine | spearman_r | top-q99 normal jaccard | missed-anomaly jaccard | V37-detect/V61-miss | V61-detect/V37-miss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for r in complement_rows:
        lines.append(
            f"| {r['seed']} | {r['machine']} | {r['spearman_r']:.6f} | {r['top_q99_normal_jaccard']:.6f} | "
            f"{r['missed_anomaly_jaccard']:.6f} | {r['v37_detect_v61_miss_count']} | {r['v61_detect_v37_miss_count']} |"
        )
    lines.extend([
        "",
        "## 6. Pass/Fail Gates for C2",
        "",
        "| gate | verdict |",
        "|---|---|",
    ])
    for k, v in pass_flags.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines.extend([
        "",
        "## 7. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "## 8. Confirmation",
        "",
        "- no training occurred",
        "- no score file was modified",
        "- no fusion weight tuning was performed",
        "- no per-machine fusion selection was used",
        "- no BEATs, V62A, UACS, or new training objective was introduced",
    ])
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
