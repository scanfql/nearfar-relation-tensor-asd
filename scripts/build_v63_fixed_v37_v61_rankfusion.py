#!/usr/bin/env python
"""V63: materialize fixed V37/V61 0.5/0.5 rank-fusion frozen route."""

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
    p = argparse.ArgumentParser(description="Build V63 fixed rank-fusion frozen route")
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
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def expected_test_reference(data_root: Path, machine: str) -> dict:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return {
        "paths": np.asarray([str(x.path) for x in test_items], dtype=str),
        "y_test": np.asarray([x.label for x in test_items], dtype=np.int64),
        "domains": np.asarray([x.domain for x in test_items], dtype=str),
    }


def evaluate_candidate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_stats(scores, y_test)
    return {
        "hmean": met["hmean"],
        "pAUC": met["pauc"],
        "KS": geom["source_target_ks"],
        "TPR@95n": tail["TPR@95n"],
        "TPR@99n": tail["TPR@99n"],
        "q95": tail["q95"],
        "q99": tail["q99"],
        "anomaly_median": tail["anomaly_median"],
        "anomaly_q75": tail["anomaly_q75"],
    }


def main():
    args = parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    alignment_rows = []
    seed_rows = []
    per_machine_source_rows = []
    complement_rows = []
    frozen_manifest_rows = []
    alignment_issue = False

    # first pass: build and save fused npz, collect candidate metrics
    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        for seed in seeds:
            v37_npz = Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v61_npz = Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            align = {
                "seed": seed,
                "machine": machine,
                "v37_exists": v37_npz.exists(),
                "v61_exists": v61_npz.exists(),
                "test_filenames_match": False,
                "y_test_match": False,
                "domains_match": False,
                "sample_order_match": False,
                "alignment_ok": False,
                "issue": "",
            }
            if not v37_npz.exists() or not v61_npz.exists():
                align["issue"] = "missing_score_file"
                alignment_rows.append(align)
                alignment_issue = True
                continue

            b37 = load_bundle(v37_npz)
            b61 = load_bundle(v61_npz)

            y_match = np.array_equal(b37["y_test"], b61["y_test"]) and np.array_equal(b37["y_test"], expected["y_test"])
            d_match = np.array_equal(b37["domains"], b61["domains"]) and np.array_equal(b37["domains"], expected["domains"])
            if b61["test_paths"] is not None:
                filenames_match = np.array_equal(b61["test_paths"], expected["paths"])
                sample_order_match = filenames_match and y_match and d_match
            else:
                filenames_match = False
                sample_order_match = y_match and d_match

            align["test_filenames_match"] = filenames_match
            align["y_test_match"] = y_match
            align["domains_match"] = d_match
            align["sample_order_match"] = sample_order_match
            align["alignment_ok"] = y_match and d_match and sample_order_match
            if not align["alignment_ok"]:
                align["issue"] = "alignment_mismatch"
                alignment_rows.append(align)
                alignment_issue = True
                continue
            alignment_rows.append(align)

            c0 = b37["test_scores"]
            c1 = b61["test_scores"]
            rn37 = rank_norm(c0)
            rn61 = rank_norm(c1)
            c2 = 0.5 * rn37 + 0.5 * rn61
            y_test = b37["y_test"]
            domains = b37["domains"]

            for cand, scores in [("V37", c0), ("V61", c1), ("V63_fixed_rankfusion", c2)]:
                ev = evaluate_candidate(scores, y_test, domains)
                seed_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "candidate": cand,
                    **ev,
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
            sp = spearmanr(c0, c1)
            complement_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_r": float(sp.correlation) if sp.correlation == sp.correlation else 0.0,
                "spearman_p": float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0,
                "top_q99_normal_jaccard": float(len(top_v37 & top_v61) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_v37 & missed_v61) / miss_union) if miss_union else 1.0,
                "v37_detect_v61_miss_count": int(np.sum(anomaly_mask & (c0 > q99_v37) & (c1 <= q99_v61))),
                "v61_detect_v37_miss_count": int(np.sum(anomaly_mask & (c1 > q99_v61) & (c0 <= q99_v37))),
            })

            fused_dir = output_root / f"seed_{seed}" / "scores"
            fused_dir.mkdir(parents=True, exist_ok=True)
            fused_npz = fused_dir / f"{machine}_full_5view_rankfusion_scores.npz"
            np.savez(
                fused_npz,
                test_scores=c2.astype(np.float64),
                rank_norm_v37=rn37.astype(np.float64),
                rank_norm_v61=rn61.astype(np.float64),
                y_test=y_test,
                domains=domains,
                test_paths=expected["paths"],
                source_v37_score_path=str(v37_npz),
                source_v61_score_path=str(v61_npz),
                score_mode="v63_fixed_rankfusion",
                fusion_formula="0.5 * rank_norm(V37) + 0.5 * rank_norm(V61)",
            )
            frozen_manifest_rows.append({
                "seed": seed,
                "machine": machine,
                "candidate": "V63_fixed_rankfusion",
                "fused_npz_path": str(fused_npz),
                "v37_npz_path": str(v37_npz),
                "v61_npz_path": str(v61_npz),
                "fusion_formula": "0.5 * rank_norm(V37) + 0.5 * rank_norm(V61)",
                "alignment_ok": True,
            })

    write_csv(output_root / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        write_csv(output_root / "final_frozen_route_manifest.csv", frozen_manifest_rows)
        write_csv(output_root / "complementarity_summary.csv", complement_rows)
        lines = [
            "# CTAF V63 Fixed V37/V61 Rank-Fusion Audit",
            "",
            "## 1. V63A Summary",
            "",
            "Alignment failed on at least one seed/machine pair. Fixed route cannot be frozen under misalignment.",
            "",
            "## 2. Final Verdict",
            "",
            "`V63 frozen route FAIL`",
            "",
            "## 3. Confirmation",
            "",
            "- no training occurred",
            "- no original V37/V61 score file was modified",
            "- no fusion weight tuning was performed",
            "- no per-machine selection was used",
            "- no oracle selection was used",
            "- no V64 opened",
        ]
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text("\n".join(lines) + "\n")
        return

    # Build summary.csv as seed-level overall comparison rows
    summary_rows = []
    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    for row in seed_rows:
        by_seed_cand[(row["seed"], row["candidate"])].append(row)
        by_machine_cand[(row["machine"], row["candidate"])].append(row)

    for (seed, cand), rows in sorted(by_seed_cand.items()):
        summary_rows.append({
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
    write_csv(output_root / "summary.csv", summary_rows)

    # Aggregate route comparison for doc
    aggregate_rows = []
    for cand in ["V37", "V61", "V63_fixed_rankfusion"]:
        rows = [r for r in summary_rows if r["candidate"] == cand]
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
        })

    per_machine_rows = []
    tail_rows = []
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "std_TPR@99n": float(np.std([r["TPR@99n"] for r in rows], ddof=0)),
        })
        tail_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_q95": float(np.mean([r["q95"] for r in rows])),
            "std_q95": float(np.std([r["q95"] for r in rows], ddof=0)),
            "mean_q99": float(np.mean([r["q99"] for r in rows])),
            "std_q99": float(np.std([r["q99"] for r in rows], ddof=0)),
            "mean_anomaly_median": float(np.mean([r["anomaly_median"] for r in rows])),
            "mean_anomaly_q75": float(np.mean([r["anomaly_q75"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(output_root / "per_machine.csv", per_machine_rows)
    write_csv(output_root / "tail_table.csv", tail_rows)

    # complementarity summary aggregated per machine
    by_machine_comp = defaultdict(list)
    for row in complement_rows:
        by_machine_comp[row["machine"]].append(row)
    comp_summary_rows = []
    for machine, rows in sorted(by_machine_comp.items()):
        comp_summary_rows.append({
            "machine": machine,
            "mean_spearman_r": float(np.mean([r["spearman_r"] for r in rows])),
            "std_spearman_r": float(np.std([r["spearman_r"] for r in rows], ddof=0)),
            "mean_top_q99_normal_jaccard": float(np.mean([r["top_q99_normal_jaccard"] for r in rows])),
            "mean_missed_anomaly_jaccard": float(np.mean([r["missed_anomaly_jaccard"] for r in rows])),
            "sum_v37_detect_v61_miss": int(np.sum([r["v37_detect_v61_miss_count"] for r in rows])),
            "sum_v61_detect_v37_miss": int(np.sum([r["v61_detect_v37_miss_count"] for r in rows])),
        })
    write_csv(output_root / "complementarity_summary.csv", comp_summary_rows)
    write_csv(output_root / "final_frozen_route_manifest.csv", frozen_manifest_rows)

    agg = {r["candidate"]: r for r in aggregate_rows}
    pm = {(r["machine"], r["candidate"]): r for r in per_machine_rows}
    v37 = agg["V37"]
    v61 = agg["V61"]
    v63 = agg["V63_fixed_rankfusion"]
    bottleneck_improved = sum(
        pm[(m, "V63_fixed_rankfusion")]["mean_TPR@99n"] > pm[(m, "V61")]["mean_TPR@99n"]
        for m in BOTTLENECKS
    )
    verdict = "V63 frozen route FAIL"
    pass_flags = {
        "mean hmean >= max(V37,V61)-0.001": v63["mean_hmean"] >= max(v37["mean_hmean"], v61["mean_hmean"]) - 0.001,
        "mean pAUC >= max(V37,V61)-0.001": v63["mean_pAUC"] >= max(v37["mean_pAUC"], v61["mean_pAUC"]) - 0.001,
        "TPR@99n >= V61": v63["mean_TPR@99n"] >= v61["mean_TPR@99n"],
        ">=2 bottleneck machines improve TPR@99n over V61": bottleneck_improved >= 2,
        "toycar does not degrade further than V61": pm[("toycar", "V63_fixed_rankfusion")]["mean_hmean"] >= pm[("toycar", "V61")]["mean_hmean"] - 1e-3,
        "bearing / valve do not collapse": pm[("bearing", "V63_fixed_rankfusion")]["mean_hmean"] >= 0.50 and pm[("valve", "V63_fixed_rankfusion")]["mean_hmean"] >= 0.50,
        "no alignment issue": True,
        "no oracle candidate selection": True,
    }
    if all(pass_flags.values()):
        verdict = "V63 frozen route PASS"

    lines = [
        "# CTAF V63 Fixed V37/V61 Rank-Fusion Audit",
        "",
        "## 1. V63A Summary",
        "",
        "- V63A established that fixed 0.5/0.5 rank-normalized fusion is the only allowed fusion candidate.",
        "- No weight tuning, no per-machine selection, no oracle candidate selection.",
        "- V63 materializes that fixed route into frozen per-seed/per-machine score files.",
        "",
        "## 2. Alignment Confirmation",
        "",
        "- V37 files: present",
        "- V61 files: present",
        "- y_test alignment: PASS",
        "- domain alignment: PASS",
        "- sample order alignment: PASS",
        "",
        "## 3. Fixed Fusion Formula",
        "",
        "`C2 = 0.5 * rank_norm(V37_score) + 0.5 * rank_norm(V61_score)`",
        "",
        "- rank_norm is computed independently per machine and per seed",
        "- no weight tuning",
        "- no per-machine selection",
        "- no oracle selection",
        "",
        "## 4. 3-Seed Comparison",
        "",
        "| route | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        lines.append(
            f"| {r['candidate']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} | "
            f"{r['mean_TPR@95n']:.6f} | {r['mean_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 5. Per-Machine Comparison",
        "",
        "| machine | route | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@99n | std TPR@99n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in per_machine_rows:
        lines.append(
            f"| {r['machine']} | {r['candidate']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} | "
            f"{r['mean_TPR@99n']:.6f} | {r['std_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 6. TPR@99n Comparison",
        "",
        f"- V37 mean TPR@99n: `{v37['mean_TPR@99n']:.6f}`",
        f"- V61 mean TPR@99n: `{v61['mean_TPR@99n']:.6f}`",
        f"- V63 fixed rank fusion mean TPR@99n: `{v63['mean_TPR@99n']:.6f}`",
        f"- bottleneck machines improved over V61: `{bottleneck_improved}/5`",
        "",
        "## 7. pAUC / hmean Comparison",
        "",
        f"- V37: `hmean={v37['mean_hmean']:.6f}`, `pAUC={v37['mean_pAUC']:.6f}`",
        f"- V61: `hmean={v61['mean_hmean']:.6f}`, `pAUC={v61['mean_pAUC']:.6f}`",
        f"- V63: `hmean={v63['mean_hmean']:.6f}`, `pAUC={v63['mean_pAUC']:.6f}`",
        "",
        "## 8. q99 Note",
        "",
        "V63 uses rank-normalized fusion scores.",
        "Therefore V63 q95/q99 values live on the rank-normalized score scale and must not be directly compared against the raw-score q95/q99 values of V37 or V61.",
        "",
        "## 9. Complementarity Summary",
        "",
        "| machine | mean spearman_r | mean top-q99 normal jaccard | mean missed-anomaly jaccard | sum(V37-detect/V61-miss) | sum(V61-detect/V37-miss) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in comp_summary_rows:
        lines.append(
            f"| {r['machine']} | {r['mean_spearman_r']:.6f} | {r['mean_top_q99_normal_jaccard']:.6f} | "
            f"{r['mean_missed_anomaly_jaccard']:.6f} | {r['sum_v37_detect_v61_miss']} | {r['sum_v61_detect_v37_miss']} |"
        )
    lines.extend([
        "",
        "## 10. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "## 11. Confirmation",
        "",
        "- no training occurred",
        "- no original V37/V61 score file was modified",
        "- no fusion weight tuning was performed",
        "- no per-machine selection was used",
        "- no oracle selection was used",
        "- no V64 opened",
    ])
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
