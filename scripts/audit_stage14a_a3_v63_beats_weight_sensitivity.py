#!/usr/bin/env python
"""Stage 14A-A3: fixed-weight V63/raw-BEATs rank-fusion sensitivity audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]
ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 14A-A3 V63/BEATs weight sensitivity audit")
    p.add_argument("--mode", required=True, choices=["check", "a3_weight_sensitivity"])
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--a0_root", default="results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats")
    p.add_argument("--v37_root", default="results/v37_teacher_student_pretrain_seedpanel")
    p.add_argument("--v61_root", default="results/v61_cross_year_original_encoder_seedpanel")
    p.add_argument("--seeds", default="13711,13721,13731")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    return p.parse_args()


def write_csv(path: Path, rows):
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


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}"


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


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


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
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


def expected_test_reference(data_root: Path, machine: str) -> dict:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return {
        "paths": np.asarray([str(x.path) for x in test_items], dtype=str),
        "y_test": np.asarray([x.label for x in test_items], dtype=np.int64),
        "domains": np.asarray([x.domain for x in test_items], dtype=str),
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]

    all_v63 = all(
        (Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz").exists()
        for seed in seeds for machine in machines
    )
    all_beats = all(
        (Path(args.a0_root) / "scores" / f"{machine}_frozen_beats_knn_scores.npz").exists()
        for machine in machines
    )
    status = {
        "mode": "check",
        "v63_root_exists": Path(args.v63_root).exists(),
        "a0_root_exists": Path(args.a0_root).exists(),
        "v37_root_exists": Path(args.v37_root).exists(),
        "v61_root_exists": Path(args.v61_root).exists(),
        "data_root_exists": Path(args.data_root).exists(),
        "all_v63_score_files_exist": all_v63,
        "all_a0_raw_beats_score_files_exist": all_beats,
    }
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")
    Path(args.doc_path).write_text(
        "# CTAF Stage14A A3 Weight Sensitivity Audit\n\n"
        "Mode=check completed. No metrics computed.\n"
    )


def run_a3(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]

    alignment_rows = []
    seed_rows = []
    comp_rows = []
    alignment_issue = False

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        a0_npz = Path(args.a0_root) / "scores" / f"{machine}_frozen_beats_knn_scores.npz"
        if not a0_npz.exists():
            raise FileNotFoundError(f"Missing A0 raw BEATs score for {machine}: {a0_npz}")
        beats = load_bundle(a0_npz)
        for seed in seeds:
            v37_npz = Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v61_npz = Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v63_npz = Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"
            row = {
                "seed": seed,
                "machine": machine,
                "v63_exists": v63_npz.exists(),
                "beats_exists": a0_npz.exists(),
                "y_test_match": False,
                "domains_match": False,
                "sample_order_match": False,
                "alignment_ok": False,
                "issue": "",
            }
            if not v63_npz.exists():
                row["issue"] = "missing_score_file"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            b37 = load_bundle(v37_npz)
            b61 = load_bundle(v61_npz)
            v63 = load_bundle(v63_npz)
            y_match = np.array_equal(v63["y_test"], beats["y_test"]) and np.array_equal(v63["y_test"], expected["y_test"])
            d_match = np.array_equal(v63["domains"], beats["domains"]) and np.array_equal(v63["domains"], expected["domains"])
            paths_match = v63["test_paths"] is not None and beats["test_paths"] is not None and np.array_equal(v63["test_paths"], beats["test_paths"]) and np.array_equal(v63["test_paths"], expected["paths"])
            row["y_test_match"] = y_match
            row["domains_match"] = d_match
            row["sample_order_match"] = paths_match
            row["alignment_ok"] = y_match and d_match and paths_match
            if not row["alignment_ok"]:
                row["issue"] = "alignment_mismatch"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            alignment_rows.append(row)

            c_v37 = b37["test_scores"]
            c_v61 = b61["test_scores"]
            c_v63 = v63["test_scores"]
            c_beats = beats["test_scores"]
            rn_v63 = rank_norm(c_v63)
            rn_beats = rank_norm(c_beats)
            y_test = v63["y_test"]
            domains = v63["domains"]

            # base routes
            for cand, scores in [("V37", c_v37), ("V61", c_v61), ("V63", c_v63), ("raw_BEATs", c_beats)]:
                ev = evaluate(scores, y_test, domains)
                seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, "alpha": "", **ev})

            for alpha in ALPHAS:
                fused = alpha * rn_v63 + (1.0 - alpha) * rn_beats
                ev = evaluate(fused, y_test, domains)
                cand = f"alpha_{alpha_tag(alpha)}"
                seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, "alpha": alpha, **ev})
                fused_dir = out_root / "a3_weight_sensitivity" / f"alpha_{alpha_tag(alpha)}" / f"seed_{seed}" / "scores"
                fused_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    fused_dir / f"{machine}_v63_beats_alpha_{alpha_tag(alpha)}_scores.npz",
                    test_scores=fused,
                    y_test=y_test,
                    domains=domains,
                    test_paths=expected["paths"],
                    score_mode=f"v63_beats_alpha_{alpha_tag(alpha)}",
                    alpha=alpha,
                    source_v63_score_path=str(v63_npz),
                    source_beats_score_path=str(a0_npz),
                    fusion_formula=f"{alpha:.2f} * rank_norm(V63) + {1-alpha:.2f} * rank_norm(BEATs)",
                )

            q99_v63 = tail_stats(c_v63, y_test)["q99"]
            q99_b = tail_stats(c_beats, y_test)["q99"]
            normal_mask = y_test == 0
            anomaly_mask = y_test == 1
            top_v63 = set(np.where(normal_mask & (c_v63 >= q99_v63))[0].tolist())
            top_b = set(np.where(normal_mask & (c_beats >= q99_b))[0].tolist())
            missed_v63 = set(np.where(anomaly_mask & (c_v63 <= q99_v63))[0].tolist())
            missed_b = set(np.where(anomaly_mask & (c_beats <= q99_b))[0].tolist())
            top_union = len(top_v63 | top_b)
            miss_union = len(missed_v63 | missed_b)
            sp = spearmanr(c_v63, c_beats)
            comp_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_r": float(sp.correlation) if sp.correlation == sp.correlation else 0.0,
                "spearman_p": float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0,
                "top_normal_tail_jaccard": float(len(top_v63 & top_b) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_v63 & missed_b) / miss_union) if miss_union else 1.0,
                "v63_detect_beats_miss": int(np.sum(anomaly_mask & (c_v63 > q99_v63) & (c_beats <= q99_b))),
                "beats_detect_v63_miss": int(np.sum(anomaly_mask & (c_beats > q99_b) & (c_v63 <= q99_v63))),
            })

    write_csv(out_root / "a3_weight_sensitivity" / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        Path(args.doc_path).write_text(
            "# CTAF Stage14A A3 Weight Sensitivity Audit\n\n"
            "Mode=a3_weight_sensitivity failed due to alignment issue.\n"
        )
        return

    write_csv(out_root / "a3_weight_sensitivity" / "seed_metrics.csv", seed_rows)
    write_csv(out_root / "a3_weight_sensitivity" / "complementarity_diagnostics.csv", comp_rows)

    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    for row in seed_rows:
        by_seed_cand[(row["seed"], row["candidate"])].append(row)
        by_machine_cand[(row["machine"], row["candidate"])].append(row)

    summary_rows = []
    for (seed, cand), rows in sorted(by_seed_cand.items()):
        alpha = rows[0]["alpha"] if rows[0]["alpha"] != "" else ""
        summary_rows.append({
            "seed": seed,
            "candidate": cand,
            "alpha": alpha,
            "hmean": float(np.mean([r["hmean"] for r in rows])),
            "pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "KS": float(np.mean([r["KS"] for r in rows])),
            "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "summary.csv", summary_rows)

    candidate_order = ["V37", "V61", "V63", "raw_BEATs"] + [f"alpha_{alpha_tag(a)}" for a in ALPHAS]
    aggregate_rows = []
    for cand in candidate_order:
        rows = [r for r in summary_rows if r["candidate"] == cand]
        aggregate_rows.append({
            "candidate": cand,
            "alpha": rows[0]["alpha"] if rows and rows[0]["alpha"] != "" else "",
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "alpha_sensitivity.csv", aggregate_rows)

    per_machine_rows = []
    tail_rows = []
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        alpha = rows[0]["alpha"] if rows[0]["alpha"] != "" else ""
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "alpha": alpha,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
        tail_rows.append({
            "machine": machine,
            "candidate": cand,
            "alpha": alpha,
            "mean_q95": float(np.mean([r["q95"] for r in rows])),
            "mean_q99": float(np.mean([r["q99"] for r in rows])),
            "mean_anomaly_median": float(np.mean([r["anomaly_median"] for r in rows])),
            "mean_anomaly_q75": float(np.mean([r["anomaly_q75"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)

    by_machine_comp = defaultdict(list)
    for row in comp_rows:
        by_machine_comp[row["machine"]].append(row)
    comp_summary_rows = []
    for machine, rows in sorted(by_machine_comp.items()):
        comp_summary_rows.append({
            "machine": machine,
            "mean_spearman_r": float(np.mean([r["spearman_r"] for r in rows])),
            "mean_top_normal_tail_jaccard": float(np.mean([r["top_normal_tail_jaccard"] for r in rows])),
            "mean_missed_anomaly_jaccard": float(np.mean([r["missed_anomaly_jaccard"] for r in rows])),
            "sum_v63_detect_beats_miss": int(np.sum([r["v63_detect_beats_miss"] for r in rows])),
            "sum_beats_detect_v63_miss": int(np.sum([r["beats_detect_v63_miss"] for r in rows])),
        })
    write_csv(out_root / "complementarity_summary.csv", comp_summary_rows)

    # report
    agg = {r["candidate"]: r for r in aggregate_rows}
    a1 = {"mean_hmean": 0.5813, "mean_pAUC": 0.5484, "mean_TPR@99n": 0.0976}
    best_alpha = max(
        [r for r in aggregate_rows if str(r["candidate"]).startswith("alpha_")],
        key=lambda r: r["mean_hmean"],
    )
    pm = {(r["machine"], r["candidate"]): r for r in per_machine_rows}
    a1_hmean_map = {
        "bearing": 0.5828,
        "fan": 0.5553,
        "gearbox": 0.5571,
        "slider": 0.5660,
        "toycar": 0.5510,
        "toytrain": 0.5888,
        "valve": 0.6684,
    }
    lines = [
        "# CTAF Stage14A A3 Weight Sensitivity Audit",
        "",
        "## 1. Alignment Check",
        "",
        "- V63 scores exist",
        "- raw BEATs scores exist",
        "- y_test aligned",
        "- d_test/domain aligned",
        "- sample order aligned",
        "",
        "## 2. Alpha Sensitivity Table",
        "",
        "| candidate | alpha | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        lines.append(
            f"| {r['candidate']} | {r['alpha']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} | "
            f"{r['mean_TPR@95n']:.6f} | {r['mean_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 3. Comparison Against V63 / A0 / A1",
        "",
        f"- V63 mean hmean = `{agg['V63']['mean_hmean']:.6f}`",
        f"- V63 mean pAUC = `{agg['V63']['mean_pAUC']:.6f}`",
        f"- raw BEATs mean hmean = `{agg['raw_BEATs']['mean_hmean']:.6f}`",
        f"- raw BEATs mean pAUC = `{agg['raw_BEATs']['mean_pAUC']:.6f}`",
        f"- A1 reference hmean = `{a1['mean_hmean']:.4f}`",
        f"- A1 reference pAUC = `{a1['mean_pAUC']:.4f}`",
        f"- A1 reference TPR@99n = `{a1['mean_TPR@99n']:.4f}`",
        "",
        "## 4. Hmean-Oriented Analysis",
        "",
        f"- highest-hmean alpha: `{best_alpha['candidate']}` with mean hmean `{best_alpha['mean_hmean']:.6f}`",
        f"- pAUC drop vs A1: `{best_alpha['mean_pAUC'] - a1['mean_pAUC']:.6f}`",
        "",
        "## 5. Complementarity Summary",
        "",
        "| machine | mean spearman_r | mean top normal-tail jaccard | mean missed-anomaly jaccard | V63-detect / BEATs-miss | BEATs-detect / V63-miss |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in comp_summary_rows:
        lines.append(
            f"| {r['machine']} | {r['mean_spearman_r']:.6f} | {r['mean_top_normal_tail_jaccard']:.6f} | "
            f"{r['mean_missed_anomaly_jaccard']:.6f} | {r['sum_v63_detect_beats_miss']} | {r['sum_beats_detect_v63_miss']} |"
        )
    lines.extend([
        "",
        "## 6. Interpretation Rule",
        "",
        "Do not replace the A1 frozen route automatically.",
        "If some alpha beats A1 on hmean, report it only as an hmean-oriented exploratory best unless separately frozen in a new preregistration.",
        "",
        "## 7. Confirmation",
        "",
        "- no training occurred",
        "- no BEATs fine-tuning occurred",
        "- no projection-head training occurred",
        "- no fusion weight tuning occurred",
        "- no per-machine alpha selection occurred",
        "- no Stage 14B opened",
    ])
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "a3_weight_sensitivity":
        run_a3(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
