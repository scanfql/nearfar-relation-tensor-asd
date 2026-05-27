#!/usr/bin/env python
"""Stage 16B-4: clean EAT fixed score fusion audit (no model forward)."""

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


def parse_args():
    p = argparse.ArgumentParser(description="Stage 16B clean EAT fixed fusion audit")
    p.add_argument("--mode", required=True, choices=["check", "fixed_fusion"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--eat_root", required=True)
    p.add_argument("--v63_root", required=True)
    p.add_argument("--stage14a_root", required=True)
    p.add_argument("--a3_root", default="")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--seeds", default="13711,13721,13731")
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
    q95 = float(np.percentile(normal_scores, 95))
    q99 = float(np.percentile(normal_scores, 99))
    tpr95 = float(np.mean(anomaly_scores > q95))
    tpr99 = float(np.mean(anomaly_scores > q99))
    return {
        "q95": q95,
        "q99": q99,
        "TPR@95n": tpr95,
        "TPR@99n": tpr99,
        "anomaly_median": float(np.median(anomaly_scores)),
        "anomaly_q75": float(np.percentile(anomaly_scores, 75)),
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


def candidate_specs(args):
    return {
        "clean_EAT": {"seeded": False, "path": lambda seed, m: Path(args.eat_root) / "scores" / f"{m}_clean_eat_knn_scores.npz"},
        "V63": {"seeded": True, "path": lambda seed, m: Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_rankfusion_scores.npz"},
        "A0_raw_BEATs": {"seeded": False, "path": lambda seed, m: Path(args.stage14a_root) / "a0_frozen_beats" / "scores" / f"{m}_frozen_beats_knn_scores.npz"},
        "A1": {"seeded": True, "path": lambda seed, m: Path(args.stage14a_root) / "a1_v63_beats_fusion" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_rankfusion_scores.npz"},
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    specs = candidate_specs(args)
    status = {
        "mode": "check",
        "clean_eat_root_exists": Path(args.eat_root).exists(),
        "v63_root_exists": Path(args.v63_root).exists(),
        "stage14a_root_exists": Path(args.stage14a_root).exists(),
        "data_root_exists": Path(args.data_root).exists(),
    }
    rows = []
    for cand, meta in specs.items():
        for machine in machines:
            for seed in seeds:
                if not meta["seeded"] and seed != seeds[0]:
                    continue
                path = meta["path"](seed, machine)
                rows.append({
                    "candidate": cand,
                    "machine": machine,
                    "seed": seed if meta["seeded"] else "shared",
                    "exists": path.exists(),
                    "path": str(path),
                })
    write_csv(out_root / "check_status.csv", rows)
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")


def run_fixed_fusion(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    specs = candidate_specs(args)

    alignment_rows = []
    comp_rows = []
    seed_rows = []
    per_machine_rows = []
    tail_rows = []

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        eat_key = ("clean_EAT", seeds[0])
        eat_bundle = load_bundle(specs["clean_EAT"]["path"](seeds[0], machine))
        for seed in seeds:
            bundles = {}
            for cand, meta in specs.items():
                actual_seed = seed if meta["seeded"] else seeds[0]
                bundles[cand] = load_bundle(meta["path"](actual_seed, machine))

            # align all to expected
            ok = True
            for cand, bundle in bundles.items():
                y_match = np.array_equal(bundle["y_test"], expected["y_test"])
                d_match = np.array_equal(bundle["domains"], expected["domains"])
                order_match = np.array_equal(bundle["test_paths"], expected["paths"]) if bundle["test_paths"] is not None else (y_match and d_match)
                alignment_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "candidate": cand,
                    "y_test_match": y_match,
                    "domains_match": d_match,
                    "sample_order_match": order_match,
                    "alignment_ok": bool(y_match and d_match and order_match),
                })
                ok = ok and y_match and d_match and order_match
            if not ok:
                raise RuntimeError(f"Alignment failed for machine={machine}, seed={seed}")

            y_test = bundles["V63"]["y_test"]
            domains = bundles["V63"]["domains"]
            c0 = bundles["clean_EAT"]["test_scores"]
            c1 = bundles["V63"]["test_scores"]
            c2 = bundles["A0_raw_BEATs"]["test_scores"]
            c3 = bundles["A1"]["test_scores"]

            candidates = {
                "clean_EAT": c0,
                "V63": c1,
                "A0_raw_BEATs": c2,
                "A1": c3,
                "F_EAT_V63": 0.5 * rank_norm(c0) + 0.5 * rank_norm(c1),
                "F_EAT_BEATs": 0.5 * rank_norm(c0) + 0.5 * rank_norm(c2),
                "F_EAT_A1": 0.5 * rank_norm(c0) + 0.5 * rank_norm(c3),
                "F_TRI_EAT_V63_BEATs": (rank_norm(c0) + rank_norm(c1) + rank_norm(c2)) / 3.0,
                "F_HMEAN_ORIENTED_EAT_A1": 0.65 * rank_norm(c0) + 0.35 * rank_norm(c3),
                "F_BALANCED_EAT_A1": 0.35 * rank_norm(c0) + 0.65 * rank_norm(c3),
            }

            sp_v63 = spearmanr(c0, c1).correlation
            sp_beats = spearmanr(c0, c2).correlation
            q99_v63 = tail_stats(c1, y_test)["q99"]
            q99_eat = tail_stats(c0, y_test)["q99"]
            normal_mask = y_test == 0
            anomaly_mask = y_test == 1
            top_v63 = set(np.where(normal_mask & (c1 >= q99_v63))[0].tolist())
            top_eat = set(np.where(normal_mask & (c0 >= q99_eat))[0].tolist())
            missed_v63 = set(np.where(anomaly_mask & (c1 <= q99_v63))[0].tolist())
            missed_eat = set(np.where(anomaly_mask & (c0 <= q99_eat))[0].tolist())
            top_union = len(top_v63 | top_eat)
            miss_union = len(missed_v63 | missed_eat)
            comp_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_cleanEAT_vs_V63": float(sp_v63) if sp_v63 == sp_v63 else 0.0,
                "spearman_cleanEAT_vs_rawBEATs": float(sp_beats) if sp_beats == sp_beats else 0.0,
                "top_normal_tail_jaccard": float(len(top_v63 & top_eat) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_v63 & missed_eat) / miss_union) if miss_union else 1.0,
            })

            for cand, scores in candidates.items():
                ev = evaluate(scores, y_test, domains)
                seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, **ev})
                fused_dir = scores_root / cand
                fused_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    fused_dir / f"{machine}_scores.npz",
                    test_scores=scores,
                    y_test=y_test,
                    domains=domains,
                    test_paths=expected["paths"],
                    score_mode=cand,
                )

    # aggregate
    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    by_machine_comp = defaultdict(list)
    for row in seed_rows:
        by_seed_cand[(row["seed"], row["candidate"])].append(row)
        by_machine_cand[(row["machine"], row["candidate"])].append(row)
    for row in comp_rows:
        by_machine_comp[row["machine"]].append(row)

    summary_rows = []
    for (seed, cand), rows in sorted(by_seed_cand.items()):
        summary_rows.append({
            "seed": seed,
            "candidate": cand,
            "hmean": float(np.mean([r["hmean"] for r in rows])),
            "pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "KS": float(np.mean([r["KS"] for r in rows])),
            "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "summary.csv", summary_rows)

    for (machine, cand), rows in sorted(by_machine_cand.items()):
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
        tail_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_q95": float(np.mean([r["q95"] for r in rows])),
            "mean_q99": float(np.mean([r["q99"] for r in rows])),
            "mean_anomaly_median": float(np.mean([r["anomaly_median"] for r in rows])),
            "mean_anomaly_q75": float(np.mean([r["anomaly_q75"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)

    # score geometry summary
    geom_summary = []
    for cand in sorted({r["candidate"] for r in per_machine_rows}):
        rows = [r for r in per_machine_rows if r["candidate"] == cand]
        geom_summary.append({
            "candidate": cand,
            "mean_KS": float(np.mean([r["mean_KS"] for r in rows])),
        })
    write_csv(out_root / "score_geometry.csv", geom_summary)

    # baseline comparison summary
    baseline_names = ["V63", "A0_raw_BEATs", "A1", "clean_EAT"]
    baseline_comp = []
    for cand in baseline_names:
        rows = [r for r in summary_rows if r["candidate"] == cand]
        baseline_comp.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "baseline_comparison.csv", baseline_comp)

    comp_summary_rows = []
    for machine, rows in sorted(by_machine_comp.items()):
        comp_summary_rows.append({
            "machine": machine,
            "mean_spearman_cleanEAT_vs_V63": float(np.mean([r["spearman_cleanEAT_vs_V63"] for r in rows])),
            "mean_spearman_cleanEAT_vs_rawBEATs": float(np.mean([r["spearman_cleanEAT_vs_rawBEATs"] for r in rows])),
            "mean_top_normal_tail_jaccard": float(np.mean([r["top_normal_tail_jaccard"] for r in rows])),
            "mean_missed_anomaly_jaccard": float(np.mean([r["missed_anomaly_jaccard"] for r in rows])),
        })
    write_csv(out_root / "complementarity_summary.csv", comp_summary_rows)

    Path(args.doc_path).write_text(
        "# CTAF Stage16B Clean EAT Fixed Fusion Audit\n\n"
        "Mode=fixed_fusion completed.\n"
    )


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "fixed_fusion":
        run_fixed_fusion(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
