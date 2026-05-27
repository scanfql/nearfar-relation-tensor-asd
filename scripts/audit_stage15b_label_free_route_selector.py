#!/usr/bin/env python
"""Stage 15B: label-free machine-level route selector feasibility audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 15B label-free route selector audit")
    p.add_argument("--mode", required=True, choices=["feature_audit", "selector_audit"])
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--stage15a_root", default="results/stage15a_existing_route_hmean_ceiling")
    p.add_argument("--v37_root", default="results/v37_teacher_student_pretrain_seedpanel")
    p.add_argument("--v61_root", default="results/v61_cross_year_original_encoder_seedpanel")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--stage14a_root", default="results/stage14a_frozen_beats_v63_fusion")
    p.add_argument("--a3_root", default="results/stage14a_a3_v63_beats_weight_sensitivity")
    p.add_argument("--a4_root", default="results/stage14a_a4_cross_year_frozen_beats_memory")
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


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        "train_scores": arr["train_scores"].astype(np.float64) if "train_scores" in arr.files else None,
    }


def expected_test_reference(data_root: Path, machine: str) -> dict:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return {
        "paths": np.asarray([str(x.path) for x in test_items], dtype=str),
        "y_test": np.asarray([x.label for x in test_items], dtype=np.int64),
        "domains": np.asarray([x.domain for x in test_items], dtype=str),
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
    }


def score_entropy(scores: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(scores, bins=bins, density=False)
    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob)).sum())


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    corr = spearmanr(a, b).correlation
    return float(corr) if corr == corr else 0.0


def top_tail_overlap(scores_a: np.ndarray, scores_b: np.ndarray, q: float = 0.99) -> float:
    qa = float(np.quantile(scores_a, q))
    qb = float(np.quantile(scores_b, q))
    set_a = set(np.where(scores_a >= qa)[0].tolist())
    set_b = set(np.where(scores_b >= qb)[0].tolist())
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return float(len(set_a & set_b) / union)


def candidate_specs(args):
    return {
        "V37": {"seeded": True, "path": lambda seed, m: Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_latent_summary_knn_scores.npz", "beats_like": False},
        "V61": {"seeded": True, "path": lambda seed, m: Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_latent_summary_knn_scores.npz", "beats_like": False},
        "V63": {"seeded": True, "path": lambda seed, m: Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_rankfusion_scores.npz", "beats_like": False},
        "A0_raw_BEATs": {"seeded": False, "path": lambda seed, m: Path(args.stage14a_root) / "a0_frozen_beats" / "scores" / f"{m}_frozen_beats_knn_scores.npz", "beats_like": True},
        "A1_alpha_0.50": {"seeded": True, "path": lambda seed, m: Path(args.stage14a_root) / "a1_v63_beats_fusion" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_rankfusion_scores.npz", "beats_like": True},
        "A3_alpha_0.20": {"seeded": True, "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.20" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.20_scores.npz", "beats_like": True},
        "A3_alpha_0.35": {"seeded": True, "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.35" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.35_scores.npz", "beats_like": True},
        "A3_alpha_0.65": {"seeded": True, "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.65" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.65_scores.npz", "beats_like": True},
        "A3_alpha_0.80": {"seeded": True, "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.80" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.80_scores.npz", "beats_like": True},
        "A4_B2s": {"seeded": False, "path": lambda seed, m: Path(args.a4_root) / "b2_cross_year_beats" / "scores" / f"{m}_b2_source_beats_scores.npz", "beats_like": True},
        "A4_F2s": {"seeded": True, "path": lambda seed, m: Path(args.a4_root) / "f2_v63_cross_year_beats_fusion" / f"seed_{seed}" / "scores" / f"{m}_f2s_scores.npz", "beats_like": True},
    }


def gather_candidates(args):
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    specs = candidate_specs(args)
    missing = []
    bundles = {}
    alignment_issue = False
    alignment_rows = []
    feature_rows = []

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        # reference scores for feature construction
        ref_v63 = None
        ref_a0 = None
        # first pass load available bundles
        local = {}
        for cand, meta in specs.items():
            for seed in seeds:
                if not meta["seeded"] and seed != seeds[0]:
                    continue
                path = meta["path"](seed, machine)
                key = (cand, seed, machine)
                if not path.exists():
                    missing.append({"candidate": cand, "seed": seed if meta["seeded"] else "shared", "machine": machine, "path": str(path)})
                    continue
                bundle = load_bundle(path)
                y_match = np.array_equal(bundle["y_test"], expected["y_test"])
                d_match = np.array_equal(bundle["domains"], expected["domains"])
                order_match = np.array_equal(bundle["test_paths"], expected["paths"]) if bundle["test_paths"] is not None else (y_match and d_match)
                alignment_ok = bool(y_match and d_match and order_match)
                alignment_rows.append({
                    "candidate": cand,
                    "seed": seed if meta["seeded"] else "shared",
                    "machine": machine,
                    "y_test_match": y_match,
                    "domains_match": d_match,
                    "sample_order_match": order_match,
                    "alignment_ok": alignment_ok,
                })
                if not alignment_ok:
                    alignment_issue = True
                    continue
                bundles[key] = bundle
                local[key] = bundle

        for seed in seeds:
            v63_key = ("V63", seed, machine)
            a0_key = ("A0_raw_BEATs", seeds[0], machine)
            if v63_key not in local:
                continue
            ref_v63 = local[v63_key]["test_scores"]
            ref_a0 = local[a0_key]["test_scores"] if a0_key in local else None
            for cand, meta in specs.items():
                key = (cand, seed, machine) if meta["seeded"] else (cand, seeds[0], machine)
                if key not in local:
                    continue
                bundle = local[key]
                scores = bundle["test_scores"]
                domains = bundle["domains"]
                src = scores[domains == "source"]
                tgt = scores[domains == "target"]
                train_scores = bundle["train_scores"]
                feat = {
                    "seed": seed,
                    "machine": machine,
                    "candidate": cand,
                    "source_score_mean": float(np.mean(src)) if len(src) else 0.0,
                    "target_score_mean": float(np.mean(tgt)) if len(tgt) else 0.0,
                    "source_score_std": float(np.std(src)) if len(src) else 0.0,
                    "target_score_std": float(np.std(tgt)) if len(tgt) else 0.0,
                    "source_target_ks": float(ks_2samp(src, tgt).statistic) if len(src) and len(tgt) else 0.0,
                    "score_q50": float(np.quantile(scores, 0.50)),
                    "score_q90": float(np.quantile(scores, 0.90)),
                    "score_q95": float(np.quantile(scores, 0.95)),
                    "score_q99": float(np.quantile(scores, 0.99)),
                    "target_score_q99": float(np.quantile(tgt, 0.99)) if len(tgt) else 0.0,
                    "score_iqr": float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25)),
                    "score_skewness": float(np.mean((scores - scores.mean()) ** 3) / max(scores.std() ** 3, 1e-12)),
                    "score_kurtosis": float(np.mean((scores - scores.mean()) ** 4) / max(scores.std() ** 4, 1e-12)),
                    "train_score_mean": float(np.mean(train_scores)) if train_scores is not None and len(train_scores) else 0.0,
                    "train_score_std": float(np.std(train_scores)) if train_scores is not None and len(train_scores) else 0.0,
                    "train_score_q95": float(np.quantile(train_scores, 0.95)) if train_scores is not None and len(train_scores) else 0.0,
                    "train_score_q99": float(np.quantile(train_scores, 0.99)) if train_scores is not None and len(train_scores) else 0.0,
                    "train_test_score_gap": float(np.mean(scores) - np.mean(train_scores)) if train_scores is not None and len(train_scores) else 0.0,
                    "rank_corr_v63": safe_spearman(scores, ref_v63),
                    "rank_corr_a0": safe_spearman(scores, ref_a0) if ref_a0 is not None else 0.0,
                    "top_normal_tail_overlap_v63": top_tail_overlap(scores, ref_v63),
                    "score_entropy": score_entropy(scores),
                    "score_dispersion": float(np.std(scores)),
                }
                feature_rows.append(feat)

    return bundles, feature_rows, alignment_rows, missing, alignment_issue


def normalize_feature(rows: list[dict], key: str):
    vals = np.asarray([r[key] for r in rows], dtype=np.float64)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-12:
        return {id(r): 0.0 for r in rows}
    return {id(r): float((r[key] - vmin) / (vmax - vmin)) for r in rows}


def selector_pick(selector: str, rows: list[dict]) -> dict:
    # rows = candidates for one seed/machine
    if selector == "S0":
        return next(r for r in rows if r["candidate"] == "A1_alpha_0.50")
    if selector == "S1":
        return min(rows, key=lambda r: r["source_target_ks"])
    if selector == "S2":
        return min(rows, key=lambda r: r["target_score_q99"])
    if selector == "S3":
        n_ks = normalize_feature(rows, "source_target_ks")
        n_q99 = normalize_feature(rows, "target_score_q99")
        n_std = normalize_feature(rows, "score_dispersion")
        for r in rows:
            r["_stability"] = n_ks[id(r)] + n_q99[id(r)] + n_std[id(r)]
        return min(rows, key=lambda r: (r["_stability"], r["candidate"]))
    if selector == "S4":
        n_ks = normalize_feature(rows, "source_target_ks")
        n_q99 = normalize_feature(rows, "target_score_q99")
        for r in rows:
            r["_risk"] = n_ks[id(r)] + n_q99[id(r)]
        beats_rows = [r for r in rows if r["candidate"].startswith("A")]
        ctf_rows = [r for r in rows if not r["candidate"].startswith("A")]
        best_beats = min(beats_rows, key=lambda r: (r["_risk"], r["candidate"]))
        best_ctf = min(ctf_rows, key=lambda r: (r["_risk"], r["candidate"]))
        # fixed conservative rule:
        # if BEATs route has risk <= 0.9 choose it, else fallback to CTAF route
        return best_beats if best_beats["_risk"] <= 0.9 else best_ctf
    if selector == "S5":
        n_ks = normalize_feature(rows, "source_target_ks")
        n_q99 = normalize_feature(rows, "target_score_q99")
        n_std = normalize_feature(rows, "score_dispersion")
        scored = []
        for r in rows:
            stability = n_ks[id(r)] + n_q99[id(r)] + n_std[id(r)]
            scored.append((stability, r))
        scored.sort(key=lambda x: (x[0], x[1]["candidate"]))
        top3 = scored[:3]
        # median rank among top-3 stability candidates => second entry
        return top3[1][1] if len(top3) >= 2 else top3[0][1]
    raise ValueError(f"unknown selector: {selector}")


def run_feature_audit(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bundles, feature_rows, alignment_rows, missing, alignment_issue = gather_candidates(args)
    write_csv(out_root / "feature_table.csv", feature_rows)
    write_csv(out_root / "missing_candidates.csv", missing)
    write_csv(out_root / "alignment_summary.csv", alignment_rows)
    Path(args.doc_path).write_text(
        "# CTAF Stage15B Label-Free Route Selector Audit\n\n"
        "Mode=feature_audit completed. No selector decision was executed.\n"
    )


def run_selector_audit(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bundles, feature_rows, alignment_rows, missing, alignment_issue = gather_candidates(args)
    write_csv(out_root / "feature_table.csv", feature_rows)
    write_csv(out_root / "missing_candidates.csv", missing)
    write_csv(out_root / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        Path(args.doc_path).write_text(
            "# CTAF Stage15B Label-Free Route Selector Audit\n\n"
            "Alignment issue detected. Selector audit stopped.\n"
        )
        return

    # load Stage15A oracle ceiling
    stage15a_root = Path(args.stage15a_root)
    oracle_no_diag = list(csv.DictReader((stage15a_root / "oracle_no_diagnostic.csv").open()))
    oracle_no_diag_overall = next(r for r in oracle_no_diag if r["machine"] == "OVERALL")
    oracle_hmean = float(oracle_no_diag_overall["mean_hmean"])

    # build metric lookup
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    feature_by_key = {(r["seed"], r["machine"], r["candidate"]): r for r in feature_rows}
    metric_rows = []
    for (cand, seed, machine), bundle in bundles.items():
        # normalize seed for non-seeded candidates: only first seed is present
        pass
    # compute metrics for all routes in selector pool
    routes = []
    for key, bundle in bundles.items():
        cand, seed, machine = key
        y = bundle["y_test"]
        d = bundle["domains"]
        ev = evaluate(bundle["test_scores"], y, d)
        routes.append({
            "seed": seed,
            "machine": machine,
            "candidate": cand,
            **ev,
        })
    metric_by_key = {(r["seed"], r["machine"], r["candidate"]): r for r in routes}

    selectors = ["S0", "S1", "S2", "S3", "S4", "S5"]
    selected_rows = []
    selector_result_rows = []
    selector_per_machine_rows = []
    selector_stability_rows = []

    for selector in selectors:
        per_seed_machine = []
        for seed in seeds:
            for machine in machines:
                rows = []
                for cand in ["V37","V61","V63","A0_raw_BEATs","A1_alpha_0.50","A3_alpha_0.20","A3_alpha_0.35","A3_alpha_0.65","A3_alpha_0.80","A4_B2s","A4_F2s"]:
                    key = (seed, machine, cand)
                    if key in feature_by_key and key in metric_by_key:
                        merged = dict(feature_by_key[key])
                        merged.update(metric_by_key[key])
                        rows.append(merged)
                    elif cand in ("A0_raw_BEATs","A4_B2s"):
                        shared_key = (seeds[0], machine, cand)
                        if shared_key in feature_by_key and shared_key in metric_by_key:
                            merged = dict(feature_by_key[shared_key])
                            merged.update(metric_by_key[shared_key])
                            merged["seed"] = seed
                            rows.append(merged)
                chosen = selector_pick(selector, rows)
                per_seed_machine.append({
                    "selector": selector,
                    "seed": seed,
                    "machine": machine,
                    "selected_candidate": chosen["candidate"],
                    "hmean": chosen["hmean"],
                    "pAUC": chosen["pAUC"],
                    "TPR@95n": chosen["TPR@95n"],
                    "TPR@99n": chosen["TPR@99n"],
                    "KS": chosen["KS"],
                })
        selected_rows.extend(per_seed_machine)

        # overall selector metrics per seed then aggregate
        by_seed = defaultdict(list)
        by_machine = defaultdict(list)
        for row in per_seed_machine:
            by_seed[row["seed"]].append(row)
            by_machine[row["machine"]].append(row)
        seed_overall = []
        for seed, rows in sorted(by_seed.items()):
            seed_overall.append({
                "selector": selector,
                "seed": seed,
                "hmean": float(np.mean([r["hmean"] for r in rows])),
                "pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
                "KS": float(np.mean([r["KS"] for r in rows])),
            })
        selector_result_rows.append({
            "selector": selector,
            "mean_hmean": float(np.mean([r["hmean"] for r in seed_overall])),
            "std_hmean": float(np.std([r["hmean"] for r in seed_overall], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in seed_overall])),
            "std_pAUC": float(np.std([r["pAUC"] for r in seed_overall], ddof=0)),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in seed_overall])),
            "std_TPR@99n": float(np.std([r["TPR@99n"] for r in seed_overall], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in seed_overall])),
            "reaches_0_6": float(np.mean([r["hmean"] for r in seed_overall])) >= 0.6,
        })
        for machine, rows in sorted(by_machine.items()):
            selector_per_machine_rows.append({
                "selector": selector,
                "machine": machine,
                "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
                "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
                "mean_KS": float(np.mean([r["KS"] for r in rows])),
            })
        for machine, rows in sorted(by_machine.items()):
            counts = Counter(r["selected_candidate"] for r in rows)
            majority, count = counts.most_common(1)[0]
            selector_stability_rows.append({
                "selector": selector,
                "machine": machine,
                "seed_13711": next(r["selected_candidate"] for r in rows if r["seed"] == "13711"),
                "seed_13721": next(r["selected_candidate"] for r in rows if r["seed"] == "13721"),
                "seed_13731": next(r["selected_candidate"] for r in rows if r["seed"] == "13731"),
                "unique_selected_candidates": len(counts),
                "majority_candidate": majority,
                "majority_count": count,
                "fully_stable": len(counts) == 1,
            })

    write_csv(out_root / "selector_results.csv", selector_result_rows)
    write_csv(out_root / "selector_per_machine.csv", selector_per_machine_rows)
    write_csv(out_root / "selected_routes_by_machine.csv", selected_rows)
    write_csv(out_root / "selector_seed_stability.csv", selector_stability_rows)

    oracle_gap_rows = []
    for row in selector_result_rows:
        oracle_gap_rows.append({
            "selector": row["selector"],
            "mean_hmean": row["mean_hmean"],
            "gap_to_no_diagnostic_oracle": row["mean_hmean"] - oracle_hmean,
            "mean_pAUC": row["mean_pAUC"],
            "mean_TPR@99n": row["mean_TPR@99n"],
            "reaches_0_6": row["reaches_0_6"],
        })
    write_csv(out_root / "selector_vs_oracle_gap.csv", oracle_gap_rows)

    best_selector = max(selector_result_rows, key=lambda r: r["mean_hmean"])
    if best_selector["mean_hmean"] >= 0.6:
        interpretation = "Stage 15B positive: at least one label-free selector reaches hmean >= 0.6; next step should be to freeze that selector in a preregistered route"
    elif best_selector["mean_hmean"] >= 0.595:
        interpretation = "selector promising but not enough: it approaches 0.6 but does not clearly reach it; next step should be to improve the label-free rule or add a new external representation"
    else:
        interpretation = "current unlabeled statistics are insufficient; selectors stay near A1 and do not reach 0.6; next step should be new external pretrained representations"

    lines = [
        "# CTAF Stage15B Label-Free Route Selector Audit",
        "",
        "## 1. Stage 15A Ceiling Recap",
        "",
        f"- no-diagnostic oracle hmean ≈ `{oracle_hmean:.4f}`",
        "- current best frozen route A1 hmean ≈ `0.5813`",
        "",
        "## 2. Label-Free Selector Results",
        "",
        "| selector | mean hmean | std hmean | mean pAUC | std pAUC | mean TPR@99n | std TPR@99n | mean KS | reaches 0.6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in selector_result_rows:
        lines.append(
            f"| {r['selector']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | {r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | "
            f"{r['mean_TPR@99n']:.6f} | {r['std_TPR@99n']:.6f} | {r['mean_KS']:.6f} | {r['reaches_0_6']} |"
        )
    lines.extend([
        "",
        "## 3. Stability",
        "",
        "| selector | machine | seed_13711 | seed_13721 | seed_13731 | fully stable |",
        "|---|---|---|---|---|---|",
    ])
    for r in selector_stability_rows:
        lines.append(
            f"| {r['selector']} | {r['machine']} | {r['seed_13711']} | {r['seed_13721']} | {r['seed_13731']} | {r['fully_stable']} |"
        )
    lines.extend([
        "",
        "## 4. Interpretation",
        "",
        interpretation,
        "",
        "## 5. Warnings",
        "",
        "- oracle results are ceiling only",
        "- selector must not use hmean/pAUC/TPR@99n for route choice",
        "- anomaly labels are not used for selection",
        "- per-machine oracle is not a final method",
        "- no training occurred",
    ])
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "feature_audit":
        run_feature_audit(args)
        return
    if args.mode == "selector_audit":
        run_selector_audit(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
