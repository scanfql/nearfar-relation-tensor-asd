#!/usr/bin/env python
"""Stage 15A: existing-route hmean ceiling audit (oracle analysis only)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 15A existing-route hmean ceiling audit")
    p.add_argument("--mode", required=True, choices=["check", "ceiling_audit"])
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
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


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def candidate_specs(args):
    return {
        "V37": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_latent_summary_knn_scores.npz"},
        "V61": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_latent_summary_knn_scores.npz"},
        "V63": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{m}_full_5view_rankfusion_scores.npz"},
        "A0_raw_BEATs": {"seeded": False, "diagnostic": False,
                "path": lambda seed, m: Path(args.stage14a_root) / "a0_frozen_beats" / "scores" / f"{m}_frozen_beats_knn_scores.npz"},
        "A1_alpha_0.50": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.stage14a_root) / "a1_v63_beats_fusion" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_rankfusion_scores.npz"},
        "A3_alpha_0.20": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.20" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.20_scores.npz"},
        "A3_alpha_0.35": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.35" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.35_scores.npz"},
        "A3_alpha_0.65": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.65" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.65_scores.npz"},
        "A3_alpha_0.80": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.a3_root) / "a3_weight_sensitivity" / "alpha_0.80" / f"seed_{seed}" / "scores" / f"{m}_v63_beats_alpha_0.80_scores.npz"},
        "A4_B2s": {"seeded": False, "diagnostic": False,
                "path": lambda seed, m: Path(args.a4_root) / "b2_cross_year_beats" / "scores" / f"{m}_b2_source_beats_scores.npz"},
        "A4_B2f": {"seeded": False, "diagnostic": True,
                "path": lambda seed, m: Path(args.a4_root) / "b2_cross_year_beats" / "scores" / f"{m}_b2_full_beats_scores.npz"},
        "A4_F2s": {"seeded": True, "diagnostic": False,
                "path": lambda seed, m: Path(args.a4_root) / "f2_v63_cross_year_beats_fusion" / f"seed_{seed}" / "scores" / f"{m}_f2s_scores.npz"},
        "A4_F2f": {"seeded": True, "diagnostic": True,
                "path": lambda seed, m: Path(args.a4_root) / "f2_v63_cross_year_beats_fusion" / f"seed_{seed}" / "scores" / f"{m}_f2f_scores.npz"},
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    specs = candidate_specs(args)
    missing = []
    avail_counter = Counter()
    for cand, meta in specs.items():
        for seed in seeds:
            for machine in machines:
                path = meta["path"](seed, machine)
                if meta["seeded"]:
                    exists = path.exists()
                else:
                    if seed != seeds[0]:
                        continue
                    exists = path.exists()
                if exists:
                    avail_counter[cand] += 1
                else:
                    missing.append({"candidate": cand, "seed": seed if meta["seeded"] else "shared", "machine": machine, "path": str(path)})
    write_csv(out_root / "missing_candidates.csv", missing)
    status = {
        "mode": "check",
        "data_root_exists": Path(args.data_root).exists(),
        "v37_root_exists": Path(args.v37_root).exists(),
        "v61_root_exists": Path(args.v61_root).exists(),
        "v63_root_exists": Path(args.v63_root).exists(),
        "stage14a_root_exists": Path(args.stage14a_root).exists(),
        "a3_root_exists": Path(args.a3_root).exists(),
        "a4_root_exists": Path(args.a4_root).exists(),
        "candidate_availability": dict(avail_counter),
    }
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")
    Path(args.doc_path).write_text(
        "# CTAF Stage15A Existing Route Hmean Ceiling Audit\n\n"
        "Mode=check completed. No metrics or oracle analysis computed.\n"
    )


def run_ceiling_audit(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    specs = candidate_specs(args)
    missing = []
    alignment_rows = []
    metric_rows = []
    alignment_issue = False

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        for cand, meta in specs.items():
            for seed in seeds:
                if not meta["seeded"] and seed != seeds[0]:
                    continue
                path = meta["path"](seed, machine)
                if not path.exists():
                    missing.append({"candidate": cand, "seed": seed if meta["seeded"] else "shared", "machine": machine, "path": str(path)})
                    continue
                bundle = load_bundle(path)
                y_match = np.array_equal(bundle["y_test"], expected["y_test"])
                d_match = np.array_equal(bundle["domains"], expected["domains"])
                if bundle["test_paths"] is not None:
                    order_match = np.array_equal(bundle["test_paths"], expected["paths"])
                else:
                    order_match = y_match and d_match
                alignment_rows.append({
                    "candidate": cand,
                    "seed": seed if meta["seeded"] else "shared",
                    "machine": machine,
                    "y_test_match": y_match,
                    "domains_match": d_match,
                    "sample_order_match": order_match,
                    "alignment_ok": bool(y_match and d_match and order_match),
                    "path": str(path),
                })
                if not (y_match and d_match and order_match):
                    alignment_issue = True
                    continue

                ev = evaluate(bundle["test_scores"], bundle["y_test"], bundle["domains"])
                metric_rows.append({
                    "candidate": cand,
                    "seed": seed,
                    "machine": machine,
                    "diagnostic": bool(meta["diagnostic"]),
                    **ev,
                })

    write_csv(out_root / "missing_candidates.csv", missing)
    write_csv(out_root / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        Path(args.doc_path).write_text(
            "# CTAF Stage15A Existing Route Hmean Ceiling Audit\n\n"
            "Alignment issue detected. Ceiling analysis stopped.\n"
        )
        return

    write_csv(out_root / "candidate_metrics_raw.csv", metric_rows)

    # candidate_summary
    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    for row in metric_rows:
        by_seed_cand[(row["candidate"], row["seed"])].append(row)
        by_machine_cand[(row["candidate"], row["machine"])].append(row)

    candidate_summary = []
    per_machine_metrics = []
    for (cand, seed), rows in sorted(by_seed_cand.items()):
        candidate_summary.append({
            "candidate": cand,
            "seed": seed,
            "hmean": float(np.mean([r["hmean"] for r in rows])),
            "pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "KS": float(np.mean([r["KS"] for r in rows])),
        })
    write_csv(out_root / "candidate_summary.csv", candidate_summary)

    for (cand, machine), rows in sorted(by_machine_cand.items()):
        per_machine_metrics.append({
            "candidate": cand,
            "machine": machine,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "std_TPR@99n": float(np.std([r["TPR@99n"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
        })
    write_csv(out_root / "per_machine_candidate_metrics.csv", per_machine_metrics)

    # Aggregate per candidate across seeds
    agg_rows = []
    by_candidate = defaultdict(list)
    for row in candidate_summary:
        by_candidate[row["candidate"]].append(row)
    for cand, rows in sorted(by_candidate.items()):
        diagnostic = specs[cand]["diagnostic"]
        agg_rows.append({
            "candidate": cand,
            "diagnostic": diagnostic,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
        })

    # Oracle A: global best single candidate
    global_best = max(agg_rows, key=lambda r: r["mean_hmean"])
    write_csv(out_root / "oracle_global_best.csv", [global_best])

    pm = {(r["candidate"], r["machine"]): r for r in per_machine_metrics}
    v63_pm = {r["machine"]: r for r in per_machine_metrics if r["candidate"] == "V63"}
    diagnostic_candidates = {c for c, meta in specs.items() if meta["diagnostic"]}

    def build_oracle(name: str, allow_fn):
        rows = []
        selected = []
        for machine in machines:
            candidates = [r for r in per_machine_metrics if r["machine"] == machine and allow_fn(r)]
            best = max(candidates, key=lambda r: r["mean_hmean"])
            rows.append({
                "oracle": name,
                "machine": machine,
                "selected_candidate": best["candidate"],
                "mean_hmean": best["mean_hmean"],
                "mean_pAUC": best["mean_pAUC"],
                "mean_TPR@99n": best["mean_TPR@99n"],
                "mean_KS": best["mean_KS"],
            })
            selected.append(best)
        overall = {
            "oracle": name,
            "machine": "OVERALL",
            "selected_candidate": "per_machine_oracle",
            "mean_hmean": float(np.mean([r["mean_hmean"] for r in selected])),
            "mean_pAUC": float(np.mean([r["mean_pAUC"] for r in selected])),
            "mean_TPR@99n": float(np.mean([r["mean_TPR@99n"] for r in selected])),
            "mean_KS": float(np.mean([r["mean_KS"] for r in selected])),
        }
        rows.append(overall)
        return rows

    oracle_per_machine = build_oracle("per_machine_oracle", lambda r: True)
    write_csv(out_root / "oracle_per_machine_best.csv", oracle_per_machine)

    oracle_constrained = build_oracle(
        "constrained_oracle",
        lambda r: r["mean_pAUC"] >= (v63_pm[r["machine"]]["mean_pAUC"] - 0.01),
    )
    write_csv(out_root / "oracle_constrained.csv", oracle_constrained)

    oracle_no_diag = build_oracle(
        "no_diagnostic_oracle",
        lambda r: r["candidate"] not in diagnostic_candidates,
    )
    write_csv(out_root / "oracle_no_diagnostic.csv", oracle_no_diag)

    # Seed stability for per-machine oracle and no-diagnostic oracle
    seed_stability_rows = []
    by_seed_machine = defaultdict(list)
    for row in metric_rows:
        by_seed_machine[(row["seed"], row["machine"])].append(row)
    for oracle_name, allow_fn in [
        ("per_machine_oracle", lambda r: True),
        ("constrained_oracle", lambda r: r["pAUC"] >= (v63_pm[r["machine"]]["mean_pAUC"] - 0.01)),
        ("no_diagnostic_oracle", lambda r: r["candidate"] not in diagnostic_candidates),
    ]:
        for machine in machines:
            chosen = {}
            for seed in seeds:
                rows = [r for r in by_seed_machine[(seed, machine)] if allow_fn(r)]
                best = max(rows, key=lambda r: r["hmean"])
                chosen[seed] = best["candidate"]
            counts = Counter(chosen.values())
            majority_candidate, majority_count = counts.most_common(1)[0]
            seed_stability_rows.append({
                "oracle": oracle_name,
                "machine": machine,
                "seed_13711": chosen.get("13711", ""),
                "seed_13721": chosen.get("13721", ""),
                "seed_13731": chosen.get("13731", ""),
                "unique_selected_candidates": len(counts),
                "majority_candidate": majority_candidate,
                "majority_count": majority_count,
                "fully_stable": len(counts) == 1,
            })
    write_csv(out_root / "seed_stability.csv", seed_stability_rows)

    # Doc interpretation
    per_machine_overall = next(r for r in oracle_per_machine if r["machine"] == "OVERALL")
    no_diag_overall = next(r for r in oracle_no_diag if r["machine"] == "OVERALL")
    constrained_overall = next(r for r in oracle_constrained if r["machine"] == "OVERALL")
    if per_machine_overall["mean_hmean"] < 0.6:
        interpretation = "current score pool cannot reach hmean target; next step should be new external pretrained representations"
    elif no_diag_overall["mean_hmean"] < 0.6:
        interpretation = "only diagnostic candidates push oracle toward 0.6; target-domain normal memory is essential, not purely source-only generalized success"
    else:
        interpretation = "existing scores contain enough complementary signal; next step should be label-free selector/router, not a new model"

    lines = [
        "# CTAF Stage15A Existing Route Hmean Ceiling Audit",
        "",
        "## 1. Scope",
        "",
        "Oracle ceiling analysis only.",
        "This is not a final method and must not be used as a paper main result.",
        "",
        "## 2. Candidate Pool",
        "",
        "- V37",
        "- V61",
        "- V63",
        "- A0 raw BEATs",
        "- A1 alpha=0.5",
        "- A3 alpha=0.2 / 0.35 / 0.65 / 0.8",
        "- A4 B2s / B2f / F2s / F2f",
        "",
        "## 3. Alignment Check",
        "",
        f"- checked seeds: `{','.join(seeds)}`",
        f"- checked machines: `{','.join(machines)}`",
        f"- alignment rows: `{len(alignment_rows)}`",
        f"- missing candidates: `{len(missing)}`",
        "- oracle is ceiling-only and must not replace frozen routes",
        "",
        "## 4. Global Best Single Candidate",
        "",
        f"- candidate: `{global_best['candidate']}`",
        f"- mean hmean: `{global_best['mean_hmean']:.6f}`",
        f"- mean pAUC: `{global_best['mean_pAUC']:.6f}`",
        f"- mean TPR@99n: `{global_best['mean_TPR@99n']:.6f}`",
        "",
        "## 5. Oracle Summaries",
        "",
        f"- per-machine oracle mean hmean: `{per_machine_overall['mean_hmean']:.6f}`",
        f"- constrained oracle mean hmean: `{constrained_overall['mean_hmean']:.6f}`",
        f"- no-diagnostic oracle mean hmean: `{no_diag_overall['mean_hmean']:.6f}`",
        "",
        "## 6. Interpretation",
        "",
        interpretation,
        "",
        "## 7. Warning",
        "",
        "- oracle uses labels/metrics for ceiling analysis only",
        "- oracle is not a final method",
        "- do not replace A1/V63 based on oracle",
        "- no training occurred",
    ]
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "ceiling_audit":
        run_ceiling_audit(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
