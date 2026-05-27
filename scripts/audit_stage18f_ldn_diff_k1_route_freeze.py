#!/usr/bin/env python
"""Stage 18F: LDN-Diff-K1 route freeze / confirmation / paper-ready audit.

This script is strictly read-only on Stage18E result artifacts.

It does NOT:
- recompute embeddings
- load EAT
- load any pretrained model
- run model forward
- train
- score from embeddings
- tune any hyperparameters
- add any candidates

It only:
- verifies frozen-route conditions from existing Stage18E outputs
- checks score-file alignment for baseline and frozen route
- reformats results into paper-ready tables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STAGE18E_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
OUTPUT_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18F_LDN_DIFF_K1_ROUTE_FREEZE_AUDIT.md"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"

FROZEN_ROUTE = "ldn_diff_K1_scorek5"
REFERENCE_BASELINE = "baseline_knn_k5"
REFERENCE_ROWS = [
    "baseline_knn_k5",
    "ldn_ratio_K1_scorek5",
    "ldn_diff_K1_scorek5",
    "ldn_ratio_K5_scorek5",
    "ldn_diff_K5_scorek5",
    "gwrp_ratio_r09_scorek5",
    "gwrp_diff_r09_scorek5",
    "rasdsr_ratio_avg_K5_r09_scorek5",
]

REGISTERED_BASELINE_MEAN_HMEAN = 0.604713
REGISTERED_FROZEN_MEAN_HMEAN = 0.615947
REGISTERED_OLD_BEST_MEAN_HMEAN = 0.607175
TOL = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Stage18F LDN-Diff-K1 route freeze audit")
    p.add_argument("--mode", required=True, choices=["check", "freeze_audit"])
    p.add_argument("--stage18e_root", default=STAGE18E_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    return p.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def import_status():
    status = {"csv_builtin": True}
    for name in ["numpy", "pandas", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    return json.loads(path.read_text())


def require_columns(path: Path, rows: list[dict], required: list[str]) -> list[str]:
    if not rows:
        return required[:]
    header = set(rows[0].keys())
    return [c for c in required if c not in header]


def score_path(stage18e_root: Path, candidate: str, machine: str) -> Path:
    return stage18e_root / "scores" / candidate / f"{machine}_scores.npz"


def load_score_bundle(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "d_test": arr["d_test"].astype(str) if "d_test" in arr.files else arr["domains"].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        "candidate": str(arr["candidate"]) if "candidate" in arr.files else "",
        "machine": str(arr["machine"]) if "machine" in arr.files else "",
    }


def get_row(rows: list[dict], **kwargs) -> dict:
    for row in rows:
        ok = True
        for k, v in kwargs.items():
            if row.get(k) != v:
                ok = False
                break
        if ok:
            return row
    raise KeyError(f"Missing row with filters: {kwargs}")


def approx_equal(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage18F LDN-Diff-K1 Route Freeze Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no metric computation",
        "- no score recomputation",
        "- no model load",
        "- no forward",
        "- no embedding extraction",
        "",
        "## Status",
        "",
        f"- Stage18E root exists: `{status['stage18e_root_exists']}`",
        f"- all required files present: `{status['all_required_files_present']}`",
        f"- frozen candidate present in required files: `{status['frozen_candidate_present_everywhere']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_freeze_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage18F LDN-Diff-K1 Route Freeze Audit",
        "",
        "## Frozen Route",
        "",
        f"- route: `{verdict['frozen_route_name']}`",
        f"- freeze accepted: `{verdict['freeze_accepted']}`",
        f"- mean hmean: `{verdict['mean_hmean']:.6f}`",
        f"- mean pAUC: `{verdict['mean_pAUC']:.6f}`",
        f"- mean TPR@99n: `{verdict['mean_TPR99']:.6f}`",
        f"- delta hmean vs baseline: `{verdict['delta_hmean_vs_baseline']:.6f}`",
        f"- delta hmean vs old best: `{verdict['delta_hmean_vs_old_best']:.6f}`",
        "",
        "## Guardrails",
        "",
        f"- improved machine count vs baseline: `{verdict['improved_machine_count_vs_baseline']}`",
        f"- strong-machine severe collapse: `{verdict['strong_machine_severe_collapse']}`",
    ]


def stage18e_paths(root: Path) -> dict[str, Path]:
    return {
        "check_status_json": root / "check_status.json",
        "embedding_check_csv": root / "embedding_check.csv",
        "summary_csv": root / "summary.csv",
        "candidate_rank_csv": root / "candidate_rank.csv",
        "per_machine_csv": root / "per_machine.csv",
        "tail_table_csv": root / "tail_table.csv",
        "score_geometry_csv": root / "score_geometry.csv",
        "comparison_vs_baseline_csv": root / "comparison_vs_baseline.csv",
        "collapse_check_csv": root / "collapse_check.csv",
        "verdict_json": root / "stage18e_ldn_verdict.json",
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    root = Path(args.stage18e_root)
    machines = parse_csv_list(args.machines)
    paths = stage18e_paths(root)

    required_files_rows = []
    all_present = True
    for label, path in paths.items():
        exists = path.exists()
        all_present = all_present and exists
        required_files_rows.append({"artifact": label, "path": str(path), "exists": exists})

    score_rows = []
    for candidate in [REFERENCE_BASELINE, FROZEN_ROUTE]:
        for machine in machines:
            path = score_path(root, candidate, machine)
            exists = path.exists()
            all_present = all_present and exists
            score_rows.append({
                "artifact": f"score_{candidate}_{machine}",
                "path": str(path),
                "exists": exists,
            })

    missing_columns_rows = []
    frozen_present = True
    if paths["summary_csv"].exists():
        summary_rows = load_csv_rows(paths["summary_csv"])
        missing = require_columns(paths["summary_csv"], summary_rows, ["candidate", "mean_hmean", "mean_pAUC", "mean_TPR@99n"])
        missing_columns_rows.append({"file": str(paths["summary_csv"]), "missing_columns": ",".join(missing)})
        frozen_present = frozen_present and any(r["candidate"] == FROZEN_ROUTE for r in summary_rows)
    if paths["candidate_rank_csv"].exists():
        rank_rows = load_csv_rows(paths["candidate_rank_csv"])
        missing = require_columns(paths["candidate_rank_csv"], rank_rows, ["rank", "candidate", "mean_hmean", "mean_pAUC", "mean_TPR@99n"])
        missing_columns_rows.append({"file": str(paths["candidate_rank_csv"]), "missing_columns": ",".join(missing)})
        frozen_present = frozen_present and any(r["candidate"] == FROZEN_ROUTE for r in rank_rows)
    if paths["per_machine_csv"].exists():
        pm_rows = load_csv_rows(paths["per_machine_csv"])
        missing = require_columns(paths["per_machine_csv"], pm_rows, ["candidate", "machine", "hmean", "pAUC", "TPR@99n"])
        missing_columns_rows.append({"file": str(paths["per_machine_csv"]), "missing_columns": ",".join(missing)})
        frozen_present = frozen_present and any(r["candidate"] == FROZEN_ROUTE for r in pm_rows)
    if paths["comparison_vs_baseline_csv"].exists():
        comp_rows = load_csv_rows(paths["comparison_vs_baseline_csv"])
        missing = require_columns(paths["comparison_vs_baseline_csv"], comp_rows, ["candidate", "mean_hmean", "delta_hmean_vs_baseline", "improved_machine_count_vs_baseline", "strong_machine_severe_collapse"])
        missing_columns_rows.append({"file": str(paths["comparison_vs_baseline_csv"]), "missing_columns": ",".join(missing)})
        frozen_present = frozen_present and any(r["candidate"] == FROZEN_ROUTE for r in comp_rows)
    if paths["verdict_json"].exists():
        verdict = load_json(paths["verdict_json"])
        frozen_present = frozen_present and verdict.get("best_candidate") == FROZEN_ROUTE

    status = {
        "mode": "check",
        "stage18e_root": str(root),
        "stage18e_root_exists": root.exists(),
        "machines": machines,
        "all_required_files_present": all_present,
        "frozen_candidate_present_everywhere": frozen_present,
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "required_files_check.csv", required_files_rows + score_rows)
    write_csv(out_root / "required_columns_check.csv", missing_columns_rows)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_freeze_audit(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    root = Path(args.stage18e_root)
    machines = parse_csv_list(args.machines)
    paths = stage18e_paths(root)

    summary_rows = load_csv_rows(paths["summary_csv"])
    rank_rows = load_csv_rows(paths["candidate_rank_csv"])
    pm_rows = load_csv_rows(paths["per_machine_csv"])
    tail_rows = load_csv_rows(paths["tail_table_csv"])
    geom_rows = load_csv_rows(paths["score_geometry_csv"])
    comp_rows = load_csv_rows(paths["comparison_vs_baseline_csv"])
    collapse_rows = load_csv_rows(paths["collapse_check_csv"])
    verdict = load_json(paths["verdict_json"])

    summary_frozen = get_row(summary_rows, candidate=FROZEN_ROUTE)
    rank_frozen = get_row(rank_rows, candidate=FROZEN_ROUTE)
    comp_frozen = get_row(comp_rows, candidate=FROZEN_ROUTE)

    checks = {
        "best_candidate_matches": verdict.get("best_candidate") == FROZEN_ROUTE,
        "rank_is_1": int(rank_frozen["rank"]) == 1,
        "baseline_mean_hmean_matches": approx_equal(get_row(summary_rows, candidate=REFERENCE_BASELINE)["mean_hmean"], REGISTERED_BASELINE_MEAN_HMEAN),
        "frozen_mean_hmean_matches": approx_equal(summary_frozen["mean_hmean"], REGISTERED_FROZEN_MEAN_HMEAN),
        "frozen_exceeds_old_best": float(summary_frozen["mean_hmean"]) > REGISTERED_OLD_BEST_MEAN_HMEAN,
        "strong_machine_severe_collapse_false": str(comp_frozen["strong_machine_severe_collapse"]).lower() == "false",
        "improved_machine_count_ge_4": int(comp_frozen["improved_machine_count_vs_baseline"]) >= 4,
        "no_selector_no_fusion_extra_candidate": set(r["candidate"] for r in rank_rows) == set(REFERENCE_ROWS),
    }

    score_file_rows = []
    for machine in machines:
        base_bundle = load_score_bundle(score_path(root, REFERENCE_BASELINE, machine))
        frozen_bundle = load_score_bundle(score_path(root, FROZEN_ROUTE, machine))
        y_match = np.array_equal(base_bundle["y_test"], frozen_bundle["y_test"])
        d_match = np.array_equal(base_bundle["d_test"], frozen_bundle["d_test"])
        if base_bundle["test_paths"] is not None and frozen_bundle["test_paths"] is not None:
            p_match = np.array_equal(base_bundle["test_paths"], frozen_bundle["test_paths"])
        else:
            p_match = bool(y_match and d_match)
        score_file_rows.append({
            "machine": machine,
            "baseline_score_path": str(score_path(root, REFERENCE_BASELINE, machine)),
            "frozen_score_path": str(score_path(root, FROZEN_ROUTE, machine)),
            "y_test_match": bool(y_match),
            "d_test_match": bool(d_match),
            "test_paths_match": bool(p_match),
            "alignment_ok": bool(y_match and d_match and p_match),
        })
    write_csv(out_root / "score_file_alignment_check.csv", score_file_rows)

    baseline_pm = {r["machine"]: r for r in pm_rows if r["candidate"] == REFERENCE_BASELINE}
    frozen_pm = {r["machine"]: r for r in pm_rows if r["candidate"] == FROZEN_ROUTE}
    baseline_tail = {r["machine"]: r for r in tail_rows if r["candidate"] == REFERENCE_BASELINE}
    frozen_tail = {r["machine"]: r for r in tail_rows if r["candidate"] == FROZEN_ROUTE}

    comments = {
        "fan": "fan decreases and must be explicitly flagged",
        "valve": "valve improves strongly",
        "gearbox": "gearbox improves",
        "slider": "slider improves",
        "toycar": "toycar improves slightly",
        "bearing": "bearing is nearly stable",
        "toytrain": "toytrain is nearly stable",
    }

    per_machine_delta_rows = []
    tail_delta_rows = []
    strong_guardrail_rows = []
    improved_count = 0
    for machine in machines:
        b = baseline_pm[machine]
        f = frozen_pm[machine]
        bt = baseline_tail[machine]
        ft = frozen_tail[machine]
        delta_h = float(f["hmean"]) - float(b["hmean"])
        delta_p = float(f["pAUC"]) - float(b["pAUC"])
        delta_t99 = float(f["TPR@99n"]) - float(b["TPR@99n"])
        if delta_h > 0:
            improved_count += 1
        per_machine_delta_rows.append({
            "machine": machine,
            "baseline_hmean": b["hmean"],
            "ldn_diff_K1_hmean": f["hmean"],
            "delta_hmean": delta_h,
            "baseline_pAUC": b["pAUC"],
            "ldn_diff_K1_pAUC": f["pAUC"],
            "delta_pAUC": delta_p,
            "baseline_TPR@99n": b["TPR@99n"],
            "ldn_diff_K1_TPR@99n": f["TPR@99n"],
            "delta_TPR@99n": delta_t99,
            "comment": comments.get(machine, ""),
        })
        tail_delta_rows.append({
            "machine": machine,
            "baseline_q95": bt["q95"],
            "frozen_q95": ft["q95"],
            "delta_q95": float(ft["q95"]) - float(bt["q95"]),
            "baseline_q99": bt["q99"],
            "frozen_q99": ft["q99"],
            "delta_q99": float(ft["q99"]) - float(bt["q99"]),
            "baseline_anomaly_median": bt["anomaly_median"],
            "frozen_anomaly_median": ft["anomaly_median"],
            "delta_anomaly_median": float(ft["anomaly_median"]) - float(bt["anomaly_median"]),
            "baseline_anomaly_q75": bt["anomaly_q75"],
            "frozen_anomaly_q75": ft["anomaly_q75"],
            "delta_anomaly_q75": float(ft["anomaly_q75"]) - float(bt["anomaly_q75"]),
        })
        if machine in {"bearing", "toytrain", "valve"}:
            severe = delta_h < -0.02
            strong_guardrail_rows.append({
                "machine": machine,
                "baseline_hmean": b["hmean"],
                "frozen_hmean": f["hmean"],
                "delta": delta_h,
                "severe_collapse_gt_0p02": severe,
                "pass_guardrail": not severe,
            })

    route_summary_rows = [{
        "route_name": FROZEN_ROUTE,
        "input_embedding": "EAT min_clamp__mean",
        "score_protocol": FROZEN_ROUTE,
        "mean_hmean": summary_frozen["mean_hmean"],
        "mean_pAUC": summary_frozen["mean_pAUC"],
        "mean_TPR@99n": summary_frozen["mean_TPR@99n"],
        "delta_hmean_vs_baseline": float(summary_frozen["mean_hmean"]) - REGISTERED_BASELINE_MEAN_HMEAN,
        "delta_pAUC_vs_baseline": float(comp_frozen["delta_pAUC_vs_baseline"]),
        "delta_TPR99_vs_baseline": float(comp_frozen["delta_TPR99_vs_baseline"]),
        "delta_hmean_vs_old_best": float(summary_frozen["mean_hmean"]) - REGISTERED_OLD_BEST_MEAN_HMEAN,
        "improved_machine_count_vs_baseline": int(comp_frozen["improved_machine_count_vs_baseline"]),
        "strong_machine_severe_collapse": comp_frozen["strong_machine_severe_collapse"],
        "freeze_decision": "ACCEPT" if all(checks.values()) else "REJECT",
    }]

    candidate_ranking_rows = []
    reasons = {
        FROZEN_ROUTE: "Best candidate by mean_hmean; selected for main route.",
        "ldn_diff_K5_scorek5": "Close second-best difference-based local density route; not selected.",
        "gwrp_diff_r09_scorek5": "Difference-based GWRP improves over baseline but trails K1 diff.",
        "rasdsr_ratio_avg_K5_r09_scorek5": "Fixed score-level average improves but is not selected.",
        "ldn_ratio_K1_scorek5": "Ratio-based K1 route improves, but less than diff-based K1.",
        "gwrp_ratio_r09_scorek5": "Ratio-based GWRP improves modestly.",
        "ldn_ratio_K5_scorek5": "Ratio-based K5 route provides smaller improvement.",
        "baseline_knn_k5": "Reference frozen baseline route.",
    }
    for row in rank_rows:
        candidate = row["candidate"]
        candidate_ranking_rows.append({
            **row,
            "selected_for_main_route": candidate == FROZEN_ROUTE,
            "reason": reasons.get(candidate, ""),
        })

    paper_ready_main = [
        {
            "route": "EAT min_clamp__mean baseline",
            "training_free": True,
            "backbone": "clean EAT",
            "protocol": "baseline_knn_k5",
            "mean_hmean": REGISTERED_BASELINE_MEAN_HMEAN,
            "mean_pAUC": get_row(summary_rows, candidate=REFERENCE_BASELINE)["mean_pAUC"],
            "mean_TPR@99n": get_row(summary_rows, candidate=REFERENCE_BASELINE)["mean_TPR@99n"],
            "notes": "Frozen clean EAT baseline with min_clamp mean pooling.",
        },
        {
            "route": "EAT min_clamp__mean + ldn_diff_K1_scorek5",
            "training_free": True,
            "backbone": "clean EAT",
            "protocol": FROZEN_ROUTE,
            "mean_hmean": summary_frozen["mean_hmean"],
            "mean_pAUC": summary_frozen["mean_pAUC"],
            "mean_TPR@99n": summary_frozen["mean_TPR@99n"],
            "notes": "Frozen route accepted after Stage18E local-density audit.",
        },
        {
            "route": "old best F_P1_mean_std_A1_065",
            "training_free": True,
            "backbone": "clean EAT + A1",
            "protocol": "fixed score fusion",
            "mean_hmean": REGISTERED_OLD_BEST_MEAN_HMEAN,
            "mean_pAUC": 0.558321,
            "mean_TPR@99n": 0.098095,
            "notes": "Previous registered best route.",
        },
    ]

    ablation_interpret = {
        "baseline_knn_k5": "Reference baseline route.",
        "ldn_ratio_K1_scorek5": "Difference-free local density ratio improves but is weaker than diff-based K1.",
        "ldn_diff_K1_scorek5": "Difference-based normalization with K1 is best and selected.",
        "ldn_ratio_K5_scorek5": "Ratio-based K5 route gives only marginal improvement.",
        "ldn_diff_K5_scorek5": "Difference-based K5 route is close to best but not selected.",
        "gwrp_ratio_r09_scorek5": "Ratio-based GWRP improves but less than diff-based candidates.",
        "gwrp_diff_r09_scorek5": "Difference-based GWRP outperforms ratio version but trails K1 diff.",
        "rasdsr_ratio_avg_K5_r09_scorek5": "Rasd-sr-style fixed average is competitive but not selected.",
    }
    paper_ready_ablation = []
    for row in rank_rows:
        candidate = row["candidate"]
        paper_ready_ablation.append({
            "candidate": candidate,
            "mean_hmean": row["mean_hmean"],
            "mean_pAUC": row["mean_pAUC"],
            "mean_TPR@99n": row["mean_TPR@99n"],
            "selected": candidate == FROZEN_ROUTE,
            "interpretation": ablation_interpret[candidate],
        })

    freeze_accepted = all(checks.values())
    verdict_out = {
        "freeze_accepted": freeze_accepted,
        "frozen_route_name": FROZEN_ROUTE,
        "mean_hmean": float(summary_frozen["mean_hmean"]),
        "mean_pAUC": float(summary_frozen["mean_pAUC"]),
        "mean_TPR99": float(summary_frozen["mean_TPR@99n"]),
        "delta_hmean_vs_baseline": float(route_summary_rows[0]["delta_hmean_vs_baseline"]),
        "delta_hmean_vs_old_best": float(route_summary_rows[0]["delta_hmean_vs_old_best"]),
        "improved_machine_count_vs_baseline": improved_count,
        "strong_machine_severe_collapse": any(r["severe_collapse_gt_0p02"] for r in strong_guardrail_rows),
        "limitations": [
            "fan decreases",
            "still only DCASE current available protocol",
            "no cross-year validation yet",
        ],
        "next_allowed_steps": [
            "paper table packaging",
            "optional cross-year sanity if data/protocol available",
            "optional local-density explanation figure",
        ],
        "forbidden_next_steps": [
            "tuning K",
            "tuning r",
            "adding selector",
            "adding fusion just to chase score",
        ],
    }

    write_json(out_root / "freeze_audit_status.json", {"checks": checks})
    write_csv(out_root / "route_summary.csv", route_summary_rows)
    write_csv(out_root / "candidate_ranking_paper.csv", candidate_ranking_rows)
    write_csv(out_root / "per_machine_delta_paper.csv", per_machine_delta_rows)
    write_csv(out_root / "tail_delta_paper.csv", tail_delta_rows)
    write_csv(out_root / "strong_machine_guardrail.csv", strong_guardrail_rows)
    write_csv(out_root / "paper_ready_table_main.csv", paper_ready_main)
    write_csv(out_root / "paper_ready_table_ablation.csv", paper_ready_ablation)
    write_json(out_root / "stage18f_route_freeze_verdict.json", verdict_out)
    write_markdown(Path(args.doc_path), build_freeze_markdown(verdict_out))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "freeze_audit":
        run_freeze_audit(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
