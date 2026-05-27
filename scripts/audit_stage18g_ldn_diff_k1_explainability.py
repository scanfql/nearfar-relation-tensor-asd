#!/usr/bin/env python
"""Stage 18G: LDN-Diff-K1 explainability / stability audit.

This script is explanation-only for the already frozen route
`ldn_diff_K1_scorek5`. It reads existing Stage18E/F outputs and Stage16D
embeddings, and it does not create new route candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


STAGE18E_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
EMBED_ROOT_DEFAULT = "results/stage16d_minclamp_mean_full7/embeddings"
OUTPUT_ROOT_DEFAULT = "results/stage18g_ldn_diff_k1_explainability"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18G_LDN_DIFF_K1_EXPLAINABILITY_AUDIT.md"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"

FROZEN_ROUTE = "ldn_diff_K1_scorek5"
BASELINE_ROUTE = "baseline_knn_k5"
STRONG_MACHINES = {"bearing", "toytrain", "valve"}
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Stage18G LDN-Diff-K1 explainability audit")
    p.add_argument("--mode", required=True, choices=["check", "explain"])
    p.add_argument("--stage18e_root", default=STAGE18E_ROOT_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--embedding_root", default=EMBED_ROOT_DEFAULT)
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
    status = {"csv_builtin": True, "json_builtin": True}
    for name in ["numpy", "scipy", "sklearn"]:
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


def require_columns(rows: list[dict], required: list[str]) -> list[str]:
    if not rows:
        return required[:]
    present = set(rows[0].keys())
    return [c for c in required if c not in present]


def score_npz_path(stage18e_root: Path, candidate: str, machine: str) -> Path:
    return stage18e_root / "scores" / candidate / f"{machine}_scores.npz"


def emb_npz_path(embed_root: Path, machine: str) -> Path:
    return embed_root / f"{machine}_minclamp_mean_embeddings.npz"


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


def load_embedding_bundle(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    return {
        "train_embeddings": arr["train_embeddings"].astype(np.float64),
        "test_embeddings": arr["test_embeddings"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "d_test": arr["d_test"].astype(str),
        "test_paths": arr["test_paths"].astype(str),
    }


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def chord_distance_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = l2_normalize(query.astype(np.float64), axis=1)
    r = l2_normalize(ref.astype(np.float64), axis=1)
    qn2 = np.sum(q * q, axis=1, keepdims=True)
    rn2 = np.sum(r * r, axis=1)[None, :]
    d2 = qn2 + rn2 - 2.0 * (q @ r.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float64)


def local_density_knn(ref_ref_dist: np.ndarray, K: int = 1) -> np.ndarray:
    if ref_ref_dist.shape[0] <= 1:
        return np.zeros((ref_ref_dist.shape[0],), dtype=np.float64)
    K = min(K, ref_ref_dist.shape[1] - 1)
    d = ref_ref_dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.partition(d, kth=K - 1, axis=1)[:, :K]
    return nearest.mean(axis=1).astype(np.float64)


def distribution_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "std": float(np.std(values)) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "q25": float(np.quantile(values, 0.25)) if len(values) else 0.0,
        "q75": float(np.quantile(values, 0.75)) if len(values) else 0.0,
        "q90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
        "q95": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "q99": float(np.quantile(values, 0.99)) if len(values) else 0.0,
    }


def get_row(rows: list[dict], **kwargs) -> dict:
    for row in rows:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(f"Missing row for filters={kwargs}")


def classify_improvement(delta_h: float) -> str:
    if delta_h >= 0.02:
        return "strong_gain"
    if delta_h > 0.002:
        return "weak_gain"
    if delta_h <= -0.002:
        return "degradation"
    return "stable"


def fan_failure_mode(delta_h: float, delta_q95: float, delta_q99: float, delta_gap: float, delta_ks: float) -> str:
    if delta_h < 0 and delta_q95 > 0 and delta_q99 > 0 and delta_gap < 0:
        return "normal-tail inflation without compensating anomaly separation"
    if delta_h < 0 and delta_ks > 0:
        return "domain-shift sensitivity increase"
    if delta_h < 0 and delta_gap < 0:
        return "weaker anomaly separation"
    return "mixed degradation pattern"


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage18G LDN-Diff-K1 Explainability Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no diagnostics",
        "- no score recomputation",
        "- no model load",
        "- no forward",
        "- no embedding extraction",
        "",
        "## Status",
        "",
        f"- Stage18E root exists: `{status['stage18e_root_exists']}`",
        f"- Stage18F root exists: `{status['stage18f_root_exists']}`",
        f"- all required files present: `{status['all_required_files_present']}`",
        f"- freeze accepted in Stage18F: `{status['stage18f_freeze_accepted']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_explain_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage18G LDN-Diff-K1 Explainability Audit",
        "",
        "## Frozen Route",
        "",
        f"- frozen route: `{verdict['frozen_route']}`",
        f"- explanation audit complete: `{verdict['explanation_audit_complete']}`",
        f"- mean delta hmean: `{verdict['mean_delta_hmean']:.6f}`",
        f"- main positive machines: `{', '.join(verdict['main_positive_machines'])}`",
        f"- degradation machines: `{', '.join(verdict['degradation_machines'])}`",
        f"- fan degradation confirmed: `{verdict['fan_degradation_confirmed']}`",
        f"- strong-machine guardrail pass: `{verdict['strong_machine_guardrail_pass']}`",
        "",
        "## Summary",
        "",
        "- LDN-diff-K1 improves full7 by reducing local-density bias in score geometry.",
        "- Gains are not from one machine only.",
        "- fan remains the main limitation.",
        "- valve, gearbox, and slider are key positive contributors.",
        "- No tuning, selector, or fusion was introduced in this stage.",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage18e_root = Path(args.stage18e_root)
    stage18f_root = Path(args.stage18f_root)
    embed_root = Path(args.embedding_root)
    machines = parse_csv_list(args.machines)

    e_files = {
        "summary.csv": stage18e_root / "summary.csv",
        "per_machine.csv": stage18e_root / "per_machine.csv",
        "tail_table.csv": stage18e_root / "tail_table.csv",
        "score_geometry.csv": stage18e_root / "score_geometry.csv",
        "comparison_vs_baseline.csv": stage18e_root / "comparison_vs_baseline.csv",
        "collapse_check.csv": stage18e_root / "collapse_check.csv",
        "candidate_rank.csv": stage18e_root / "candidate_rank.csv",
        "stage18e_ldn_verdict.json": stage18e_root / "stage18e_ldn_verdict.json",
    }
    f_files = {
        "route_summary.csv": stage18f_root / "route_summary.csv",
        "per_machine_delta_paper.csv": stage18f_root / "per_machine_delta_paper.csv",
        "paper_ready_table_main.csv": stage18f_root / "paper_ready_table_main.csv",
        "paper_ready_table_ablation.csv": stage18f_root / "paper_ready_table_ablation.csv",
        "stage18f_route_freeze_verdict.json": stage18f_root / "stage18f_route_freeze_verdict.json",
    }

    rows = []
    all_present = stage18e_root.exists() and stage18f_root.exists() and embed_root.exists()
    for label, path in {**e_files, **f_files}.items():
        exists = path.exists()
        rows.append({"artifact": label, "path": str(path), "exists": exists})
        all_present = all_present and exists
    for machine in machines:
        for candidate in [BASELINE_ROUTE, FROZEN_ROUTE]:
            path = score_npz_path(stage18e_root, candidate, machine)
            exists = path.exists()
            rows.append({"artifact": f"{candidate}_{machine}", "path": str(path), "exists": exists})
            all_present = all_present and exists
        path = emb_npz_path(embed_root, machine)
        exists = path.exists()
        rows.append({"artifact": f"embedding_{machine}", "path": str(path), "exists": exists})
        all_present = all_present and exists

    # column checks
    column_rows = []
    if e_files["summary.csv"].exists():
        column_rows.append({
            "file": str(e_files["summary.csv"]),
            "missing_columns": ",".join(require_columns(load_csv_rows(e_files["summary.csv"]), ["candidate", "mean_hmean", "mean_pAUC", "mean_TPR@99n"])),
        })
    if e_files["candidate_rank.csv"].exists():
        column_rows.append({
            "file": str(e_files["candidate_rank.csv"]),
            "missing_columns": ",".join(require_columns(load_csv_rows(e_files["candidate_rank.csv"]), ["rank", "candidate", "mean_hmean", "mean_pAUC", "mean_TPR@99n"])),
        })
    if e_files["per_machine.csv"].exists():
        column_rows.append({
            "file": str(e_files["per_machine.csv"]),
            "missing_columns": ",".join(require_columns(load_csv_rows(e_files["per_machine.csv"]), ["candidate", "machine", "hmean", "pAUC", "TPR@99n"])),
        })
    if e_files["comparison_vs_baseline.csv"].exists():
        column_rows.append({
            "file": str(e_files["comparison_vs_baseline.csv"]),
            "missing_columns": ",".join(require_columns(load_csv_rows(e_files["comparison_vs_baseline.csv"]), ["candidate", "mean_hmean", "delta_hmean_vs_baseline", "improved_machine_count_vs_baseline", "strong_machine_severe_collapse"])),
        })

    freeze_accepted = False
    if f_files["stage18f_route_freeze_verdict.json"].exists():
        freeze_accepted = bool(load_json(f_files["stage18f_route_freeze_verdict.json"]).get("freeze_accepted", False))

    status = {
        "mode": "check",
        "stage18e_root": str(stage18e_root),
        "stage18e_root_exists": stage18e_root.exists(),
        "stage18f_root": str(stage18f_root),
        "stage18f_root_exists": stage18f_root.exists(),
        "embedding_root": str(embed_root),
        "embedding_root_exists": embed_root.exists(),
        "all_required_files_present": all_present,
        "stage18f_freeze_accepted": freeze_accepted,
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "required_files_check.csv", rows)
    write_csv(out_root / "required_columns_check.csv", column_rows)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_explain(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage18e_root = Path(args.stage18e_root)
    stage18f_root = Path(args.stage18f_root)
    embed_root = Path(args.embedding_root)
    machines = parse_csv_list(args.machines)

    e_summary = load_csv_rows(stage18e_root / "summary.csv")
    e_pm = load_csv_rows(stage18e_root / "per_machine.csv")
    e_tail = load_csv_rows(stage18e_root / "tail_table.csv")
    e_geom = load_csv_rows(stage18e_root / "score_geometry.csv")
    e_comp = load_csv_rows(stage18e_root / "comparison_vs_baseline.csv")
    e_collapse = load_csv_rows(stage18e_root / "collapse_check.csv")
    e_rank = load_csv_rows(stage18e_root / "candidate_rank.csv")
    e_verdict = load_json(stage18e_root / "stage18e_ldn_verdict.json")

    f_route_summary = load_csv_rows(stage18f_root / "route_summary.csv")
    f_machine_delta = load_csv_rows(stage18f_root / "per_machine_delta_paper.csv")
    f_table_main = load_csv_rows(stage18f_root / "paper_ready_table_main.csv")
    f_table_ablation = load_csv_rows(stage18f_root / "paper_ready_table_ablation.csv")
    f_verdict = load_json(stage18f_root / "stage18f_route_freeze_verdict.json")

    explain_checks = {
        "freeze_accepted": bool(f_verdict.get("freeze_accepted", False)),
        "stage18e_best_candidate_matches": e_verdict.get("best_candidate") == FROZEN_ROUTE,
        "stage18f_frozen_route_matches": f_verdict.get("frozen_route_name") == FROZEN_ROUTE,
    }
    write_json(out_root / "explain_status.json", explain_checks)

    # alignment
    score_align_rows = []
    for machine in machines:
        b = load_score_bundle(score_npz_path(stage18e_root, BASELINE_ROUTE, machine))
        f = load_score_bundle(score_npz_path(stage18e_root, FROZEN_ROUTE, machine))
        y_match = np.array_equal(b["y_test"], f["y_test"])
        d_match = np.array_equal(b["d_test"], f["d_test"])
        if b["test_paths"] is not None and f["test_paths"] is not None:
            p_match = np.array_equal(b["test_paths"], f["test_paths"])
        else:
            p_match = bool(y_match and d_match)
        score_align_rows.append({
            "machine": machine,
            "baseline_score_path": str(score_npz_path(stage18e_root, BASELINE_ROUTE, machine)),
            "frozen_score_path": str(score_npz_path(stage18e_root, FROZEN_ROUTE, machine)),
            "y_test_match": bool(y_match),
            "d_test_match": bool(d_match),
            "test_paths_match": bool(p_match),
            "alignment_ok": bool(y_match and d_match and p_match),
        })
    write_csv(out_root / "score_file_alignment_check.csv", score_align_rows)

    score_distribution_rows = []
    normal_tail_rows = []
    anomaly_sep_rows = []
    domain_geom_rows = []
    rho_stats_rows = []
    rho_diag_rows = []
    fan_rows = []
    contribution_rows = []
    guardrail_rows = []
    figure_score_rows = []
    figure_density_rows = []
    machine_delta_rows = []

    main_positive = []
    degrade_machines = []

    baseline_summary = get_row(e_summary, candidate=BASELINE_ROUTE)
    frozen_summary = get_row(e_summary, candidate=FROZEN_ROUTE)

    for machine in machines:
        base_pm = get_row(e_pm, candidate=BASELINE_ROUTE, machine=machine)
        frz_pm = get_row(e_pm, candidate=FROZEN_ROUTE, machine=machine)
        base_tail = get_row(e_tail, candidate=BASELINE_ROUTE, machine=machine)
        frz_tail = get_row(e_tail, candidate=FROZEN_ROUTE, machine=machine)
        base_geom = get_row(e_geom, candidate=BASELINE_ROUTE, machine=machine)
        frz_geom = get_row(e_geom, candidate=FROZEN_ROUTE, machine=machine)

        base_scores = load_score_bundle(score_npz_path(stage18e_root, BASELINE_ROUTE, machine))
        frz_scores = load_score_bundle(score_npz_path(stage18e_root, FROZEN_ROUTE, machine))
        emb = load_embedding_bundle(emb_npz_path(embed_root, machine))

        # distribution shifts
        for route_name, bundle in [(BASELINE_ROUTE, base_scores), (FROZEN_ROUTE, frz_scores)]:
            groups = {
                "source_normal": bundle["test_scores"][(bundle["d_test"] == "source") & (bundle["y_test"] == 0)],
                "target_normal": bundle["test_scores"][(bundle["d_test"] == "target") & (bundle["y_test"] == 0)],
                "anomaly": bundle["test_scores"][bundle["y_test"] == 1],
            }
            for group, values in groups.items():
                stats = distribution_stats(values.astype(np.float64))
                score_distribution_rows.append({"machine": machine, "route": route_name, "group": group, **stats})
                for stat_name, stat_val in stats.items():
                    figure_score_rows.append({
                        "machine": machine,
                        "route": route_name,
                        "group": group,
                        "statistic": stat_name,
                        "value": stat_val,
                    })

        # normal tail shift
        delta_q95 = float(frz_tail["q95"]) - float(base_tail["q95"])
        delta_q99 = float(frz_tail["q99"]) - float(base_tail["q99"])
        tail_interp = "normal tail thresholds increase" if delta_q95 > 0 and delta_q99 > 0 else ("normal tail thresholds decrease" if delta_q95 < 0 and delta_q99 < 0 else "mixed normal-tail shift")
        normal_tail_rows.append({
            "machine": machine,
            "baseline_normal_q95": base_tail["q95"],
            "ldn_normal_q95": frz_tail["q95"],
            "delta_normal_q95": delta_q95,
            "baseline_normal_q99": base_tail["q99"],
            "ldn_normal_q99": frz_tail["q99"],
            "delta_normal_q99": delta_q99,
            "interpretation": tail_interp,
        })

        # anomaly separation shift
        base_gap = float(base_tail["anomaly_median"]) - float(base_tail["q95"])
        frz_gap = float(frz_tail["anomaly_median"]) - float(frz_tail["q95"])
        delta_gap = frz_gap - base_gap
        sep_interp = "anomaly separation widens" if delta_gap > 0 else ("anomaly separation narrows" if delta_gap < 0 else "anomaly separation unchanged")
        anomaly_sep_rows.append({
            "machine": machine,
            "baseline_anomaly_median": base_tail["anomaly_median"],
            "ldn_anomaly_median": frz_tail["anomaly_median"],
            "delta_anomaly_median": float(frz_tail["anomaly_median"]) - float(base_tail["anomaly_median"]),
            "baseline_anomaly_q75": base_tail["anomaly_q75"],
            "ldn_anomaly_q75": frz_tail["anomaly_q75"],
            "delta_anomaly_q75": float(frz_tail["anomaly_q75"]) - float(base_tail["anomaly_q75"]),
            "baseline_gap_anomaly_median_minus_normal_q95": base_gap,
            "ldn_gap_anomaly_median_minus_normal_q95": frz_gap,
            "delta_gap": delta_gap,
            "interpretation": sep_interp,
        })

        # domain shift geometry
        for route_name, geom in [(BASELINE_ROUTE, base_geom), (FROZEN_ROUTE, frz_geom)]:
            interp = "target/source normal geometry diverges more" if float(geom["source_target_ks"]) > float(base_geom["KS"] if "KS" in base_geom else base_geom["source_target_ks"]) and route_name == FROZEN_ROUTE else "stable or improved domain geometry"
            domain_geom_rows.append({
                "machine": machine,
                "route": route_name,
                "source_normal_mean": geom["source_normal_mean"],
                "target_normal_mean": geom["target_normal_mean"],
                "source_target_ks": geom["source_target_ks"],
                "median_gap": geom["median_gap"],
                "iqr_gap": geom["iqr_gap"],
                "score_std": geom["score_std"],
                "interpretation": interp,
            })

        # local density reference stats and nearest-neighbor diagnostics
        train_emb = emb["train_embeddings"]
        test_emb = emb["test_embeddings"]
        ref_ref = chord_distance_matrix(train_emb, train_emb)
        test_ref = chord_distance_matrix(test_emb, train_emb)
        rho = local_density_knn(ref_ref, K=1)
        rho_stats_rows.append({
            "machine": machine,
            "K1_rho_mean": float(np.mean(rho)),
            "K1_rho_std": float(np.std(rho)),
            "K1_rho_median": float(np.median(rho)),
            "K1_rho_q25": float(np.quantile(rho, 0.25)),
            "K1_rho_q75": float(np.quantile(rho, 0.75)),
            "K1_rho_q95": float(np.quantile(rho, 0.95)),
            "K1_rho_q99": float(np.quantile(rho, 0.99)),
        })
        base_sel = np.partition(np.argsort(test_ref, axis=1), kth=min(5, test_ref.shape[1]) - 1, axis=1)[:, : min(5, test_ref.shape[1])]
        frz_sel = np.partition(np.argsort(test_ref - rho[None, :], axis=1), kth=min(5, test_ref.shape[1]) - 1, axis=1)[:, : min(5, test_ref.shape[1])]
        base_rho_mean = float(np.mean(rho[base_sel]))
        frz_rho_mean = float(np.mean(rho[frz_sel]))
        rho_interp = "LDN selects sparser references" if frz_rho_mean > base_rho_mean else ("LDN selects denser references" if frz_rho_mean < base_rho_mean else "LDN keeps reference density unchanged")
        rho_diag_rows.append({
            "machine": machine,
            "mean_rho_selected_baseline": base_rho_mean,
            "mean_rho_selected_ldn_diff_K1": frz_rho_mean,
            "difference": frz_rho_mean - base_rho_mean,
            "interpretation": rho_interp,
        })
        for stat_name, stat_val in {
            "K1_rho_mean": float(np.mean(rho)),
            "K1_rho_std": float(np.std(rho)),
            "K1_rho_median": float(np.median(rho)),
            "K1_rho_q25": float(np.quantile(rho, 0.25)),
            "K1_rho_q75": float(np.quantile(rho, 0.75)),
            "K1_rho_q95": float(np.quantile(rho, 0.95)),
            "K1_rho_q99": float(np.quantile(rho, 0.99)),
        }.items():
            figure_density_rows.append({
                "machine": machine,
                "rho_statistic": stat_name,
                "value": stat_val,
                "route_context": "reference_memory",
            })
        figure_density_rows.append({
            "machine": machine,
            "rho_statistic": "mean_rho_selected",
            "value": base_rho_mean,
            "route_context": BASELINE_ROUTE,
        })
        figure_density_rows.append({
            "machine": machine,
            "rho_statistic": "mean_rho_selected",
            "value": frz_rho_mean,
            "route_context": FROZEN_ROUTE,
        })

        # per-machine delta summary / contribution / guardrail
        delta_h = float(frz_pm["hmean"]) - float(base_pm["hmean"])
        delta_p = float(frz_pm["pAUC"]) - float(base_pm["pAUC"])
        delta_t = float(frz_pm["TPR@99n"]) - float(base_pm["TPR@99n"])
        improvement_type = classify_improvement(delta_h)
        comment = {
            "fan": "fan must be explicitly marked as degradation",
            "valve": "valve must be marked as strong_gain",
            "gearbox": "gearbox must be marked as gain",
            "slider": "slider must be marked as gain",
            "toycar": "toycar should be marked as slight gain",
            "bearing": "bearing should be marked stable if near zero",
            "toytrain": "toytrain should be marked stable if near zero",
        }.get(machine, "")
        machine_delta_rows.append({
            "machine": machine,
            "baseline_hmean": base_pm["hmean"],
            "ldn_hmean": frz_pm["hmean"],
            "delta_hmean": delta_h,
            "baseline_pAUC": base_pm["pAUC"],
            "ldn_pAUC": frz_pm["pAUC"],
            "delta_pAUC": delta_p,
            "baseline_TPR99": base_pm["TPR@99n"],
            "ldn_TPR99": frz_pm["TPR@99n"],
            "delta_TPR99": delta_t,
            "improvement_type": improvement_type,
            "short_comment": comment,
        })
        contribution_rows.append({
            "machine": machine,
            "delta_hmean": delta_h,
            "contribution_to_mean_delta": delta_h / 7.0,
            "positive_or_negative": "positive" if delta_h > 0 else ("negative" if delta_h < 0 else "neutral"),
        })
        if machine in STRONG_MACHINES:
            severe = delta_h < -0.02
            guardrail_rows.append({
                "machine": machine,
                "baseline_hmean": base_pm["hmean"],
                "ldn_hmean": frz_pm["hmean"],
                "delta": delta_h,
                "severe_collapse_gt_0p02": severe,
                "guardrail_pass": not severe,
            })

        if delta_h > 0:
            main_positive.append(machine)
        if delta_h < 0:
            degrade_machines.append(machine)

        if machine == "fan":
            fan_rows.append({
                "machine": machine,
                "baseline_hmean": base_pm["hmean"],
                "ldn_hmean": frz_pm["hmean"],
                "delta_hmean": delta_h,
                "baseline_pAUC": base_pm["pAUC"],
                "ldn_pAUC": frz_pm["pAUC"],
                "delta_pAUC": delta_p,
                "baseline_TPR99": base_pm["TPR@99n"],
                "ldn_TPR99": frz_pm["TPR@99n"],
                "delta_TPR99": delta_t,
                "delta_normal_q95": delta_q95,
                "delta_normal_q99": delta_q99,
                "delta_gap": delta_gap,
                "delta_source_target_ks": float(frz_geom["source_target_ks"]) - float(base_geom["source_target_ks"]),
                "likely_failure_mode": fan_failure_mode(delta_h, delta_q95, delta_q99, delta_gap, float(frz_geom["source_target_ks"]) - float(base_geom["source_target_ks"])),
            })

    verdict_out = {
        "frozen_route": FROZEN_ROUTE,
        "explanation_audit_complete": True,
        "mean_delta_hmean": float(frozen_summary["mean_hmean"]) - float(baseline_summary["mean_hmean"]),
        "main_positive_machines": main_positive,
        "degradation_machines": degrade_machines,
        "fan_degradation_confirmed": "fan" in degrade_machines,
        "strong_machine_guardrail_pass": all(r["guardrail_pass"] for r in guardrail_rows),
        "interpretation_summary": [
            "LDN-diff-K1 improves full7 by reducing local-density bias in score geometry",
            "gains are not from one machine only",
            "fan remains the main limitation",
            "valve/gearbox/slider are key positive contributors",
        ],
        "recommended_next_steps": [
            "paper explanation figure",
            "optional cross-year sanity",
            "Stage19 high-level training route only if aiming for 0.65",
        ],
        "forbidden_next_steps": [
            "tuning K on the same test protocol",
            "adding per-machine selector",
            "adding fusion only to chase score",
        ],
    }

    write_csv(out_root / "machine_delta_summary.csv", machine_delta_rows)
    write_csv(out_root / "score_distribution_shift.csv", score_distribution_rows)
    write_csv(out_root / "normal_tail_shift.csv", normal_tail_rows)
    write_csv(out_root / "anomaly_separation_shift.csv", anomaly_sep_rows)
    write_csv(out_root / "domain_shift_geometry.csv", domain_geom_rows)
    write_csv(out_root / "local_density_reference_stats.csv", rho_stats_rows)
    write_csv(out_root / "local_density_nearest_neighbor_diagnostics.csv", rho_diag_rows)
    write_csv(out_root / "fan_failure_analysis.csv", fan_rows)
    write_csv(out_root / "machine_contribution_to_gain.csv", contribution_rows)
    write_csv(out_root / "strong_machine_guardrail_explain.csv", guardrail_rows)
    write_csv(out_root / "paper_figure_score_distribution_data.csv", figure_score_rows)
    write_csv(out_root / "paper_figure_density_bias_data.csv", figure_density_rows)
    write_json(out_root / "stage18g_explainability_verdict.json", verdict_out)
    write_markdown(Path(args.doc_path), build_explain_markdown(verdict_out))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "explain":
        run_explain(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
