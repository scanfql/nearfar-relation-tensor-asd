#!/usr/bin/env python
"""Stage22C: summarize Stage22 geometry-objective route outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE22A_ROOT_DEFAULT = "results/stage22a_backend_geometry_failure"
STAGE22B_ROOT_DEFAULT = "results/stage22b_geometry_objective_sanity"
ADAPTED_EVAL_ROOT_DEFAULT = "results/stage22b_adapted_eval"
OUTPUT_ROOT_DEFAULT = "results/stage22c_route_summary"
DOC_PATH_DEFAULT = "docs/CTAF_STAGE22C_ROUTE_SUMMARY.md"
MACHINE_DEFAULT = "fan"

MACHINE_REFS = {
    "fan": {
        "stage18_baseline_hmean": 0.5820070015781968,
        "stage18_ldn_hmean": 0.5527191284766243,
    },
    "gearbox": {
        "stage18_baseline_hmean": 0.5770340180864264,
        "stage18_ldn_hmean": 0.5995502077152959,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Summarize Stage22 route outputs")
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--stage22a_root", default=STAGE22A_ROOT_DEFAULT)
    p.add_argument("--stage22b_root", default=STAGE22B_ROOT_DEFAULT)
    p.add_argument("--adapted_eval_root", default=ADAPTED_EVAL_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=DOC_PATH_DEFAULT)
    return p.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def find_row(rows: list[dict], key: str, value: str) -> dict:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def main():
    args = parse_args()
    machine = args.machine
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    geometry = read_json(Path(args.stage22a_root) / machine / "geometry_failure_summary.json")
    geometry_verdict = read_json(Path(args.stage22a_root) / machine / "stage22a_geometry_failure_verdict.json")
    train_verdict = read_json(Path(args.stage22b_root) / "stage22_geometry_training_verdict.json")
    train_log = read_csv(Path(args.stage22b_root) / "train_log.csv")
    eval_summary = read_csv(Path(args.adapted_eval_root) / machine / "summary.csv")
    eval_comparison = read_csv(Path(args.adapted_eval_root) / machine / "comparison_vs_stage18_machine.csv")
    eval_verdict = read_json(Path(args.adapted_eval_root) / machine / f"stage21e_{machine}_adapted_eval_verdict.json")

    baseline_row = find_row(eval_summary, "candidate", "adapted_baseline_knn_k5")
    ldn_row = find_row(eval_summary, "candidate", "adapted_ldn_diff_K1_scorek5")
    last_train = train_log[-1] if train_log else {}

    adapted_baseline_hmean = as_float(baseline_row, "hmean")
    adapted_ldn_hmean = as_float(ldn_row, "hmean")
    adapted_ldn_pauc = as_float(ldn_row, "pAUC")
    adapted_ldn_tpr99 = as_float(ldn_row, "TPR@99n")
    refs = MACHINE_REFS.get(machine, {})
    ref_ldn = refs.get("stage18_ldn_hmean")
    ref_baseline = refs.get("stage18_baseline_hmean")
    delta_vs_stage18_ldn = adapted_ldn_hmean - ref_ldn if ref_ldn is not None else None
    delta_vs_stage18_baseline = adapted_ldn_hmean - ref_baseline if ref_baseline is not None else None
    gate_pass = adapted_ldn_hmean > ref_ldn if machine == "fan" and ref_ldn is not None else bool(eval_verdict and eval_verdict.get("local_guardrail_pass"))
    strong_pass = adapted_ldn_hmean >= ref_baseline if machine == "fan" and ref_baseline is not None else bool(eval_verdict and eval_verdict.get("baseline_guardrail_pass"))

    next_action = "stop_stage22b_do_not_expand"
    if strong_pass:
        next_action = "promote_to_full7_eval_candidate"
    elif gate_pass:
        next_action = "weak_pass_consider_gearbox_guardrail_before_full7"

    summary = {
        "machine": machine,
        "stage22a_geometry_available": geometry is not None,
        "stage22b_training_available": train_verdict is not None,
        "stage22b_fan_eval_available": bool(eval_summary),
        "stage22a_neighbor_retention_at5": geometry.get("neighbor_retention_at5") if geometry else None,
        "stage22a_effective_rank_ratio_adapted_over_clean": geometry.get("effective_rank_ratio_adapted_over_clean") if geometry else None,
        "stage22a_rho_mean_ratio_adapted_over_clean": geometry.get("rho_mean_ratio_adapted_over_clean") if geometry else None,
        "stage22a_failure_flags": geometry_verdict.get("finding_flags") if geometry_verdict else None,
        "best_checkpoint": train_verdict.get("best_checkpoint") if train_verdict else None,
        "best_epoch": train_verdict.get("best_epoch") if train_verdict else None,
        "best_val_loss": train_verdict.get("best_val_loss") if train_verdict else None,
        "last_epoch": last_train.get("epoch"),
        "last_train_total": last_train.get("train_total"),
        "last_val_total": last_train.get("val_total"),
        "adapted_baseline_hmean": adapted_baseline_hmean,
        "adapted_ldn_hmean": adapted_ldn_hmean,
        "adapted_ldn_pAUC": adapted_ldn_pauc,
        "adapted_ldn_TPR99": adapted_ldn_tpr99,
        f"delta_vs_stage18_{machine}_ldn": delta_vs_stage18_ldn,
        f"delta_vs_stage18_{machine}_baseline": delta_vs_stage18_baseline,
        f"{machine}_gate_pass_vs_stage18_ldn": gate_pass,
        f"{machine}_strong_pass_vs_stage18_baseline": strong_pass,
        "next_action": next_action,
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_test_label_checkpoint_selection": True,
        },
    }

    write_json(out_root / f"stage22c_{machine}_route_summary.json", summary)
    write_csv(out_root / f"stage22c_{machine}_eval_comparison.csv", eval_comparison)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage22C Route Summary",
        "",
        f"- machine: `{machine}`",
        f"- Stage22A neighbor retention @5: `{summary['stage22a_neighbor_retention_at5']}`",
        f"- Stage22A effective-rank ratio adapted/clean: `{summary['stage22a_effective_rank_ratio_adapted_over_clean']}`",
        f"- Stage22A rho mean ratio adapted/clean: `{summary['stage22a_rho_mean_ratio_adapted_over_clean']}`",
        f"- best checkpoint: `{summary['best_checkpoint']}`",
        f"- best epoch: `{summary['best_epoch']}`",
        f"- best validation loss: `{summary['best_val_loss']}`",
        "",
        f"## {machine.capitalize()} Gate",
        "",
        f"- adapted baseline hmean: `{adapted_baseline_hmean:.12f}`",
        f"- adapted LDN hmean: `{adapted_ldn_hmean:.12f}`",
        f"- adapted LDN pAUC: `{adapted_ldn_pauc:.12f}`",
        f"- adapted LDN TPR@99n: `{adapted_ldn_tpr99:.12f}`",
        f"- delta vs Stage18 {machine} LDN: `{delta_vs_stage18_ldn}`",
        f"- delta vs Stage18 {machine} baseline: `{delta_vs_stage18_baseline}`",
        f"- pass Stage18 {machine} LDN gate: `{gate_pass}`",
        f"- pass Stage18 {machine} baseline gate: `{strong_pass}`",
        "",
        "## Decision",
        "",
        f"- next action: `{next_action}`",
        "",
        "No LDN parameter tuning, selector, or test-label checkpoint selection is introduced by this summary.",
    ])


if __name__ == "__main__":
    main()
