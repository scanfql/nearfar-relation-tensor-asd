#!/usr/bin/env python
"""Stage22E: freeze audit for weak hmean-positive Stage22D route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE22D_ROOT_DEFAULT = "results/stage22d_adapted_full7_eval"
OUTPUT_ROOT_DEFAULT = "results/stage22e_route_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE22.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_ROUTE = "EAT min_clamp__mean + ldn_diff_K1_scorek5"
CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572


def parse_args():
    p = argparse.ArgumentParser(description="Stage22E route freeze audit")
    p.add_argument("--stage22d_root", default=STAGE22D_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=DOC_PATH_DEFAULT)
    return p.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def get_row(rows: list[dict], **filters) -> dict:
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    raise KeyError(f"Missing row: {filters}")


def f(row: dict, key: str) -> float:
    return float(row[key])


def main():
    args = parse_args()
    root = Path(args.stage22d_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(root / "per_machine.csv")
    collapse = read_csv(root / "collapse_check.csv")

    machine_rows = []
    for row in collapse:
        if row["candidate"] != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score_row = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base_row = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        machine_rows.append({
            "machine": machine,
            "adapted_baseline_hmean": f(base_row, "hmean"),
            "stage22d_hmean": f(score_row, "hmean"),
            "delta_hmean_vs_adapted_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "stage22d_pAUC": f(score_row, "pAUC"),
            "stage22d_TPR99": f(score_row, "TPR@99n"),
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        })

    positive = [r for r in machine_rows if r["delta_hmean_vs_adapted_baseline"] > 0]
    negative = [r for r in machine_rows if r["delta_hmean_vs_adapted_baseline"] < 0]
    strongest_gain = max(machine_rows, key=lambda r: r["delta_hmean_vs_adapted_baseline"])
    largest_drop = min(machine_rows, key=lambda r: r["delta_hmean_vs_adapted_baseline"])

    weak_hmean_positive = bool(
        verdict["promotable"]
        and verdict["mean_hmean"] > CLEAN_BEST_MEAN_HMEAN
        and verdict["mean_pAUC"] < CLEAN_BEST_MEAN_PAUC
        and verdict["mean_TPR99"] < CLEAN_BEST_MEAN_TPR99
    )
    freeze_status = "weak_hmean_positive_freeze" if weak_hmean_positive else "not_freezable"

    route_summary = {
        "freeze_status": freeze_status,
        "route": FROZEN_ROUTE,
        "route_description": "Stage22 geometry-objective adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "previous_confirmed_best_route": CLEAN_BEST_ROUTE,
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "delta_hmean_vs_clean_best": verdict["delta_hmean_vs_confirmed_clean_best"],
        "delta_pAUC_vs_clean_best": verdict["mean_pAUC"] - CLEAN_BEST_MEAN_PAUC,
        "delta_TPR99_vs_clean_best": verdict["mean_TPR99"] - CLEAN_BEST_MEAN_TPR99,
        "improved_machine_count_vs_adapted_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_adapted_baseline": [r["machine"] for r in positive],
        "negative_machines_vs_adapted_baseline": [r["machine"] for r in negative],
        "strongest_gain_machine": strongest_gain["machine"],
        "strongest_gain_delta_hmean": strongest_gain["delta_hmean_vs_adapted_baseline"],
        "largest_drop_machine": largest_drop["machine"],
        "largest_drop_delta_hmean": largest_drop["delta_hmean_vs_adapted_baseline"],
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "recommended_status": "new hmean best only; keep clean route as balanced reference",
        "forbidden_followups": [
            "Do not tune LDN K/r/score_k.",
            "Do not add a selector to hide fan degradation.",
            "Do not call this a pAUC or TPR@99n improvement.",
            "Do not select checkpoints with test labels.",
        ],
    }

    write_json(out_root / "stage22e_route_freeze_summary.json", route_summary)
    write_csv(out_root / "stage22e_machine_delta_table.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE22",
        "",
        "## 1. Stage22D frozen result",
        "",
        "Stage22B/C geometry-objective adapted EAT produced a weak hmean-positive full7 route:",
        "",
        f"- route: `{FROZEN_ROUTE}`",
        f"- route description: `Stage22 geometry-objective adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5`",
        f"- mean_hmean: `{verdict['mean_hmean']}`",
        f"- mean_pAUC: `{verdict['mean_pAUC']}`",
        f"- mean_TPR@99n: `{verdict['mean_TPR99']}`",
        f"- delta_hmean_vs_clean_best: `{verdict['delta_hmean_vs_confirmed_clean_best']}`",
        f"- delta_pAUC_vs_clean_best: `{route_summary['delta_pAUC_vs_clean_best']}`",
        f"- delta_TPR99_vs_clean_best: `{route_summary['delta_TPR99_vs_clean_best']}`",
        f"- improved_machine_count_vs_adapted_baseline: `{verdict['improved_machine_count_vs_adapted_baseline']}`",
        f"- strong_machine_severe_collapse: `{verdict['strong_machine_severe_collapse']}`",
        f"- freeze_status: `{freeze_status}`",
        "",
        "Conclusion: this is a new hmean best, but not a balanced metric improvement.",
        "",
        "## 2. Machine-level pattern",
        "",
        f"- positive machines vs adapted baseline: `{', '.join(route_summary['positive_machines_vs_adapted_baseline'])}`",
        f"- negative machines vs adapted baseline: `{', '.join(route_summary['negative_machines_vs_adapted_baseline'])}`",
        f"- strongest gain: `{strongest_gain['machine']} ({strongest_gain['delta_hmean_vs_adapted_baseline']:+.12f})`",
        f"- largest drop: `{largest_drop['machine']} ({largest_drop['delta_hmean_vs_adapted_baseline']:+.12f})`",
        "",
        "The gain is mostly carried by valve, with additional gains on gearbox, slider, bearing, and toycar. Fan drops substantially; toytrain drops slightly.",
        "",
        "## 3. Current route hierarchy",
        "",
        f"- hmean-best route: `Stage22D geometry-objective adapted EAT + {FROZEN_ROUTE}`",
        f"- balanced reference route: `{CLEAN_BEST_ROUTE}`",
        "",
        "Use Stage22D only when optimizing mean hmean. Use the clean frozen route as the balanced reference because Stage22D lowers mean pAUC and mean TPR@99n.",
        "",
        "## 4. Forbidden next moves",
        "",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not add per-machine selector or fusion to hide fan degradation.",
        "- Do not claim pAUC or TPR@99n improvement.",
        "- Do not use test labels for checkpoint selection.",
        "- Do not continue fan-only adaptation as a main route.",
        "",
        "## 5. Recommended next step",
        "",
        "Stop Stage22B as a route-search branch and freeze the result as `weak_hmean_positive_freeze`. If continuing, design a new objective specifically to recover fan without sacrificing the valve gain, but it must be validated without selector, LDN tuning, or test-label checkpoint selection.",
    ])


if __name__ == "__main__":
    main()
