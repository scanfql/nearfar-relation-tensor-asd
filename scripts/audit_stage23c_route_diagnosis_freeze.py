#!/usr/bin/env python
"""Stage23C: freeze fan-density diagnosis for Stage22/23."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE22E_ROOT_DEFAULT = "results/stage22e_route_freeze"
STAGE23A_ROOT_DEFAULT = "results/stage23a_loss_component_probe"
STAGE23B_ROOT_DEFAULT = "results/stage23b_fan_density_failure"
OUTPUT_ROOT_DEFAULT = "results/stage23c_route_diagnosis_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE23.md"


def parse_args():
    p = argparse.ArgumentParser(description="Stage23C route diagnosis freeze")
    p.add_argument("--stage22e_root", default=STAGE22E_ROOT_DEFAULT)
    p.add_argument("--stage23a_root", default=STAGE23A_ROOT_DEFAULT)
    p.add_argument("--stage23b_root", default=STAGE23B_ROOT_DEFAULT)
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


def f(row: dict, key: str) -> float:
    return float(row[key])


def get_route(rows: list[dict], name: str) -> dict:
    for row in rows:
        if row.get("route") == name:
            return row
    raise KeyError(f"Missing route row: {name}")


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage22e = read_json(Path(args.stage22e_root) / "stage22e_route_freeze_summary.json")
    stage23a = read_json(Path(args.stage23a_root) / "stage23a_loss_component_probe_verdict.json")
    stage23a_table = read_csv(Path(args.stage23a_root) / "stage23a_loss_component_probe_table.csv")
    stage23b = read_json(Path(args.stage23b_root) / "stage23b_fan_density_failure_verdict.json")
    stage23b_table = read_csv(Path(args.stage23b_root) / "stage23b_fan_density_failure_table.csv")

    no_ldn_proxy = get_route(stage23a_table, "no_ldn_proxy")
    no_aug = get_route(stage23a_table, "no_aug")
    no_dispersion = get_route(stage23a_table, "no_dispersion")
    stronger_neighbor = get_route(stage23b_table, "stronger_neighbor")
    stage22b = get_route(stage23b_table, "stage22b")

    diagnosis = {
        "stage": "Stage23C route diagnosis freeze",
        "stage22_freeze_status": stage22e["freeze_status"],
        "stage22_route_status": stage22e["recommended_status"],
        "stage22_mean_hmean": stage22e["mean_hmean"],
        "stage22_delta_hmean_vs_clean_best": stage22e["delta_hmean_vs_clean_best"],
        "stage22_delta_pAUC_vs_clean_best": stage22e["delta_pAUC_vs_clean_best"],
        "stage22_delta_TPR99_vs_clean_best": stage22e["delta_TPR99_vs_clean_best"],
        "single_component_culprit_rejected": True,
        "fan_density_mismatch_confirmed": bool(stage23b["all_routes_ldn_worse_than_adapted_baseline"]),
        "best_fan_baseline_route": stage23b["best_adapted_baseline_route"],
        "best_fan_baseline_hmean": stage23b["best_adapted_baseline_hmean"],
        "worst_ldn_gap_route": stage23b["worst_ldn_gap_route"],
        "worst_ldn_gap_hmean": stage23b["worst_ldn_gap_hmean"],
        "no_ldn_proxy_fan_recovery": f(no_ldn_proxy, "fan_delta_vs_stage22b_ldn"),
        "no_aug_fan_delta": f(no_aug, "fan_delta_vs_stage22b_ldn"),
        "no_dispersion_fan_delta": f(no_dispersion, "fan_delta_vs_stage22b_ldn"),
        "stronger_neighbor_baseline_hmean": f(stronger_neighbor, "baseline_hmean"),
        "stronger_neighbor_ldn_hmean": f(stronger_neighbor, "ldn_hmean"),
        "stronger_neighbor_ldn_minus_baseline": f(stronger_neighbor, "ldn_minus_baseline_hmean"),
        "stage22b_fan_ldn_minus_baseline": f(stage22b, "ldn_minus_baseline_hmean"),
        "diagnosis": "fan harm is a frozen-density recalibration mismatch under adapted geometry, not a single loss-component failure",
        "recommended_next_research_question": "Can a future objective preserve fan density ordering while keeping valve gain, without tuning LDN parameters or adding selectors?",
        "forbidden_followups": [
            "Do not tune LDN K/r/score_k.",
            "Do not add per-machine selector or fusion to hide fan degradation.",
            "Do not continue one-component ablations as if they identified a culprit.",
            "Do not increase neighbor/anchor blindly; stronger_neighbor raised fan baseline but worsened LDN.",
            "Do not claim Stage22 improves pAUC or TPR@99n.",
            "Do not use test labels for checkpoint selection.",
        ],
    }

    write_json(out_root / "stage23c_route_diagnosis_freeze_summary.json", diagnosis)
    write_csv(out_root / "stage23c_component_findings.csv", [
        {
            "component_probe": "no_ldn_proxy",
            "finding": "LDN proxy has only minor fan harm signal",
            "fan_delta_vs_stage22b_ldn": no_ldn_proxy["fan_delta_vs_stage22b_ldn"],
            "valve_delta_vs_stage22b_ldn": no_ldn_proxy["valve_delta_vs_stage22b_ldn"],
        },
        {
            "component_probe": "no_aug",
            "finding": "augmentation protects fan; removing it worsens fan",
            "fan_delta_vs_stage22b_ldn": no_aug["fan_delta_vs_stage22b_ldn"],
            "valve_delta_vs_stage22b_ldn": no_aug["valve_delta_vs_stage22b_ldn"],
        },
        {
            "component_probe": "no_dispersion",
            "finding": "dispersion guard is not the fan-harm driver",
            "fan_delta_vs_stage22b_ldn": no_dispersion["fan_delta_vs_stage22b_ldn"],
            "valve_delta_vs_stage22b_ldn": no_dispersion["valve_delta_vs_stage22b_ldn"],
        },
        {
            "component_probe": "stronger_neighbor",
            "finding": "fan adapted baseline improves, but frozen LDN mismatch worsens",
            "fan_baseline_hmean": stronger_neighbor["baseline_hmean"],
            "fan_ldn_hmean": stronger_neighbor["ldn_hmean"],
            "ldn_minus_baseline_hmean": stronger_neighbor["ldn_minus_baseline_hmean"],
        },
    ])

    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE23",
        "",
        "## 1. Stage22 status after Stage23 diagnosis",
        "",
        "Stage22D remains frozen as a weak hmean-positive route:",
        "",
        f"- freeze_status: `{stage22e['freeze_status']}`",
        f"- route: `{stage22e['route_description']}`",
        f"- mean_hmean: `{stage22e['mean_hmean']}`",
        f"- delta_hmean_vs_clean_best: `{stage22e['delta_hmean_vs_clean_best']}`",
        f"- delta_pAUC_vs_clean_best: `{stage22e['delta_pAUC_vs_clean_best']}`",
        f"- delta_TPR99_vs_clean_best: `{stage22e['delta_TPR99_vs_clean_best']}`",
        "",
        "Conclusion: Stage22D is a new hmean best only. Keep clean EAT + frozen LDN as the balanced reference.",
        "",
        "## 2. Stage23A component probe",
        "",
        "Single-component ablations did not identify a sufficient fan-harm culprit.",
        "",
        f"- no_ldn_proxy fan delta vs Stage22B LDN: `{no_ldn_proxy['fan_delta_vs_stage22b_ldn']}`",
        f"- no_aug fan delta vs Stage22B LDN: `{no_aug['fan_delta_vs_stage22b_ldn']}`",
        f"- no_dispersion fan delta vs Stage22B LDN: `{no_dispersion['fan_delta_vs_stage22b_ldn']}`",
        f"- candidate_for_full7_followup_count: `{stage23a['candidate_for_full7_followup_count']}`",
        "",
        "Interpretation: LDN proxy has a small fan-harm signal, augmentation protects fan, and dispersion is not the main driver.",
        "",
        "## 3. Stage23B fan density failure",
        "",
        "The fan failure is backend-density mismatch under adapted geometry:",
        "",
        f"- all routes LDN worse than adapted baseline: `{stage23b['all_routes_ldn_worse_than_adapted_baseline']}`",
        f"- best fan adapted baseline route: `{stage23b['best_adapted_baseline_route']}`",
        f"- best fan adapted baseline hmean: `{stage23b['best_adapted_baseline_hmean']}`",
        f"- worst LDN gap route: `{stage23b['worst_ldn_gap_route']}`",
        f"- worst LDN gap hmean: `{stage23b['worst_ldn_gap_hmean']}`",
        f"- stronger_neighbor baseline hmean: `{stronger_neighbor['baseline_hmean']}`",
        f"- stronger_neighbor LDN hmean: `{stronger_neighbor['ldn_hmean']}`",
        f"- stronger_neighbor LDN-baseline hmean: `{stronger_neighbor['ldn_minus_baseline_hmean']}`",
        "",
        "Key conclusion: stronger_neighbor improves fan baseline geometry, but frozen LDN subtraction amplifies the wrong fan tail geometry and destroys the score.",
        "",
        "## 4. Current diagnosis",
        "",
        "`fan harm is a frozen-density recalibration mismatch under adapted geometry, not a single loss-component failure.`",
        "",
        "## 5. Forbidden next moves",
        "",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not add per-machine selector or fusion to hide fan degradation.",
        "- Do not continue one-component ablations as if they identified a culprit.",
        "- Do not increase neighbor/anchor blindly; stronger_neighbor raised fan baseline but worsened LDN.",
        "- Do not claim Stage22 improves pAUC or TPR@99n.",
        "- Do not use test labels for checkpoint selection.",
        "",
        "## 6. Recommended next question",
        "",
        "If continuing, ask: can a future objective preserve fan density ordering while keeping valve gain, without tuning LDN parameters or adding selectors?",
    ])


if __name__ == "__main__":
    main()
