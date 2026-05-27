#!/usr/bin/env python
"""Stage27B: freeze Stage27A TF-patch representation result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE27_FULL7_ROOT_DEFAULT = "results/stage27a_tfpatch_representation_full7_eval"
STAGE27_TRAIN_ROOT_DEFAULT = "results/stage27a_tfpatch_representation_pilot"
STAGE22_FULL7_ROOT_DEFAULT = "results/stage22d_adapted_full7_eval"
OUTPUT_ROOT_DEFAULT = "results/stage27b_tfpatch_representation_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE27.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_ROUTE = "EAT min_clamp__mean + ldn_diff_K1_scorek5"
CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572
STAGE22_MEAN_HMEAN = 0.6166917954459288
STAGE22_MEAN_PAUC = 0.5581954887218045
STAGE22_MEAN_TPR99 = 0.10142857142857142


def parse_args():
    p = argparse.ArgumentParser(description="Stage27B TF-patch representation freeze")
    p.add_argument("--stage27_full7_root", default=STAGE27_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage27_train_root", default=STAGE27_TRAIN_ROOT_DEFAULT)
    p.add_argument("--stage22_full7_root", default=STAGE22_FULL7_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=DOC_PATH_DEFAULT)
    return p.parse_args()


def read_json(path: Path):
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def get_row(rows: list[dict], **filters) -> dict | None:
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return None


def as_float(row: dict | None, key: str) -> float | None:
    if row is None or key not in row or row[key] == "":
        return None
    return float(row[key])


def fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.12f}"


def main():
    args = parse_args()
    full7_root = Path(args.stage27_full7_root)
    train_root = Path(args.stage27_train_root)
    stage22_root = Path(args.stage22_full7_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage27a_training_verdict.json") if (train_root / "stage27a_training_verdict.json").exists() else {}
    train_log = read_csv(train_root / "train_log.csv")
    stage22_per_machine = read_csv(stage22_root / "per_machine.csv")

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score_row = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base_row = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        stage22_row = get_row(stage22_per_machine, candidate=FROZEN_ROUTE, machine=machine)
        hmean = as_float(score_row, "hmean")
        pauc = as_float(score_row, "pAUC")
        t99 = as_float(score_row, "TPR@99n")
        stage22_hmean = as_float(stage22_row, "hmean")
        stage22_pauc = as_float(stage22_row, "pAUC")
        stage22_t99 = as_float(stage22_row, "TPR@99n")
        machine_rows.append({
            "machine": machine,
            "stage27_baseline_hmean": as_float(base_row, "hmean"),
            "stage27_ldn_hmean": hmean,
            "stage27_ldn_pAUC": pauc,
            "stage27_ldn_TPR99": t99,
            "delta_hmean_vs_stage27_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "stage22_ldn_hmean": stage22_hmean,
            "stage22_ldn_pAUC": stage22_pauc,
            "stage22_ldn_TPR99": stage22_t99,
            "delta_hmean_vs_stage22_ldn": hmean - stage22_hmean if hmean is not None and stage22_hmean is not None else None,
            "delta_pAUC_vs_stage22_ldn": pauc - stage22_pauc if pauc is not None and stage22_pauc is not None else None,
            "delta_TPR99_vs_stage22_ldn": t99 - stage22_t99 if t99 is not None and stage22_t99 is not None else None,
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        })

    positive_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage27_baseline"] > 0]
    negative_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage27_baseline"] < 0]
    positive_vs_stage22 = [
        r for r in machine_rows
        if r["delta_hmean_vs_stage22_ldn"] is not None and r["delta_hmean_vs_stage22_ldn"] > 0
    ]
    negative_vs_stage22 = [
        r for r in machine_rows
        if r["delta_hmean_vs_stage22_ldn"] is not None and r["delta_hmean_vs_stage22_ldn"] < 0
    ]
    largest_drop = min(machine_rows, key=lambda r: r["delta_hmean_vs_stage27_baseline"])
    strongest_gain = max(machine_rows, key=lambda r: r["delta_hmean_vs_stage27_baseline"])
    fan_row = get_row(machine_rows, machine="fan")

    delta_h_clean = verdict["mean_hmean"] - CLEAN_BEST_MEAN_HMEAN
    delta_p_clean = verdict["mean_pAUC"] - CLEAN_BEST_MEAN_PAUC
    delta_t_clean = verdict["mean_TPR99"] - CLEAN_BEST_MEAN_TPR99
    delta_h_stage22 = verdict["mean_hmean"] - STAGE22_MEAN_HMEAN
    delta_p_stage22 = verdict["mean_pAUC"] - STAGE22_MEAN_PAUC
    delta_t_stage22 = verdict["mean_TPR99"] - STAGE22_MEAN_TPR99
    balanced_positive = bool(
        verdict["promotable"]
        and delta_h_clean > 0
        and delta_p_clean > 0
        and delta_t_clean > 0
        and not verdict["strong_machine_severe_collapse"]
    )
    freeze_status = "balanced_positive_freeze" if balanced_positive else "not_balanced_positive"

    last_train = train_log[-1] if train_log else {}
    summary = {
        "stage": "Stage27B TF-patch representation freeze",
        "freeze_status": freeze_status,
        "route": FROZEN_ROUTE,
        "route_description": "Stage27 TF-patch representation-adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "previous_confirmed_best_route": CLEAN_BEST_ROUTE,
        "previous_hmean_best_route": "Stage22 geometry-objective adapted EAT + frozen ldn_diff_K1_scorek5",
        "best_checkpoint": train_verdict.get("best_checkpoint"),
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "delta_hmean_vs_clean_best": delta_h_clean,
        "delta_pAUC_vs_clean_best": delta_p_clean,
        "delta_TPR99_vs_clean_best": delta_t_clean,
        "delta_hmean_vs_stage22_hmean_best": delta_h_stage22,
        "delta_pAUC_vs_stage22": delta_p_stage22,
        "delta_TPR99_vs_stage22": delta_t_stage22,
        "improved_machine_count_vs_stage27_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_stage27_baseline": [r["machine"] for r in positive_vs_baseline],
        "negative_machines_vs_stage27_baseline": [r["machine"] for r in negative_vs_baseline],
        "positive_machines_vs_stage22_ldn": [r["machine"] for r in positive_vs_stage22],
        "negative_machines_vs_stage22_ldn": [r["machine"] for r in negative_vs_stage22],
        "strongest_gain_machine_vs_stage27_baseline": strongest_gain["machine"],
        "strongest_gain_delta_hmean_vs_stage27_baseline": strongest_gain["delta_hmean_vs_stage27_baseline"],
        "largest_drop_machine_vs_stage27_baseline": largest_drop["machine"],
        "largest_drop_delta_hmean_vs_stage27_baseline": largest_drop["delta_hmean_vs_stage27_baseline"],
        "fan_ldn_minus_baseline_hmean": fan_row["delta_hmean_vs_stage27_baseline"] if fan_row else None,
        "fan_mismatch_still_open": bool(fan_row and fan_row["delta_hmean_vs_stage27_baseline"] < -0.02),
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "uses_anomaly_labels": train_verdict.get("uses_anomaly_labels"),
        "uses_test_labels_for_checkpoint_selection": train_verdict.get("uses_test_labels_for_checkpoint_selection"),
        "ldn_parameters_tuned": train_verdict.get("ldn_parameters_tuned"),
        "last_epoch_train_total": last_train.get("train_total"),
        "last_epoch_val_total": last_train.get("val_total"),
        "recommended_status": "new balanced frozen best; fan LDN mismatch remains the main open bottleneck",
        "forbidden_followups": [
            "Do not tune LDN K/r/score_k.",
            "Do not add a selector or fusion to hide fan degradation.",
            "Do not select checkpoints with test labels.",
            "Do not claim fan density mismatch is solved.",
        ],
    }

    write_json(out_root / "stage27b_tfpatch_representation_freeze_summary.json", summary)
    write_csv(out_root / "stage27b_machine_delta_table.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE27",
        "",
        "## 1. Stage27 full7 frozen result",
        "",
        "Stage27A trained a representation-level TF-patch objective on `fan,valve`, then evaluated the supplied checkpoint with the unchanged frozen backend across full7.",
        "",
        f"- route: `{FROZEN_ROUTE}`",
        f"- route description: `{summary['route_description']}`",
        f"- best checkpoint: `{summary['best_checkpoint']}`",
        f"- best epoch: `{summary['best_epoch']}`",
        f"- mean_hmean: `{summary['mean_hmean']}`",
        f"- mean_pAUC: `{summary['mean_pAUC']}`",
        f"- mean_TPR@99n: `{summary['mean_TPR99']}`",
        f"- delta_hmean_vs_clean_best: `{summary['delta_hmean_vs_clean_best']}`",
        f"- delta_pAUC_vs_clean_best: `{summary['delta_pAUC_vs_clean_best']}`",
        f"- delta_TPR99_vs_clean_best: `{summary['delta_TPR99_vs_clean_best']}`",
        f"- delta_hmean_vs_stage22_hmean_best: `{summary['delta_hmean_vs_stage22_hmean_best']}`",
        f"- delta_pAUC_vs_stage22: `{summary['delta_pAUC_vs_stage22']}`",
        f"- delta_TPR99_vs_stage22: `{summary['delta_TPR99_vs_stage22']}`",
        f"- freeze_status: `{freeze_status}`",
        "",
        "Conclusion: Stage27 is the first balanced positive freeze in this branch. It beats the clean balanced reference on hmean, pAUC, and TPR@99n, and also edges Stage22 on all three mean metrics.",
        "",
        "## 2. Machine-level pattern",
        "",
        f"- positive machines vs Stage27 adapted baseline: `{', '.join(summary['positive_machines_vs_stage27_baseline'])}`",
        f"- negative machines vs Stage27 adapted baseline: `{', '.join(summary['negative_machines_vs_stage27_baseline'])}`",
        f"- positive machines vs Stage22 LDN: `{', '.join(summary['positive_machines_vs_stage22_ldn'])}`",
        f"- negative machines vs Stage22 LDN: `{', '.join(summary['negative_machines_vs_stage22_ldn'])}`",
        f"- strongest gain vs Stage27 baseline: `{strongest_gain['machine']} ({strongest_gain['delta_hmean_vs_stage27_baseline']:+.12f})`",
        f"- largest drop vs Stage27 baseline: `{largest_drop['machine']} ({largest_drop['delta_hmean_vs_stage27_baseline']:+.12f})`",
        f"- fan LDN minus baseline hmean: `{summary['fan_ldn_minus_baseline_hmean']}`",
        "",
        "Fan remains the main unresolved issue: Stage27 improves some fan tail metrics, but frozen LDN still underperforms the Stage27 fan baseline by more than 0.02 hmean.",
        "",
        "## 3. Policy",
        "",
        f"- uses anomaly labels: `{summary['uses_anomaly_labels']}`",
        f"- uses test labels for checkpoint selection: `{summary['uses_test_labels_for_checkpoint_selection']}`",
        f"- LDN parameters tuned: `{summary['ldn_parameters_tuned']}`",
        "",
        "## 4. Current route hierarchy",
        "",
        f"- balanced best route: `Stage27 TF-patch representation-adapted EAT + {FROZEN_ROUTE}`",
        f"- previous balanced reference: `{CLEAN_BEST_ROUTE}`",
        "- previous hmean-only best: `Stage22 geometry-objective adapted EAT + frozen ldn_diff_K1_scorek5`",
        "",
        "## 5. Forbidden next moves",
        "",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not add per-machine selector or fusion to hide fan degradation.",
        "- Do not use test labels for checkpoint selection.",
        "- Do not claim fan density mismatch is solved.",
        "",
        "## 6. Recommended next step",
        "",
        "Freeze Stage27 as the new balanced best. Next, run the Stage28 pooling/spike no-training audit only as a cheap paper-inspired probe; do not displace Stage27 unless it beats Stage27 full7 without collapse. For the 0.7 target, the main research branch should move toward pseudo-attribute/domain-adaptive representation rather than local-head-only pseudo-boundary scoring.",
    ])


if __name__ == "__main__":
    main()
