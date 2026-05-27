#!/usr/bin/env python
"""Stage29B: freeze Stage29A pseudo-attribute representation result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE29_FULL7_ROOT_DEFAULT = "results/stage29a_pseudo_attribute_full7_eval"
STAGE29_TRAIN_ROOT_DEFAULT = "results/stage29a_pseudo_attribute_representation"
STAGE27B_ROOT_DEFAULT = "results/stage27b_tfpatch_representation_freeze"
STAGE27_FULL7_ROOT_DEFAULT = "results/stage27a_tfpatch_representation_full7_eval"
OUTPUT_ROOT_DEFAULT = "results/stage29b_pseudo_attribute_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE29.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_ROUTE = "EAT min_clamp__mean + ldn_diff_K1_scorek5"
CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572
STAGE27_MEAN_HMEAN = 0.6168603949659727
STAGE27_MEAN_PAUC = 0.5624060150375939
STAGE27_MEAN_TPR99 = 0.11857142857142856


def parse_args():
    p = argparse.ArgumentParser(description="Stage29B pseudo-attribute freeze")
    p.add_argument("--stage29_full7_root", default=STAGE29_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage29_train_root", default=STAGE29_TRAIN_ROOT_DEFAULT)
    p.add_argument("--stage27b_root", default=STAGE27B_ROOT_DEFAULT)
    p.add_argument("--stage27_full7_root", default=STAGE27_FULL7_ROOT_DEFAULT)
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


def main():
    args = parse_args()
    full7_root = Path(args.stage29_full7_root)
    train_root = Path(args.stage29_train_root)
    stage27b_root = Path(args.stage27b_root)
    stage27_full7_root = Path(args.stage27_full7_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage29a_training_verdict.json")
    train_log = read_csv(train_root / "train_log.csv")
    pseudo_verdict_path = train_root / "pseudo_attribute_verdict.json"
    pseudo_verdict = read_json(pseudo_verdict_path) if pseudo_verdict_path.exists() else {}
    stage27_summary_path = stage27b_root / "stage27b_tfpatch_representation_freeze_summary.json"
    stage27_summary = read_json(stage27_summary_path) if stage27_summary_path.exists() else {}
    stage27_per_machine = read_csv(stage27_full7_root / "per_machine.csv")

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score_row = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base_row = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        stage27_row = get_row(stage27_per_machine, candidate=FROZEN_ROUTE, machine=machine)
        hmean = as_float(score_row, "hmean")
        pauc = as_float(score_row, "pAUC")
        t99 = as_float(score_row, "TPR@99n")
        s27_h = as_float(stage27_row, "hmean")
        s27_p = as_float(stage27_row, "pAUC")
        s27_t = as_float(stage27_row, "TPR@99n")
        machine_rows.append({
            "machine": machine,
            "stage29_baseline_hmean": as_float(base_row, "hmean"),
            "stage29_ldn_hmean": hmean,
            "stage29_ldn_pAUC": pauc,
            "stage29_ldn_TPR99": t99,
            "delta_hmean_vs_stage29_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "stage27_ldn_hmean": s27_h,
            "stage27_ldn_pAUC": s27_p,
            "stage27_ldn_TPR99": s27_t,
            "delta_hmean_vs_stage27_ldn": hmean - s27_h if hmean is not None and s27_h is not None else None,
            "delta_pAUC_vs_stage27_ldn": pauc - s27_p if pauc is not None and s27_p is not None else None,
            "delta_TPR99_vs_stage27_ldn": t99 - s27_t if t99 is not None and s27_t is not None else None,
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        })

    positive_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage29_baseline"] > 0]
    negative_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage29_baseline"] < 0]
    positive_vs_stage27 = [
        r for r in machine_rows
        if r["delta_hmean_vs_stage27_ldn"] is not None and r["delta_hmean_vs_stage27_ldn"] > 0
    ]
    negative_vs_stage27 = [
        r for r in machine_rows
        if r["delta_hmean_vs_stage27_ldn"] is not None and r["delta_hmean_vs_stage27_ldn"] < 0
    ]
    strongest_gain_vs_baseline = max(machine_rows, key=lambda r: r["delta_hmean_vs_stage29_baseline"])
    largest_drop_vs_baseline = min(machine_rows, key=lambda r: r["delta_hmean_vs_stage29_baseline"])
    strongest_gain_vs_stage27 = max(machine_rows, key=lambda r: r["delta_hmean_vs_stage27_ldn"] if r["delta_hmean_vs_stage27_ldn"] is not None else -999)
    largest_drop_vs_stage27 = min(machine_rows, key=lambda r: r["delta_hmean_vs_stage27_ldn"] if r["delta_hmean_vs_stage27_ldn"] is not None else 999)
    fan_row = get_row(machine_rows, machine="fan")

    s27_h = float(stage27_summary.get("mean_hmean", STAGE27_MEAN_HMEAN))
    s27_p = float(stage27_summary.get("mean_pAUC", STAGE27_MEAN_PAUC))
    s27_t = float(stage27_summary.get("mean_TPR99", STAGE27_MEAN_TPR99))
    delta_h_clean = verdict["mean_hmean"] - CLEAN_BEST_MEAN_HMEAN
    delta_p_clean = verdict["mean_pAUC"] - CLEAN_BEST_MEAN_PAUC
    delta_t_clean = verdict["mean_TPR99"] - CLEAN_BEST_MEAN_TPR99
    delta_h_stage27 = verdict["mean_hmean"] - s27_h
    delta_p_stage27 = verdict["mean_pAUC"] - s27_p
    delta_t_stage27 = verdict["mean_TPR99"] - s27_t
    balanced_positive = bool(
        verdict["promotable"]
        and delta_h_clean > 0
        and delta_p_clean > 0
        and delta_t_clean > 0
        and delta_h_stage27 > 0
        and delta_p_stage27 > 0
        and delta_t_stage27 > 0
        and not verdict["strong_machine_severe_collapse"]
    )
    freeze_status = "new_balanced_best_with_fan_regression" if balanced_positive else "not_new_balanced_best"

    last_train = train_log[-1] if train_log else {}
    summary = {
        "stage": "Stage29B pseudo-attribute representation freeze",
        "freeze_status": freeze_status,
        "route": FROZEN_ROUTE,
        "route_description": "Stage29 pseudo-attribute representation-adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "previous_balanced_best_route": "Stage27 TF-patch representation-adapted EAT + frozen ldn_diff_K1_scorek5",
        "best_checkpoint": train_verdict.get("best_checkpoint"),
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "num_pseudo_attributes": train_verdict.get("num_pseudo_attributes", pseudo_verdict.get("pseudo_attribute_count")),
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "delta_hmean_vs_clean_best": delta_h_clean,
        "delta_pAUC_vs_clean_best": delta_p_clean,
        "delta_TPR99_vs_clean_best": delta_t_clean,
        "delta_hmean_vs_stage27_balanced_best": delta_h_stage27,
        "delta_pAUC_vs_stage27_balanced_best": delta_p_stage27,
        "delta_TPR99_vs_stage27_balanced_best": delta_t_stage27,
        "improved_machine_count_vs_stage29_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_stage29_baseline": [r["machine"] for r in positive_vs_baseline],
        "negative_machines_vs_stage29_baseline": [r["machine"] for r in negative_vs_baseline],
        "positive_machines_vs_stage27_ldn": [r["machine"] for r in positive_vs_stage27],
        "negative_machines_vs_stage27_ldn": [r["machine"] for r in negative_vs_stage27],
        "strongest_gain_machine_vs_stage29_baseline": strongest_gain_vs_baseline["machine"],
        "strongest_gain_delta_hmean_vs_stage29_baseline": strongest_gain_vs_baseline["delta_hmean_vs_stage29_baseline"],
        "largest_drop_machine_vs_stage29_baseline": largest_drop_vs_baseline["machine"],
        "largest_drop_delta_hmean_vs_stage29_baseline": largest_drop_vs_baseline["delta_hmean_vs_stage29_baseline"],
        "strongest_gain_machine_vs_stage27_ldn": strongest_gain_vs_stage27["machine"],
        "strongest_gain_delta_hmean_vs_stage27_ldn": strongest_gain_vs_stage27["delta_hmean_vs_stage27_ldn"],
        "largest_drop_machine_vs_stage27_ldn": largest_drop_vs_stage27["machine"],
        "largest_drop_delta_hmean_vs_stage27_ldn": largest_drop_vs_stage27["delta_hmean_vs_stage27_ldn"],
        "fan_ldn_minus_baseline_hmean": fan_row["delta_hmean_vs_stage29_baseline"] if fan_row else None,
        "fan_delta_hmean_vs_stage27_ldn": fan_row["delta_hmean_vs_stage27_ldn"] if fan_row else None,
        "fan_regressed_vs_stage27": bool(fan_row and fan_row["delta_hmean_vs_stage27_ldn"] is not None and fan_row["delta_hmean_vs_stage27_ldn"] < -0.005),
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "uses_anomaly_labels": train_verdict.get("uses_anomaly_labels"),
        "uses_test_labels_for_checkpoint_selection": train_verdict.get("uses_test_labels_for_checkpoint_selection"),
        "ldn_parameters_tuned": train_verdict.get("ldn_parameters_tuned"),
        "last_epoch_train_acc": last_train.get("train_acc"),
        "last_epoch_val_acc": last_train.get("val_acc"),
        "last_epoch_train_anchor_loss": last_train.get("train_anchor_loss"),
        "last_epoch_val_anchor_loss": last_train.get("val_anchor_loss"),
        "recommended_status": "new balanced best; pseudo-attribute objective is promising, but fan regression and baseline geometry drift remain open",
        "forbidden_followups": [
            "Do not tune LDN K/r/score_k.",
            "Do not add selector or fusion to hide fan degradation.",
            "Do not select checkpoints with test labels.",
            "Do not claim fan density mismatch is solved.",
            "Do not chase pseudo-attribute training accuracy without monitoring frozen backend full7.",
        ],
    }

    write_json(out_root / "stage29b_pseudo_attribute_freeze_summary.json", summary)
    write_csv(out_root / "stage29b_machine_delta_table.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE29",
        "",
        "## 1. Stage29A result",
        "",
        "Stage29A trained a pseudo-attribute/domain-adaptive representation objective using source train-normal data only. Pseudo attributes were KMeans clusters over clean EAT `min_clamp__mean` embeddings within each machine.",
        "",
        f"- route: `{FROZEN_ROUTE}`",
        f"- route description: `{summary['route_description']}`",
        f"- best checkpoint: `{summary['best_checkpoint']}`",
        f"- best epoch: `{summary['best_epoch']}`",
        f"- pseudo attributes: `{summary['num_pseudo_attributes']}`",
        f"- mean_hmean: `{summary['mean_hmean']}`",
        f"- mean_pAUC: `{summary['mean_pAUC']}`",
        f"- mean_TPR@99n: `{summary['mean_TPR99']}`",
        f"- delta_hmean_vs_clean_best: `{summary['delta_hmean_vs_clean_best']}`",
        f"- delta_pAUC_vs_clean_best: `{summary['delta_pAUC_vs_clean_best']}`",
        f"- delta_TPR99_vs_clean_best: `{summary['delta_TPR99_vs_clean_best']}`",
        f"- delta_hmean_vs_stage27_balanced_best: `{summary['delta_hmean_vs_stage27_balanced_best']}`",
        f"- delta_pAUC_vs_stage27_balanced_best: `{summary['delta_pAUC_vs_stage27_balanced_best']}`",
        f"- delta_TPR99_vs_stage27_balanced_best: `{summary['delta_TPR99_vs_stage27_balanced_best']}`",
        f"- freeze_status: `{freeze_status}`",
        "",
        "Conclusion: Stage29A is the new balanced best. It improves hmean, pAUC, and TPR@99n over both the clean balanced route and Stage27.",
        "",
        "## 2. Machine-level pattern",
        "",
        f"- positive machines vs Stage29 baseline: `{', '.join(summary['positive_machines_vs_stage29_baseline'])}`",
        f"- negative machines vs Stage29 baseline: `{', '.join(summary['negative_machines_vs_stage29_baseline'])}`",
        f"- positive machines vs Stage27 LDN: `{', '.join(summary['positive_machines_vs_stage27_ldn'])}`",
        f"- negative machines vs Stage27 LDN: `{', '.join(summary['negative_machines_vs_stage27_ldn'])}`",
        f"- strongest gain vs Stage27 LDN: `{summary['strongest_gain_machine_vs_stage27_ldn']} ({summary['strongest_gain_delta_hmean_vs_stage27_ldn']:+.12f})`",
        f"- largest drop vs Stage27 LDN: `{summary['largest_drop_machine_vs_stage27_ldn']} ({summary['largest_drop_delta_hmean_vs_stage27_ldn']:+.12f})`",
        f"- fan LDN minus Stage29 baseline hmean: `{summary['fan_ldn_minus_baseline_hmean']}`",
        f"- fan delta hmean vs Stage27 LDN: `{summary['fan_delta_hmean_vs_stage27_ldn']}`",
        "",
        "Fan remains unresolved and regresses versus Stage27. The global win is carried by strong gains in bearing, gearbox, slider, toycar, and especially valve/pAUC.",
        "",
        "## 3. Training diagnostic",
        "",
        f"- last train acc: `{summary['last_epoch_train_acc']}`",
        f"- last val acc: `{summary['last_epoch_val_acc']}`",
        f"- last train anchor loss: `{summary['last_epoch_train_anchor_loss']}`",
        f"- last val anchor loss: `{summary['last_epoch_val_anchor_loss']}`",
        "",
        "Pseudo-attribute classification is learned, but the anchor loss grows substantially, indicating real representation movement. Future variants should control this drift while preserving the pAUC/TPR gains.",
        "",
        "## 4. Current route hierarchy",
        "",
        "- balanced best: `Stage29 pseudo-attribute representation-adapted EAT + frozen ldn_diff_K1_scorek5`",
        "- previous balanced best: `Stage27 TF-patch representation-adapted EAT + frozen ldn_diff_K1_scorek5`",
        "- old hmean-only reference: `Stage22 geometry-objective adapted EAT + frozen ldn_diff_K1_scorek5`",
        "",
        "## 5. Recommended next step",
        "",
        "Continue Stage29, but regularize representation drift. The safest next variant is not more clusters or more epochs; it is a guardrailed pseudo-attribute objective with stronger anchor and/or a geometry-preservation term, then full7 frozen eval.",
        "",
        "## 6. Forbidden next moves",
        "",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not add per-machine selector or fusion to hide fan degradation.",
        "- Do not use test labels for checkpoint selection.",
        "- Do not claim fan density mismatch is solved.",
        "- Do not chase pseudo-attribute training accuracy without monitoring frozen backend full7.",
    ])


if __name__ == "__main__":
    main()
