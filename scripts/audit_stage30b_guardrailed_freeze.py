#!/usr/bin/env python
"""Stage30B: freeze Stage30A guardrailed pseudo-attribute result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE30_FULL7_ROOT_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute_full7_eval"
STAGE30_TRAIN_ROOT_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute"
STAGE29B_ROOT_DEFAULT = "results/stage29b_pseudo_attribute_freeze"
STAGE29_FULL7_ROOT_DEFAULT = "results/stage29a_pseudo_attribute_full7_eval"
STAGE27B_ROOT_DEFAULT = "results/stage27b_tfpatch_representation_freeze"
OUTPUT_ROOT_DEFAULT = "results/stage30b_guardrailed_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE30.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572
STAGE27_MEAN_HMEAN = 0.6168603949659727
STAGE27_MEAN_PAUC = 0.5624060150375939
STAGE27_MEAN_TPR99 = 0.11857142857142856
STAGE29_MEAN_HMEAN = 0.6171391941988533
STAGE29_MEAN_PAUC = 0.5724060150375939
STAGE29_MEAN_TPR99 = 0.12142857142857143
STAGE29_VAL_ANCHOR = 0.83134942405162


def parse_args():
    p = argparse.ArgumentParser(description="Stage30B guardrailed pseudo-attribute freeze")
    p.add_argument("--stage30_full7_root", default=STAGE30_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage30_train_root", default=STAGE30_TRAIN_ROOT_DEFAULT)
    p.add_argument("--stage29b_root", default=STAGE29B_ROOT_DEFAULT)
    p.add_argument("--stage29_full7_root", default=STAGE29_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage27b_root", default=STAGE27B_ROOT_DEFAULT)
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


def last_float(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    value = rows[-1].get(key)
    if value in (None, ""):
        return None
    return float(value)


def main():
    args = parse_args()
    full7_root = Path(args.stage30_full7_root)
    train_root = Path(args.stage30_train_root)
    stage29b_root = Path(args.stage29b_root)
    stage29_full7_root = Path(args.stage29_full7_root)
    stage27b_root = Path(args.stage27b_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage30a_training_verdict.json")
    train_log = read_csv(train_root / "train_log.csv")
    stage29_summary_path = stage29b_root / "stage29b_pseudo_attribute_freeze_summary.json"
    stage29_summary = read_json(stage29_summary_path) if stage29_summary_path.exists() else {}
    stage27_summary_path = stage27b_root / "stage27b_tfpatch_representation_freeze_summary.json"
    stage27_summary = read_json(stage27_summary_path) if stage27_summary_path.exists() else {}
    stage29_per_machine = read_csv(stage29_full7_root / "per_machine.csv")

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score_row = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base_row = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        stage29_row = get_row(stage29_per_machine, candidate=FROZEN_ROUTE, machine=machine)
        hmean = as_float(score_row, "hmean")
        pauc = as_float(score_row, "pAUC")
        t99 = as_float(score_row, "TPR@99n")
        s29_h = as_float(stage29_row, "hmean")
        s29_p = as_float(stage29_row, "pAUC")
        s29_t = as_float(stage29_row, "TPR@99n")
        machine_rows.append({
            "machine": machine,
            "stage30_baseline_hmean": as_float(base_row, "hmean"),
            "stage30_ldn_hmean": hmean,
            "stage30_ldn_pAUC": pauc,
            "stage30_ldn_TPR99": t99,
            "delta_hmean_vs_stage30_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "stage29_ldn_hmean": s29_h,
            "stage29_ldn_pAUC": s29_p,
            "stage29_ldn_TPR99": s29_t,
            "delta_hmean_vs_stage29_ldn": hmean - s29_h if hmean is not None and s29_h is not None else None,
            "delta_pAUC_vs_stage29_ldn": pauc - s29_p if pauc is not None and s29_p is not None else None,
            "delta_TPR99_vs_stage29_ldn": t99 - s29_t if t99 is not None and s29_t is not None else None,
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        })

    positive_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage30_baseline"] > 0]
    negative_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage30_baseline"] < 0]
    positive_vs_stage29 = [
        r for r in machine_rows
        if r["delta_hmean_vs_stage29_ldn"] is not None and r["delta_hmean_vs_stage29_ldn"] > 0
    ]
    negative_vs_stage29 = [
        r for r in machine_rows
        if r["delta_hmean_vs_stage29_ldn"] is not None and r["delta_hmean_vs_stage29_ldn"] < 0
    ]
    strongest_gain_vs_baseline = max(machine_rows, key=lambda r: r["delta_hmean_vs_stage30_baseline"])
    largest_drop_vs_baseline = min(machine_rows, key=lambda r: r["delta_hmean_vs_stage30_baseline"])
    strongest_gain_vs_stage29 = max(machine_rows, key=lambda r: r["delta_hmean_vs_stage29_ldn"] if r["delta_hmean_vs_stage29_ldn"] is not None else -999)
    largest_drop_vs_stage29 = min(machine_rows, key=lambda r: r["delta_hmean_vs_stage29_ldn"] if r["delta_hmean_vs_stage29_ldn"] is not None else 999)
    fan_row = get_row(machine_rows, machine="fan")

    s29_h = float(stage29_summary.get("mean_hmean", STAGE29_MEAN_HMEAN))
    s29_p = float(stage29_summary.get("mean_pAUC", STAGE29_MEAN_PAUC))
    s29_t = float(stage29_summary.get("mean_TPR99", STAGE29_MEAN_TPR99))
    s27_h = float(stage27_summary.get("mean_hmean", STAGE27_MEAN_HMEAN))
    s27_p = float(stage27_summary.get("mean_pAUC", STAGE27_MEAN_PAUC))
    s27_t = float(stage27_summary.get("mean_TPR99", STAGE27_MEAN_TPR99))

    delta_h_clean = verdict["mean_hmean"] - CLEAN_BEST_MEAN_HMEAN
    delta_p_clean = verdict["mean_pAUC"] - CLEAN_BEST_MEAN_PAUC
    delta_t_clean = verdict["mean_TPR99"] - CLEAN_BEST_MEAN_TPR99
    delta_h_stage29 = verdict["mean_hmean"] - s29_h
    delta_p_stage29 = verdict["mean_pAUC"] - s29_p
    delta_t_stage29 = verdict["mean_TPR99"] - s29_t
    delta_h_stage27 = verdict["mean_hmean"] - s27_h
    delta_p_stage27 = verdict["mean_pAUC"] - s27_p
    delta_t_stage27 = verdict["mean_TPR99"] - s27_t

    val_anchor = last_float(train_log, "val_anchor_loss")
    val_pairwise = last_float(train_log, "val_pairwise_geometry_loss")
    val_center = last_float(train_log, "val_machine_center_loss")
    val_acc = last_float(train_log, "val_acc")
    anchor_reduction_vs_stage29 = STAGE29_VAL_ANCHOR - val_anchor if val_anchor is not None else None
    new_hmean_best = bool(
        verdict["promotable"]
        and delta_h_stage29 > 0
        and not verdict["strong_machine_severe_collapse"]
    )
    freeze_status = "new_hmean_best_with_pauc_tpr_tradeoff" if new_hmean_best and (delta_p_stage29 < 0 or delta_t_stage29 < 0) else "not_new_hmean_best"

    summary = {
        "stage": "Stage30B guardrailed pseudo-attribute freeze",
        "freeze_status": freeze_status,
        "route": FROZEN_ROUTE,
        "route_description": "Stage30 guardrailed pseudo-attribute EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "previous_balanced_best_route": "Stage29 pseudo-attribute representation-adapted EAT + frozen ldn_diff_K1_scorek5",
        "best_checkpoint": train_verdict.get("best_checkpoint"),
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "num_pseudo_attributes": train_verdict.get("num_pseudo_attributes"),
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "delta_hmean_vs_clean_best": delta_h_clean,
        "delta_pAUC_vs_clean_best": delta_p_clean,
        "delta_TPR99_vs_clean_best": delta_t_clean,
        "delta_hmean_vs_stage29": delta_h_stage29,
        "delta_pAUC_vs_stage29": delta_p_stage29,
        "delta_TPR99_vs_stage29": delta_t_stage29,
        "delta_hmean_vs_stage27": delta_h_stage27,
        "delta_pAUC_vs_stage27": delta_p_stage27,
        "delta_TPR99_vs_stage27": delta_t_stage27,
        "improved_machine_count_vs_stage30_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_stage30_baseline": [r["machine"] for r in positive_vs_baseline],
        "negative_machines_vs_stage30_baseline": [r["machine"] for r in negative_vs_baseline],
        "positive_machines_vs_stage29_ldn": [r["machine"] for r in positive_vs_stage29],
        "negative_machines_vs_stage29_ldn": [r["machine"] for r in negative_vs_stage29],
        "strongest_gain_machine_vs_stage30_baseline": strongest_gain_vs_baseline["machine"],
        "strongest_gain_delta_hmean_vs_stage30_baseline": strongest_gain_vs_baseline["delta_hmean_vs_stage30_baseline"],
        "largest_drop_machine_vs_stage30_baseline": largest_drop_vs_baseline["machine"],
        "largest_drop_delta_hmean_vs_stage30_baseline": largest_drop_vs_baseline["delta_hmean_vs_stage30_baseline"],
        "strongest_gain_machine_vs_stage29_ldn": strongest_gain_vs_stage29["machine"],
        "strongest_gain_delta_hmean_vs_stage29_ldn": strongest_gain_vs_stage29["delta_hmean_vs_stage29_ldn"],
        "largest_drop_machine_vs_stage29_ldn": largest_drop_vs_stage29["machine"],
        "largest_drop_delta_hmean_vs_stage29_ldn": largest_drop_vs_stage29["delta_hmean_vs_stage29_ldn"],
        "fan_ldn_minus_baseline_hmean": fan_row["delta_hmean_vs_stage30_baseline"] if fan_row else None,
        "fan_delta_hmean_vs_stage29_ldn": fan_row["delta_hmean_vs_stage29_ldn"] if fan_row else None,
        "fan_recovered_vs_stage29": bool(fan_row and fan_row["delta_hmean_vs_stage29_ldn"] is not None and fan_row["delta_hmean_vs_stage29_ldn"] > 0.005),
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "uses_anomaly_labels": train_verdict.get("uses_anomaly_labels"),
        "uses_test_labels_for_checkpoint_selection": train_verdict.get("uses_test_labels_for_checkpoint_selection"),
        "ldn_parameters_tuned": train_verdict.get("ldn_parameters_tuned"),
        "last_epoch_val_acc": val_acc,
        "last_epoch_val_anchor_loss": val_anchor,
        "last_epoch_val_pairwise_geometry_loss": val_pairwise,
        "last_epoch_val_machine_center_loss": val_center,
        "anchor_loss_reduction_vs_stage29": anchor_reduction_vs_stage29,
        "recommended_status": "new hmean best; guardrails recover fan and control drift, but Stage29 keeps slightly better pAUC and better TPR99",
        "forbidden_followups": [
            "Do not tune LDN K/r/score_k.",
            "Do not add selector or fusion to choose Stage29/Stage30 per machine.",
            "Do not select checkpoints with test labels.",
            "Do not call Stage30 a TPR@99n improvement over Stage29.",
            "Do not chase pseudo-attribute training accuracy; guardrail quality is the point.",
        ],
    }

    write_json(out_root / "stage30b_guardrailed_freeze_summary.json", summary)
    write_csv(out_root / "stage30b_machine_delta_table.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE30",
        "",
        "## 1. Stage30A result",
        "",
        "Stage30A reused Stage29 pseudo attributes and added geometry guardrails: stronger teacher anchor, batch pairwise geometry preservation, and same-machine center preservation.",
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
        f"- delta_hmean_vs_stage29: `{summary['delta_hmean_vs_stage29']}`",
        f"- delta_pAUC_vs_stage29: `{summary['delta_pAUC_vs_stage29']}`",
        f"- delta_TPR99_vs_stage29: `{summary['delta_TPR99_vs_stage29']}`",
        f"- freeze_status: `{freeze_status}`",
        "",
        "Conclusion: Stage30A is the new hmean best. It recovers fan versus Stage29 and strongly controls representation drift, but it gives back a small amount of pAUC and a larger amount of Stage29's TPR@99n gain.",
        "",
        "## 2. Guardrail diagnostic",
        "",
        f"- Stage30 final val anchor loss: `{summary['last_epoch_val_anchor_loss']}`",
        f"- Stage29 final val anchor loss reference: `{STAGE29_VAL_ANCHOR}`",
        f"- anchor loss reduction vs Stage29: `{summary['anchor_loss_reduction_vs_stage29']}`",
        f"- Stage30 final val pairwise geometry loss: `{summary['last_epoch_val_pairwise_geometry_loss']}`",
        f"- Stage30 final val machine-center loss: `{summary['last_epoch_val_machine_center_loss']}`",
        f"- Stage30 final val pseudo-attribute acc: `{summary['last_epoch_val_acc']}`",
        "",
        "The guardrails worked: pseudo-attribute accuracy stayed very low, but backend metrics improved. This suggests full pseudo-attribute separability is not required; controlled representation shaping is the useful signal.",
        "",
        "## 3. Machine-level pattern",
        "",
        f"- positive machines vs Stage30 baseline: `{', '.join(summary['positive_machines_vs_stage30_baseline'])}`",
        f"- negative machines vs Stage30 baseline: `{', '.join(summary['negative_machines_vs_stage30_baseline'])}`",
        f"- positive machines vs Stage29 LDN: `{', '.join(summary['positive_machines_vs_stage29_ldn'])}`",
        f"- negative machines vs Stage29 LDN: `{', '.join(summary['negative_machines_vs_stage29_ldn'])}`",
        f"- fan LDN minus Stage30 baseline hmean: `{summary['fan_ldn_minus_baseline_hmean']}`",
        f"- fan delta hmean vs Stage29 LDN: `{summary['fan_delta_hmean_vs_stage29_ldn']}`",
        "",
        "Fan recovered versus Stage29 and is close to Stage27, but LDN is still below Stage30's own fan baseline.",
        "",
        "## 4. Current route hierarchy",
        "",
        "- hmean best: `Stage30 guardrailed pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- pAUC/TPR@99n best in this branch: `Stage29 pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- previous balanced reference: `Stage27 TF-patch representation EAT + frozen ldn_diff_K1_scorek5`",
        "",
        "## 5. Recommended next step",
        "",
        "Continue guardrailed pseudo-attribute, but recover TPR@99n. The next variant should slightly relax guardrails or reduce anchor weight only if fan does not fall back toward Stage29.",
        "",
        "## 6. Forbidden next moves",
        "",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not add selector or fusion to choose Stage29/Stage30 per machine.",
        "- Do not use test labels for checkpoint selection.",
        "- Do not call Stage30 a TPR@99n improvement over Stage29.",
        "- Do not chase pseudo-attribute training accuracy; guardrail quality is the point.",
    ])


if __name__ == "__main__":
    main()
