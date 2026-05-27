#!/usr/bin/env python
"""Stage35B: freeze Stage35A scheduled mild-tail result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE35_FULL7_ROOT_DEFAULT = "results/stage35a_scheduled_tail_guardrail_full7_eval"
STAGE35_TRAIN_ROOT_DEFAULT = "results/stage35a_scheduled_tail_guardrail"
STAGE34_FREEZE_DEFAULT = "results/stage34b_tail_guardrailed_freeze/stage34b_tail_guardrailed_freeze_summary.json"
STAGE34_FULL7_DEFAULT = "results/stage34a_tail_guardrailed_pseudo_attribute_full7_eval"
STAGE30_FREEZE_DEFAULT = "results/stage30b_guardrailed_freeze/stage30b_guardrailed_freeze_summary.json"
STAGE30_FULL7_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute_full7_eval"
STAGE29_FREEZE_DEFAULT = "results/stage29b_pseudo_attribute_freeze/stage29b_pseudo_attribute_freeze_summary.json"
STAGE29_FULL7_DEFAULT = "results/stage29a_pseudo_attribute_full7_eval"
OUTPUT_ROOT_DEFAULT = "results/stage35b_scheduled_tail_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE35B.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572


def parse_args():
    p = argparse.ArgumentParser(description="Stage35B scheduled mild-tail freeze")
    p.add_argument("--stage35_full7_root", default=STAGE35_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage35_train_root", default=STAGE35_TRAIN_ROOT_DEFAULT)
    p.add_argument("--stage34_freeze_json", default=STAGE34_FREEZE_DEFAULT)
    p.add_argument("--stage34_full7_root", default=STAGE34_FULL7_DEFAULT)
    p.add_argument("--stage30_freeze_json", default=STAGE30_FREEZE_DEFAULT)
    p.add_argument("--stage30_full7_root", default=STAGE30_FULL7_DEFAULT)
    p.add_argument("--stage29_freeze_json", default=STAGE29_FREEZE_DEFAULT)
    p.add_argument("--stage29_full7_root", default=STAGE29_FULL7_DEFAULT)
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


def delta(value: float | None, ref: float | None) -> float | None:
    if value is None or ref is None:
        return None
    return value - ref


def ref_metric(summary: dict, key: str, fallback: float) -> float:
    return float(summary.get(key, fallback))


def main():
    args = parse_args()
    full7_root = Path(args.stage35_full7_root)
    train_root = Path(args.stage35_train_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage35a_training_verdict.json")
    train_log = read_csv(train_root / "train_log.csv")

    stage34 = read_json(Path(args.stage34_freeze_json)) if Path(args.stage34_freeze_json).exists() else {}
    stage30 = read_json(Path(args.stage30_freeze_json)) if Path(args.stage30_freeze_json).exists() else {}
    stage29 = read_json(Path(args.stage29_freeze_json)) if Path(args.stage29_freeze_json).exists() else {}
    stage34_per = read_csv(Path(args.stage34_full7_root) / "per_machine.csv")
    stage30_per = read_csv(Path(args.stage30_full7_root) / "per_machine.csv")
    stage29_per = read_csv(Path(args.stage29_full7_root) / "per_machine.csv")

    refs = {
        "stage34": {
            "hmean": ref_metric(stage34, "mean_hmean", 0.6209075339680925),
            "pAUC": ref_metric(stage34, "mean_pAUC", 0.5754887218045113),
            "TPR99": ref_metric(stage34, "mean_TPR99", 0.13857142857142857),
            "per": stage34_per,
        },
        "stage30": {
            "hmean": ref_metric(stage30, "mean_hmean", 0.6231264650860286),
            "pAUC": ref_metric(stage30, "mean_pAUC", 0.5717293233082706),
            "TPR99": ref_metric(stage30, "mean_TPR99", 0.10857142857142857),
            "per": stage30_per,
        },
        "stage29": {
            "hmean": ref_metric(stage29, "mean_hmean", 0.6171391941988533),
            "pAUC": ref_metric(stage29, "mean_pAUC", 0.5724060150375939),
            "TPR99": ref_metric(stage29, "mean_TPR99", 0.12142857142857143),
            "per": stage29_per,
        },
    }

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score_row = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base_row = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        hmean = as_float(score_row, "hmean")
        pauc = as_float(score_row, "pAUC")
        t99 = as_float(score_row, "TPR@99n")
        row_out = {
            "machine": machine,
            "stage35_baseline_hmean": as_float(base_row, "hmean"),
            "stage35_ldn_hmean": hmean,
            "stage35_ldn_pAUC": pauc,
            "stage35_ldn_TPR99": t99,
            "delta_hmean_vs_stage35_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        }
        for ref_name, ref in refs.items():
            ref_row = get_row(ref["per"], candidate=FROZEN_ROUTE, machine=machine)
            row_out[f"{ref_name}_ldn_hmean"] = as_float(ref_row, "hmean")
            row_out[f"{ref_name}_ldn_pAUC"] = as_float(ref_row, "pAUC")
            row_out[f"{ref_name}_ldn_TPR99"] = as_float(ref_row, "TPR@99n")
            row_out[f"delta_hmean_vs_{ref_name}_ldn"] = delta(hmean, row_out[f"{ref_name}_ldn_hmean"])
            row_out[f"delta_pAUC_vs_{ref_name}_ldn"] = delta(pauc, row_out[f"{ref_name}_ldn_pAUC"])
            row_out[f"delta_TPR99_vs_{ref_name}_ldn"] = delta(t99, row_out[f"{ref_name}_ldn_TPR99"])
        machine_rows.append(row_out)

    mean_h = verdict["mean_hmean"]
    mean_p = verdict["mean_pAUC"]
    mean_t = verdict["mean_TPR99"]
    deltas = {
        name: {
            "hmean": mean_h - ref["hmean"],
            "pAUC": mean_p - ref["pAUC"],
            "TPR99": mean_t - ref["TPR99"],
        }
        for name, ref in refs.items()
    }
    if mean_h > refs["stage30"]["hmean"] and not verdict["strong_machine_severe_collapse"]:
        freeze_status = "new_hmean_best"
    elif mean_p > refs["stage34"]["pAUC"] and mean_t > refs["stage34"]["TPR99"] and not verdict["strong_machine_severe_collapse"]:
        freeze_status = "new_tail_best_trails_stage30_hmean"
    elif verdict["promotable"]:
        freeze_status = "promotable_tradeoff_not_new_best"
    else:
        freeze_status = "not_promoted"

    fan_row = get_row(machine_rows, machine="fan")
    summary = {
        "stage": "Stage35B scheduled mild-tail freeze",
        "freeze_status": freeze_status,
        "route": FROZEN_ROUTE,
        "route_description": "Stage35 scheduled mild-tail pseudo-attribute EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "best_checkpoint": train_verdict.get("best_checkpoint"),
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "num_pseudo_attributes": train_verdict.get("num_pseudo_attributes"),
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_TPR99": mean_t,
        "delta_hmean_vs_clean_best": mean_h - CLEAN_BEST_MEAN_HMEAN,
        "delta_pAUC_vs_clean_best": mean_p - CLEAN_BEST_MEAN_PAUC,
        "delta_TPR99_vs_clean_best": mean_t - CLEAN_BEST_MEAN_TPR99,
        "delta_hmean_vs_stage34": deltas["stage34"]["hmean"],
        "delta_pAUC_vs_stage34": deltas["stage34"]["pAUC"],
        "delta_TPR99_vs_stage34": deltas["stage34"]["TPR99"],
        "delta_hmean_vs_stage30": deltas["stage30"]["hmean"],
        "delta_pAUC_vs_stage30": deltas["stage30"]["pAUC"],
        "delta_TPR99_vs_stage30": deltas["stage30"]["TPR99"],
        "delta_hmean_vs_stage29": deltas["stage29"]["hmean"],
        "delta_pAUC_vs_stage29": deltas["stage29"]["pAUC"],
        "delta_TPR99_vs_stage29": deltas["stage29"]["TPR99"],
        "improved_machine_count_vs_stage35_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "fan_delta_hmean_vs_stage30_ldn": fan_row["delta_hmean_vs_stage30_ldn"] if fan_row else None,
        "fan_delta_TPR99_vs_stage30_ldn": fan_row["delta_TPR99_vs_stage30_ldn"] if fan_row else None,
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "uses_anomaly_labels": train_verdict.get("uses_anomaly_labels"),
        "uses_test_labels_for_checkpoint_selection": train_verdict.get("uses_test_labels_for_checkpoint_selection"),
        "ldn_parameters_tuned": train_verdict.get("ldn_parameters_tuned"),
        "last_epoch_tail_dispersion_loss_weight": last_float(train_log, "tail_dispersion_loss_weight"),
        "last_epoch_val_acc": last_float(train_log, "val_acc"),
        "last_epoch_val_anchor_loss": last_float(train_log, "val_anchor_loss"),
        "last_epoch_val_pairwise_geometry_loss": last_float(train_log, "val_pairwise_geometry_loss"),
        "last_epoch_val_machine_center_loss": last_float(train_log, "val_machine_center_loss"),
        "last_epoch_val_tail_dispersion_loss": last_float(train_log, "val_tail_dispersion_loss"),
        "last_epoch_val_tail_dispersion_ratio": last_float(train_log, "val_tail_dispersion_ratio"),
        "recommended_status": "Stage35 improves Stage34 on hmean, pAUC, and TPR@99n, but still trails Stage30 hmean. Keep as new tail-best branch.",
    }
    write_json(out_root / "stage35b_scheduled_tail_freeze_summary.json", summary)
    write_csv(out_root / "stage35b_machine_comparison.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE35B",
        "",
        f"Freeze status: `{freeze_status}`",
        "",
        f"- mean hmean: `{mean_h}`",
        f"- mean pAUC: `{mean_p}`",
        f"- mean TPR@99n: `{mean_t}`",
        f"- delta hmean vs Stage34: `{deltas['stage34']['hmean']}`",
        f"- delta pAUC vs Stage34: `{deltas['stage34']['pAUC']}`",
        f"- delta TPR@99n vs Stage34: `{deltas['stage34']['TPR99']}`",
        f"- delta hmean vs Stage30: `{deltas['stage30']['hmean']}`",
        f"- delta pAUC vs Stage30: `{deltas['stage30']['pAUC']}`",
        f"- delta TPR@99n vs Stage30: `{deltas['stage30']['TPR99']}`",
        f"- strong-machine severe collapse: `{verdict['strong_machine_severe_collapse']}`",
        "",
        "Stage35A is the new tail-best branch. It improves Stage34A while retaining the same tradeoff: Stage30 remains hmean-best, Stage35 is pAUC/TPR@99n-best.",
    ])


if __name__ == "__main__":
    main()
