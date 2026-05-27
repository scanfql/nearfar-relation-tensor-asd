#!/usr/bin/env python
"""Stage34B: freeze Stage34A tail-guardrailed pseudo-attribute result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE34_FULL7_ROOT_DEFAULT = "results/stage34a_tail_guardrailed_pseudo_attribute_full7_eval"
STAGE34_TRAIN_ROOT_DEFAULT = "results/stage34a_tail_guardrailed_pseudo_attribute"
STAGE30_FREEZE_DEFAULT = "results/stage30b_guardrailed_freeze/stage30b_guardrailed_freeze_summary.json"
STAGE30_FULL7_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute_full7_eval"
STAGE29_FREEZE_DEFAULT = "results/stage29b_pseudo_attribute_freeze/stage29b_pseudo_attribute_freeze_summary.json"
STAGE29_FULL7_DEFAULT = "results/stage29a_pseudo_attribute_full7_eval"
OUTPUT_ROOT_DEFAULT = "results/stage34b_tail_guardrailed_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE34B.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572
STAGE30_MEAN_HMEAN = 0.6231264650860286
STAGE30_MEAN_PAUC = 0.5717293233082706
STAGE30_MEAN_TPR99 = 0.10857142857142857
STAGE29_MEAN_HMEAN = 0.6171391941988533
STAGE29_MEAN_PAUC = 0.5724060150375939
STAGE29_MEAN_TPR99 = 0.12142857142857143


def parse_args():
    p = argparse.ArgumentParser(description="Stage34B tail-guardrailed pseudo-attribute freeze")
    p.add_argument("--stage34_full7_root", default=STAGE34_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage34_train_root", default=STAGE34_TRAIN_ROOT_DEFAULT)
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


def main():
    args = parse_args()
    full7_root = Path(args.stage34_full7_root)
    train_root = Path(args.stage34_train_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage34a_training_verdict.json")
    train_log = read_csv(train_root / "train_log.csv")

    stage30 = read_json(Path(args.stage30_freeze_json)) if Path(args.stage30_freeze_json).exists() else {}
    stage29 = read_json(Path(args.stage29_freeze_json)) if Path(args.stage29_freeze_json).exists() else {}
    stage30_per_machine = read_csv(Path(args.stage30_full7_root) / "per_machine.csv")
    stage29_per_machine = read_csv(Path(args.stage29_full7_root) / "per_machine.csv")

    s30_h = float(stage30.get("mean_hmean", STAGE30_MEAN_HMEAN))
    s30_p = float(stage30.get("mean_pAUC", STAGE30_MEAN_PAUC))
    s30_t = float(stage30.get("mean_TPR99", STAGE30_MEAN_TPR99))
    s29_h = float(stage29.get("mean_hmean", STAGE29_MEAN_HMEAN))
    s29_p = float(stage29.get("mean_pAUC", STAGE29_MEAN_PAUC))
    s29_t = float(stage29.get("mean_TPR99", STAGE29_MEAN_TPR99))

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score_row = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base_row = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        s30_row = get_row(stage30_per_machine, candidate=FROZEN_ROUTE, machine=machine)
        s29_row = get_row(stage29_per_machine, candidate=FROZEN_ROUTE, machine=machine)
        hmean = as_float(score_row, "hmean")
        pauc = as_float(score_row, "pAUC")
        t99 = as_float(score_row, "TPR@99n")
        row_out = {
            "machine": machine,
            "stage34_baseline_hmean": as_float(base_row, "hmean"),
            "stage34_ldn_hmean": hmean,
            "stage34_ldn_pAUC": pauc,
            "stage34_ldn_TPR99": t99,
            "delta_hmean_vs_stage34_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "stage30_ldn_hmean": as_float(s30_row, "hmean"),
            "stage30_ldn_pAUC": as_float(s30_row, "pAUC"),
            "stage30_ldn_TPR99": as_float(s30_row, "TPR@99n"),
            "stage29_ldn_hmean": as_float(s29_row, "hmean"),
            "stage29_ldn_pAUC": as_float(s29_row, "pAUC"),
            "stage29_ldn_TPR99": as_float(s29_row, "TPR@99n"),
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        }
        row_out["delta_hmean_vs_stage30_ldn"] = delta(hmean, row_out["stage30_ldn_hmean"])
        row_out["delta_pAUC_vs_stage30_ldn"] = delta(pauc, row_out["stage30_ldn_pAUC"])
        row_out["delta_TPR99_vs_stage30_ldn"] = delta(t99, row_out["stage30_ldn_TPR99"])
        row_out["delta_hmean_vs_stage29_ldn"] = delta(hmean, row_out["stage29_ldn_hmean"])
        row_out["delta_pAUC_vs_stage29_ldn"] = delta(pauc, row_out["stage29_ldn_pAUC"])
        row_out["delta_TPR99_vs_stage29_ldn"] = delta(t99, row_out["stage29_ldn_TPR99"])
        machine_rows.append(row_out)

    positive_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage34_baseline"] > 0]
    negative_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage34_baseline"] < 0]
    positive_h_vs_stage30 = [r for r in machine_rows if r["delta_hmean_vs_stage30_ldn"] is not None and r["delta_hmean_vs_stage30_ldn"] > 0]
    negative_h_vs_stage30 = [r for r in machine_rows if r["delta_hmean_vs_stage30_ldn"] is not None and r["delta_hmean_vs_stage30_ldn"] < 0]
    positive_t_vs_stage30 = [r for r in machine_rows if r["delta_TPR99_vs_stage30_ldn"] is not None and r["delta_TPR99_vs_stage30_ldn"] > 0]
    fan_row = get_row(machine_rows, machine="fan")

    mean_h = verdict["mean_hmean"]
    mean_p = verdict["mean_pAUC"]
    mean_t = verdict["mean_TPR99"]
    delta_h_stage30 = mean_h - s30_h
    delta_p_stage30 = mean_p - s30_p
    delta_t_stage30 = mean_t - s30_t
    delta_h_stage29 = mean_h - s29_h
    delta_p_stage29 = mean_p - s29_p
    delta_t_stage29 = mean_t - s29_t

    if mean_h > s30_h and not verdict["strong_machine_severe_collapse"]:
        freeze_status = "new_hmean_best"
    elif mean_p > s29_p and mean_t > s29_t and mean_h > s29_h and not verdict["strong_machine_severe_collapse"]:
        freeze_status = "new_tail_best_trails_stage30_hmean"
    elif verdict["promotable"]:
        freeze_status = "promotable_tradeoff_not_new_best"
    else:
        freeze_status = "not_promoted"

    summary = {
        "stage": "Stage34B tail-guardrailed pseudo-attribute freeze",
        "freeze_status": freeze_status,
        "route": FROZEN_ROUTE,
        "route_description": "Stage34 tail-guardrailed pseudo-attribute EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
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
        "delta_hmean_vs_stage30": delta_h_stage30,
        "delta_pAUC_vs_stage30": delta_p_stage30,
        "delta_TPR99_vs_stage30": delta_t_stage30,
        "delta_hmean_vs_stage29": delta_h_stage29,
        "delta_pAUC_vs_stage29": delta_p_stage29,
        "delta_TPR99_vs_stage29": delta_t_stage29,
        "improved_machine_count_vs_stage34_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_stage34_baseline": [r["machine"] for r in positive_vs_baseline],
        "negative_machines_vs_stage34_baseline": [r["machine"] for r in negative_vs_baseline],
        "positive_hmean_machines_vs_stage30": [r["machine"] for r in positive_h_vs_stage30],
        "negative_hmean_machines_vs_stage30": [r["machine"] for r in negative_h_vs_stage30],
        "positive_TPR99_machines_vs_stage30": [r["machine"] for r in positive_t_vs_stage30],
        "fan_ldn_minus_baseline_hmean": fan_row["delta_hmean_vs_stage34_baseline"] if fan_row else None,
        "fan_delta_hmean_vs_stage30_ldn": fan_row["delta_hmean_vs_stage30_ldn"] if fan_row else None,
        "fan_delta_TPR99_vs_stage30_ldn": fan_row["delta_TPR99_vs_stage30_ldn"] if fan_row else None,
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "uses_anomaly_labels": train_verdict.get("uses_anomaly_labels"),
        "uses_test_labels_for_checkpoint_selection": train_verdict.get("uses_test_labels_for_checkpoint_selection"),
        "ldn_parameters_tuned": train_verdict.get("ldn_parameters_tuned"),
        "last_epoch_val_acc": last_float(train_log, "val_acc"),
        "last_epoch_val_anchor_loss": last_float(train_log, "val_anchor_loss"),
        "last_epoch_val_pairwise_geometry_loss": last_float(train_log, "val_pairwise_geometry_loss"),
        "last_epoch_val_machine_center_loss": last_float(train_log, "val_machine_center_loss"),
        "last_epoch_val_tail_dispersion_loss": last_float(train_log, "val_tail_dispersion_loss"),
        "last_epoch_val_tail_dispersion_ratio": last_float(train_log, "val_tail_dispersion_ratio"),
        "recommended_status": "Tail objective is productive: Stage34A trails Stage30 hmean but exceeds Stage29/30 pAUC and TPR@99n. Keep as tail-best branch, not hmean-best.",
        "forbidden_followups": [
            "Do not tune LDN K/r/score_k.",
            "Do not add selector or fusion to hide the Stage30/34 hmean-tail tradeoff.",
            "Do not select checkpoints with test labels.",
            "Do not treat Stage34A as the hmean best unless a follow-up recovers Stage30 hmean.",
        ],
    }

    write_json(out_root / "stage34b_tail_guardrailed_freeze_summary.json", summary)
    write_csv(out_root / "stage34b_machine_comparison.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE34B",
        "",
        "## Result",
        "",
        f"Freeze status: `{summary['freeze_status']}`",
        "",
        f"- mean hmean: `{mean_h}`",
        f"- mean pAUC: `{mean_p}`",
        f"- mean TPR@99n: `{mean_t}`",
        f"- delta hmean vs Stage30: `{delta_h_stage30}`",
        f"- delta pAUC vs Stage30: `{delta_p_stage30}`",
        f"- delta TPR@99n vs Stage30: `{delta_t_stage30}`",
        f"- delta hmean vs Stage29: `{delta_h_stage29}`",
        f"- delta pAUC vs Stage29: `{delta_p_stage29}`",
        f"- delta TPR@99n vs Stage29: `{delta_t_stage29}`",
        f"- strong-machine severe collapse: `{summary['strong_machine_severe_collapse']}`",
        "",
        "## Interpretation",
        "",
        "Stage34A is not a new hmean best because it trails Stage30 on mean hmean. It is a new tail-best branch in this pseudo-attribute family: both pAUC and TPR@99n exceed Stage29 and Stage30 while remaining promotable versus the clean best.",
        "",
        "## Follow-Up",
        "",
        "The next useful branch should keep the Stage34 tail objective but reduce the Stage30 hmean loss, likely by making the tail pressure milder or delayed. Do not tune LDN, add selectors/fusion, or select with test labels.",
    ])


if __name__ == "__main__":
    main()
