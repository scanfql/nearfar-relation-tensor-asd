#!/usr/bin/env python
"""Stage33B: freeze fixed-prototype pseudo-attribute result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE33_FULL7_ROOT_DEFAULT = "results/stage33a_fixed_prototype_full7_eval"
STAGE33_TRAIN_ROOT_DEFAULT = "results/stage33a_fixed_prototype_representation"
STAGE30_FREEZE_DEFAULT = "results/stage30b_guardrailed_freeze/stage30b_guardrailed_freeze_summary.json"
STAGE29_FREEZE_DEFAULT = "results/stage29b_pseudo_attribute_freeze/stage29b_pseudo_attribute_freeze_summary.json"
OUTPUT_ROOT_DEFAULT = "results/stage33b_fixed_prototype_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE33.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_HMEAN = 0.6159467288425517
CLEAN_BEST_PAUC = 0.5603007518796993
CLEAN_BEST_TPR99 = 0.10571428571428572


def parse_args():
    p = argparse.ArgumentParser(description="Stage33B fixed-prototype freeze")
    p.add_argument("--stage33_full7_root", default=STAGE33_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage33_train_root", default=STAGE33_TRAIN_ROOT_DEFAULT)
    p.add_argument("--stage30_freeze_json", default=STAGE30_FREEZE_DEFAULT)
    p.add_argument("--stage29_freeze_json", default=STAGE29_FREEZE_DEFAULT)
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
    full7_root = Path(args.stage33_full7_root)
    train_root = Path(args.stage33_train_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage33a_training_verdict.json")
    train_log = read_csv(train_root / "train_log.csv")
    stage30 = read_json(Path(args.stage30_freeze_json))
    stage29 = read_json(Path(args.stage29_freeze_json))

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        h = as_float(score, "hmean")
        p = as_float(score, "pAUC")
        t = as_float(score, "TPR@99n")
        machine_rows.append({
            "machine": machine,
            "stage33_baseline_hmean": as_float(base, "hmean"),
            "stage33_ldn_hmean": h,
            "stage33_ldn_pAUC": p,
            "stage33_ldn_TPR99": t,
            "delta_hmean_vs_stage33_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        })

    fan = get_row(machine_rows, machine="fan")
    valve = get_row(machine_rows, machine="valve")
    positive_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage33_baseline"] > 0]
    negative_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage33_baseline"] < 0]

    delta_h_stage30 = verdict["mean_hmean"] - float(stage30["mean_hmean"])
    delta_p_stage30 = verdict["mean_pAUC"] - float(stage30["mean_pAUC"])
    delta_t_stage30 = verdict["mean_TPR99"] - float(stage30["mean_TPR99"])
    delta_h_stage29 = verdict["mean_hmean"] - float(stage29["mean_hmean"])
    delta_p_stage29 = verdict["mean_pAUC"] - float(stage29["mean_pAUC"])
    delta_t_stage29 = verdict["mean_TPR99"] - float(stage29["mean_TPR99"])

    final_acc = last_float(train_log, "val_prototype_acc")
    final_anchor = last_float(train_log, "val_anchor_loss")
    underpowered = bool((final_acc or 0.0) <= 0.001 and delta_h_stage30 < 0 and delta_t_stage30 < 0)
    summary = {
        "stage": "Stage33B fixed-prototype freeze",
        "freeze_status": "fixed_prototype_underpowered_negative_freeze" if underpowered else "review_required",
        "route": FROZEN_ROUTE,
        "route_description": "Stage33 fixed clean-EAT prototype pseudo-attribute EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "best_checkpoint": train_verdict.get("best_checkpoint"),
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "num_pseudo_attributes": train_verdict.get("num_pseudo_attributes"),
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "delta_hmean_vs_clean_best": verdict["mean_hmean"] - CLEAN_BEST_HMEAN,
        "delta_pAUC_vs_clean_best": verdict["mean_pAUC"] - CLEAN_BEST_PAUC,
        "delta_TPR99_vs_clean_best": verdict["mean_TPR99"] - CLEAN_BEST_TPR99,
        "delta_hmean_vs_stage30": delta_h_stage30,
        "delta_pAUC_vs_stage30": delta_p_stage30,
        "delta_TPR99_vs_stage30": delta_t_stage30,
        "delta_hmean_vs_stage29": delta_h_stage29,
        "delta_pAUC_vs_stage29": delta_p_stage29,
        "delta_TPR99_vs_stage29": delta_t_stage29,
        "improved_machine_count_vs_stage33_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_stage33_baseline": [r["machine"] for r in positive_vs_baseline],
        "negative_machines_vs_stage33_baseline": [r["machine"] for r in negative_vs_baseline],
        "fan_ldn_minus_baseline_hmean": fan["delta_hmean_vs_stage33_baseline"] if fan else None,
        "fan_hmean": fan["stage33_ldn_hmean"] if fan else None,
        "fan_TPR99": fan["stage33_ldn_TPR99"] if fan else None,
        "valve_hmean": valve["stage33_ldn_hmean"] if valve else None,
        "valve_TPR99": valve["stage33_ldn_TPR99"] if valve else None,
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "final_val_prototype_acc": final_acc,
        "final_val_anchor_loss": final_anchor,
        "stage30_final_val_anchor_loss": stage30["last_epoch_val_anchor_loss"],
        "diagnosis": "Fixed clean-EAT prototypes were too weak under guardrails: prototype validation accuracy stayed at zero and the route trailed Stage30/Stage29.",
        "recommended_next": "Do not continue fixed-prototype alignment as implemented. Return to trainable pseudo-attribute shaping, but improve the pseudo labels or add a normal-only tail/dispersion objective.",
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_fusion": True,
            "no_test_label_checkpoint_selection": True,
        },
        "forbidden_followups": [
            "Do not increase prototype margin/logit scale as an open-ended score search.",
            "Do not add selector or fusion to choose routes per machine.",
            "Do not tune LDN K/r/score_k.",
            "Do not use test labels for checkpoint selection.",
        ],
    }

    write_json(out_root / "stage33b_fixed_prototype_freeze_summary.json", summary)
    write_csv(out_root / "stage33b_machine_delta_table.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE33",
        "",
        "## 1. Stage33A result",
        "",
        "Stage33A replaced the trainable pseudo-attribute head with fixed clean-EAT pseudo-attribute prototypes.",
        "",
        f"- route: `{FROZEN_ROUTE}`",
        f"- mean_hmean: `{summary['mean_hmean']}`",
        f"- mean_pAUC: `{summary['mean_pAUC']}`",
        f"- mean_TPR@99n: `{summary['mean_TPR99']}`",
        f"- delta_hmean_vs_stage30: `{summary['delta_hmean_vs_stage30']}`",
        f"- delta_pAUC_vs_stage30: `{summary['delta_pAUC_vs_stage30']}`",
        f"- delta_TPR99_vs_stage30: `{summary['delta_TPR99_vs_stage30']}`",
        f"- freeze_status: `{summary['freeze_status']}`",
        "",
        "Conclusion: Stage33A is not promoted. It is positive versus clean hmean but trails Stage30 and Stage29 on the route metrics that matter.",
        "",
        "## 2. Training diagnostic",
        "",
        f"- final val prototype accuracy: `{summary['final_val_prototype_acc']}`",
        f"- final val anchor loss: `{summary['final_val_anchor_loss']}`",
        f"- Stage30 final val anchor loss: `{summary['stage30_final_val_anchor_loss']}`",
        "",
        "Fixed prototypes did not provide a learnable pseudo-attribute signal under guardrails. Prototype validation accuracy stayed at zero.",
        "",
        "## 3. Current route hierarchy",
        "",
        "- hmean best: `Stage30 guardrailed pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- pAUC/TPR@99n best: `Stage29 pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- Stage31/32/33: not promoted",
        "",
        "## 4. Recommended next step",
        "",
        "Stop fixed-prototype alignment as implemented. The next useful branch should improve pseudo labels or add a normal-only tail/dispersion objective around the Stage30 setting.",
        "",
        "## 5. Forbidden next moves",
        "",
        "- Do not increase prototype margin/logit scale as an open-ended score search.",
        "- Do not add selector or fusion to choose routes per machine.",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not use test labels for checkpoint selection.",
    ])


if __name__ == "__main__":
    main()
