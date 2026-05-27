#!/usr/bin/env python
"""Stage32B: freeze scheduled guardrail pseudo-attribute result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE32_FULL7_ROOT_DEFAULT = "results/stage32a_scheduled_guardrail_full7_eval"
STAGE32_TRAIN_ROOT_DEFAULT = "results/stage32a_scheduled_guardrail_pseudo_attribute"
STAGE30_FREEZE_DEFAULT = "results/stage30b_guardrailed_freeze/stage30b_guardrailed_freeze_summary.json"
STAGE30_FULL7_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute_full7_eval"
STAGE29_FREEZE_DEFAULT = "results/stage29b_pseudo_attribute_freeze/stage29b_pseudo_attribute_freeze_summary.json"
OUTPUT_ROOT_DEFAULT = "results/stage32b_scheduled_guardrail_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE32.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_HMEAN = 0.6159467288425517
CLEAN_BEST_PAUC = 0.5603007518796993
CLEAN_BEST_TPR99 = 0.10571428571428572


def parse_args():
    p = argparse.ArgumentParser(description="Stage32B scheduled guardrail freeze")
    p.add_argument("--stage32_full7_root", default=STAGE32_FULL7_ROOT_DEFAULT)
    p.add_argument("--stage32_train_root", default=STAGE32_TRAIN_ROOT_DEFAULT)
    p.add_argument("--stage30_freeze_json", default=STAGE30_FREEZE_DEFAULT)
    p.add_argument("--stage30_full7_root", default=STAGE30_FULL7_DEFAULT)
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
    full7_root = Path(args.stage32_full7_root)
    train_root = Path(args.stage32_train_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(full7_root / "stage22d_adapted_full7_verdict.json")
    per_machine = read_csv(full7_root / "per_machine.csv")
    collapse = read_csv(full7_root / "collapse_check.csv")
    train_verdict = read_json(train_root / "stage32a_training_verdict.json")
    train_log = read_csv(train_root / "train_log.csv")
    stage30 = read_json(Path(args.stage30_freeze_json))
    stage29 = read_json(Path(args.stage29_freeze_json))
    stage30_pm = read_csv(Path(args.stage30_full7_root) / "per_machine.csv")

    machine_rows = []
    for row in collapse:
        if row.get("candidate") != FROZEN_ROUTE:
            continue
        machine = row["machine"]
        score = get_row(per_machine, candidate=FROZEN_ROUTE, machine=machine)
        base = get_row(per_machine, candidate=ADAPTED_BASELINE, machine=machine)
        s30 = get_row(stage30_pm, candidate=FROZEN_ROUTE, machine=machine)
        h = as_float(score, "hmean")
        p = as_float(score, "pAUC")
        t = as_float(score, "TPR@99n")
        s30_h = as_float(s30, "hmean")
        s30_p = as_float(s30, "pAUC")
        s30_t = as_float(s30, "TPR@99n")
        machine_rows.append({
            "machine": machine,
            "stage32_baseline_hmean": as_float(base, "hmean"),
            "stage32_ldn_hmean": h,
            "stage32_ldn_pAUC": p,
            "stage32_ldn_TPR99": t,
            "delta_hmean_vs_stage32_baseline": float(row["delta_hmean_vs_adapted_baseline"]),
            "stage30_ldn_hmean": s30_h,
            "stage30_ldn_pAUC": s30_p,
            "stage30_ldn_TPR99": s30_t,
            "delta_hmean_vs_stage30_ldn": h - s30_h if h is not None and s30_h is not None else None,
            "delta_pAUC_vs_stage30_ldn": p - s30_p if p is not None and s30_p is not None else None,
            "delta_TPR99_vs_stage30_ldn": t - s30_t if t is not None and s30_t is not None else None,
            "strong_machine": row["strong_machine"],
            "severe_collapse": row["severe_collapse"],
        })

    positive_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage32_baseline"] > 0]
    negative_vs_baseline = [r for r in machine_rows if r["delta_hmean_vs_stage32_baseline"] < 0]
    positive_vs_stage30 = [r for r in machine_rows if r["delta_hmean_vs_stage30_ldn"] is not None and r["delta_hmean_vs_stage30_ldn"] > 0]
    negative_vs_stage30 = [r for r in machine_rows if r["delta_hmean_vs_stage30_ldn"] is not None and r["delta_hmean_vs_stage30_ldn"] < 0]
    fan = get_row(machine_rows, machine="fan")
    valve = get_row(machine_rows, machine="valve")

    warmup_rows = [r for r in train_log if r.get("phase") == "warmup"]
    guard_rows = [r for r in train_log if r.get("phase") == "guard"]
    final_val_acc = last_float(train_log, "val_acc")
    final_anchor = last_float(train_log, "val_anchor_loss")
    warmup_final_acc = float(warmup_rows[-1]["val_acc"]) if warmup_rows else None
    guard_final_acc = float(guard_rows[-1]["val_acc"]) if guard_rows else None

    delta_h_stage30 = verdict["mean_hmean"] - float(stage30["mean_hmean"])
    delta_p_stage30 = verdict["mean_pAUC"] - float(stage30["mean_pAUC"])
    delta_t_stage30 = verdict["mean_TPR99"] - float(stage30["mean_TPR99"])
    delta_h_stage29 = verdict["mean_hmean"] - float(stage29["mean_hmean"])
    delta_p_stage29 = verdict["mean_pAUC"] - float(stage29["mean_pAUC"])
    delta_t_stage29 = verdict["mean_TPR99"] - float(stage29["mean_TPR99"])
    not_promoted = bool(delta_h_stage30 < 0 and delta_t_stage30 <= 0)

    summary = {
        "stage": "Stage32B scheduled guardrail freeze",
        "freeze_status": "scheduled_guardrail_not_promoted" if not_promoted else "review_required",
        "route": FROZEN_ROUTE,
        "route_description": "Stage32 scheduled guardrail pseudo-attribute EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5",
        "best_checkpoint": train_verdict.get("best_checkpoint"),
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "warmup_epochs": train_verdict.get("warmup_epochs"),
        "checkpoint_selection": train_verdict.get("checkpoint_selection"),
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
        "improved_machine_count_vs_stage32_baseline": verdict["improved_machine_count_vs_adapted_baseline"],
        "positive_machines_vs_stage32_baseline": [r["machine"] for r in positive_vs_baseline],
        "negative_machines_vs_stage32_baseline": [r["machine"] for r in negative_vs_baseline],
        "positive_machines_vs_stage30_ldn": [r["machine"] for r in positive_vs_stage30],
        "negative_machines_vs_stage30_ldn": [r["machine"] for r in negative_vs_stage30],
        "fan_ldn_minus_baseline_hmean": fan["delta_hmean_vs_stage32_baseline"] if fan else None,
        "fan_delta_hmean_vs_stage30_ldn": fan["delta_hmean_vs_stage30_ldn"] if fan else None,
        "valve_delta_TPR99_vs_stage30_ldn": valve["delta_TPR99_vs_stage30_ldn"] if valve else None,
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "promotable": verdict["promotable"],
        "warmup_final_val_acc": warmup_final_acc,
        "guard_final_val_acc": guard_final_acc,
        "final_val_acc": final_val_acc,
        "final_val_anchor_loss": final_anchor,
        "stage30_final_val_anchor_loss": stage30["last_epoch_val_anchor_loss"],
        "diagnosis": "Warmup was underpowered: pseudo-attribute validation accuracy stayed at zero, so the schedule did not recover Stage29 tail while also trailing Stage30 hmean.",
        "recommended_next": "Stop simple scheduling. Next useful direction is a stage30-near objective with a normal-only tail/dispersion term, or a better pseudo-attribute signal, not longer warmup with the same CE.",
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_fusion": True,
            "no_test_label_checkpoint_selection": True,
        },
        "forbidden_followups": [
            "Do not keep extending the same warmup schedule without evidence of pseudo-attribute learning.",
            "Do not add selector or fusion to choose Stage29/Stage30/Stage32 per machine.",
            "Do not tune LDN K/r/score_k.",
            "Do not select checkpoints with test labels.",
        ],
    }

    write_json(out_root / "stage32b_scheduled_guardrail_freeze_summary.json", summary)
    write_csv(out_root / "stage32b_machine_delta_table.csv", machine_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE32",
        "",
        "## 1. Stage32A result",
        "",
        "Stage32A tested scheduled pseudo-attribute training: warmup with light guardrails, then Stage30-like guardrails. Checkpoint selection was restricted to guard-phase normal validation loss.",
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
        "Conclusion: Stage32A is not promoted. It remains positive versus the clean baseline but trails Stage30 on hmean and does not recover Stage29/Stage30 tail.",
        "",
        "## 2. Training diagnostic",
        "",
        f"- warmup final val acc: `{summary['warmup_final_val_acc']}`",
        f"- guard final val acc: `{summary['guard_final_val_acc']}`",
        f"- final val anchor loss: `{summary['final_val_anchor_loss']}`",
        f"- Stage30 final val anchor loss: `{summary['stage30_final_val_anchor_loss']}`",
        "",
        "The warmup was underpowered. Pseudo-attribute validation accuracy stayed at zero, so the schedule did not create useful tail evidence before guardrails pulled the geometry back.",
        "",
        "## 3. Current route hierarchy",
        "",
        "- hmean best: `Stage30 guardrailed pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- pAUC/TPR@99n best: `Stage29 pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- Stage31/32: not promoted",
        "",
        "## 4. Recommended next step",
        "",
        "Stop simple guardrail schedule variants. The next useful direction is a Stage30-near objective with a normal-only tail/dispersion term, or a better pseudo-attribute target. Do not keep stretching warmup unless pseudo-attribute learning is demonstrably nonzero.",
        "",
        "## 5. Forbidden next moves",
        "",
        "- Do not keep extending the same warmup schedule without evidence of pseudo-attribute learning.",
        "- Do not add selector or fusion to choose Stage29/Stage30/Stage32 per machine.",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not use test labels for checkpoint selection.",
    ])


if __name__ == "__main__":
    main()
