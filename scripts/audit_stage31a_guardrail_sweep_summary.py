#!/usr/bin/env python
"""Stage31A: summarize guardrail-strength sweep.

This script does not train or score. It reads frozen-backend full7 evaluation
folders produced by Stage30A-style guardrailed pseudo-attribute variants and
compares them against Stage29 and Stage30 references.

No selector, fusion, LDN parameter tuning, or test-label checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUTPUT_ROOT_DEFAULT = "results/stage31a_guardrail_sweep_summary"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE31.md"

STAGE29_FULL7_DEFAULT = "results/stage29a_pseudo_attribute_full7_eval"
STAGE29_FREEZE_DEFAULT = "results/stage29b_pseudo_attribute_freeze/stage29b_pseudo_attribute_freeze_summary.json"
STAGE30_FULL7_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute_full7_eval"
STAGE30_FREEZE_DEFAULT = "results/stage30b_guardrailed_freeze/stage30b_guardrailed_freeze_summary.json"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
ADAPTED_BASELINE = "adapted_baseline_knn_k5"
CLEAN_BEST_HMEAN = 0.6159467288425517
CLEAN_BEST_PAUC = 0.5603007518796993
CLEAN_BEST_TPR99 = 0.10571428571428572

DEFAULT_VARIANTS = [
    ("stage31_anchor030", "results/stage31a_anchor030_full7_eval", "results/stage31a_anchor030"),
    ("stage31_anchor020", "results/stage31b_anchor020_full7_eval", "results/stage31b_anchor020"),
    ("stage31_anchor030_pair025", "results/stage31c_anchor030_pair025_full7_eval", "results/stage31c_anchor030_pair025"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Stage31A guardrail sweep summary")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=DOC_PATH_DEFAULT)
    p.add_argument("--stage29_full7_root", default=STAGE29_FULL7_DEFAULT)
    p.add_argument("--stage29_freeze_json", default=STAGE29_FREEZE_DEFAULT)
    p.add_argument("--stage30_full7_root", default=STAGE30_FULL7_DEFAULT)
    p.add_argument("--stage30_freeze_json", default=STAGE30_FREEZE_DEFAULT)
    p.add_argument(
        "--variants",
        default=";".join(f"{name},{eval_root},{train_root}" for name, eval_root, train_root in DEFAULT_VARIANTS),
        help="Semicolon-separated name,eval_root,train_root triples.",
    )
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


def parse_variants(text: str) -> list[tuple[str, Path, Path]]:
    variants = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Variant must be name,eval_root,train_root: {chunk}")
        variants.append((parts[0], Path(parts[1]), Path(parts[2])))
    return variants


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


def load_variant(name: str, eval_root: Path, train_root: Path, stage29_pm: list[dict], stage30_pm: list[dict], stage29_summary: dict, stage30_summary: dict) -> dict:
    verdict_path = eval_root / "stage22d_adapted_full7_verdict.json"
    per_machine_path = eval_root / "per_machine.csv"
    train_verdict_path = train_root / "stage30a_training_verdict.json"
    train_log_path = train_root / "train_log.csv"
    available = verdict_path.exists() and per_machine_path.exists()
    row = {
        "variant": name,
        "eval_root": str(eval_root),
        "train_root": str(train_root),
        "eval_available": available,
    }
    if not available:
        return row

    verdict = read_json(verdict_path)
    pm = read_csv(per_machine_path)
    train_verdict = read_json(train_verdict_path) if train_verdict_path.exists() else {}
    train_log = read_csv(train_log_path)
    fan = get_row(pm, candidate=FROZEN_ROUTE, machine="fan")
    valve = get_row(pm, candidate=FROZEN_ROUTE, machine="valve")
    fan_base = get_row(pm, candidate=ADAPTED_BASELINE, machine="fan")
    s29_fan = get_row(stage29_pm, candidate=FROZEN_ROUTE, machine="fan")
    s30_fan = get_row(stage30_pm, candidate=FROZEN_ROUTE, machine="fan")
    s29_valve = get_row(stage29_pm, candidate=FROZEN_ROUTE, machine="valve")
    s30_valve = get_row(stage30_pm, candidate=FROZEN_ROUTE, machine="valve")

    h = verdict["mean_hmean"]
    p = verdict["mean_pAUC"]
    t = verdict["mean_TPR99"]
    s29_h = float(stage29_summary["mean_hmean"])
    s29_p = float(stage29_summary["mean_pAUC"])
    s29_t = float(stage29_summary["mean_TPR99"])
    s30_h = float(stage30_summary["mean_hmean"])
    s30_p = float(stage30_summary["mean_pAUC"])
    s30_t = float(stage30_summary["mean_TPR99"])
    fan_h = as_float(fan, "hmean")
    fan_base_h = as_float(fan_base, "hmean")

    row.update({
        "mean_hmean": h,
        "mean_pAUC": p,
        "mean_TPR99": t,
        "delta_hmean_vs_clean": h - CLEAN_BEST_HMEAN,
        "delta_pAUC_vs_clean": p - CLEAN_BEST_PAUC,
        "delta_TPR99_vs_clean": t - CLEAN_BEST_TPR99,
        "delta_hmean_vs_stage29": h - s29_h,
        "delta_pAUC_vs_stage29": p - s29_p,
        "delta_TPR99_vs_stage29": t - s29_t,
        "delta_hmean_vs_stage30": h - s30_h,
        "delta_pAUC_vs_stage30": p - s30_p,
        "delta_TPR99_vs_stage30": t - s30_t,
        "promotable": verdict.get("promotable"),
        "strong_machine_severe_collapse": verdict.get("strong_machine_severe_collapse"),
        "improved_machine_count_vs_local_baseline": verdict.get("improved_machine_count_vs_adapted_baseline"),
        "fan_hmean": fan_h,
        "fan_pAUC": as_float(fan, "pAUC"),
        "fan_TPR99": as_float(fan, "TPR@99n"),
        "fan_baseline_hmean": fan_base_h,
        "fan_ldn_minus_baseline_hmean": fan_h - fan_base_h if fan_h is not None and fan_base_h is not None else None,
        "fan_delta_hmean_vs_stage29": fan_h - as_float(s29_fan, "hmean") if fan_h is not None and s29_fan else None,
        "fan_delta_hmean_vs_stage30": fan_h - as_float(s30_fan, "hmean") if fan_h is not None and s30_fan else None,
        "valve_hmean": as_float(valve, "hmean"),
        "valve_pAUC": as_float(valve, "pAUC"),
        "valve_TPR99": as_float(valve, "TPR@99n"),
        "valve_delta_TPR99_vs_stage29": as_float(valve, "TPR@99n") - as_float(s29_valve, "TPR@99n") if valve and s29_valve else None,
        "valve_delta_TPR99_vs_stage30": as_float(valve, "TPR@99n") - as_float(s30_valve, "TPR@99n") if valve and s30_valve else None,
        "best_epoch": train_verdict.get("best_epoch"),
        "best_val_loss": train_verdict.get("best_val_loss"),
        "last_val_acc": last_float(train_log, "val_acc"),
        "last_val_anchor_loss": last_float(train_log, "val_anchor_loss"),
        "last_val_pairwise_geometry_loss": last_float(train_log, "val_pairwise_geometry_loss"),
        "last_val_machine_center_loss": last_float(train_log, "val_machine_center_loss"),
    })
    # Guardrail target: keep Stage30 hmean neighborhood while recovering Stage29 tail.
    row["balanced_sweep_candidate"] = bool(
        h >= s30_h - 0.001
        and t > s30_t
        and fan_h is not None
        and fan_h >= as_float(s30_fan, "hmean") - 0.003
        and not verdict.get("strong_machine_severe_collapse")
    )
    return row


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage29_summary = read_json(Path(args.stage29_freeze_json))
    stage30_summary = read_json(Path(args.stage30_freeze_json))
    stage29_pm = read_csv(Path(args.stage29_full7_root) / "per_machine.csv")
    stage30_pm = read_csv(Path(args.stage30_full7_root) / "per_machine.csv")

    rows = [
        load_variant(name, eval_root, train_root, stage29_pm, stage30_pm, stage29_summary, stage30_summary)
        for name, eval_root, train_root in parse_variants(args.variants)
    ]
    available_rows = [r for r in rows if r.get("eval_available")]
    candidates = [r for r in available_rows if r.get("balanced_sweep_candidate")]
    best_h = max(available_rows, key=lambda r: r["mean_hmean"]) if available_rows else None
    best_t = max(available_rows, key=lambda r: r["mean_TPR99"]) if available_rows else None
    best_balanced = max(candidates, key=lambda r: (r["mean_hmean"], r["mean_TPR99"])) if candidates else None

    verdict = {
        "stage": "Stage31A guardrail sweep summary",
        "available_variant_count": len(available_rows),
        "candidate_count": len(candidates),
        "best_hmean_variant": best_h["variant"] if best_h else None,
        "best_hmean": best_h["mean_hmean"] if best_h else None,
        "best_TPR99_variant": best_t["variant"] if best_t else None,
        "best_TPR99": best_t["mean_TPR99"] if best_t else None,
        "best_balanced_candidate": best_balanced["variant"] if best_balanced else None,
        "stage29_reference": {
            "mean_hmean": stage29_summary["mean_hmean"],
            "mean_pAUC": stage29_summary["mean_pAUC"],
            "mean_TPR99": stage29_summary["mean_TPR99"],
        },
        "stage30_reference": {
            "mean_hmean": stage30_summary["mean_hmean"],
            "mean_pAUC": stage30_summary["mean_pAUC"],
            "mean_TPR99": stage30_summary["mean_TPR99"],
        },
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_fusion": True,
            "no_test_label_checkpoint_selection": True,
        },
        "interpretation": "A good Stage31 variant should stay within 0.001 hmean of Stage30, improve TPR99 over Stage30, and not lose fan recovery.",
    }

    write_csv(out_root / "stage31a_guardrail_sweep_table.csv", rows)
    write_json(out_root / "stage31a_guardrail_sweep_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE31",
        "",
        "## 1. Stage31A guardrail sweep",
        "",
        "Stage31A summarizes Stage30A-style guardrail-strength variants. It performs no training or scoring.",
        "",
        f"- available variants: `{len(available_rows)}`",
        f"- candidate count: `{len(candidates)}`",
        f"- best hmean variant: `{verdict['best_hmean_variant']}`",
        f"- best TPR@99n variant: `{verdict['best_TPR99_variant']}`",
        f"- best balanced candidate: `{verdict['best_balanced_candidate']}`",
        "",
        "Candidate rule: stay within 0.001 hmean of Stage30, improve TPR@99n over Stage30, preserve fan recovery, and avoid strong-machine collapse.",
        "",
        "## 2. References",
        "",
        f"- Stage29: hmean `{stage29_summary['mean_hmean']}`, pAUC `{stage29_summary['mean_pAUC']}`, TPR99 `{stage29_summary['mean_TPR99']}`",
        f"- Stage30: hmean `{stage30_summary['mean_hmean']}`, pAUC `{stage30_summary['mean_pAUC']}`, TPR99 `{stage30_summary['mean_TPR99']}`",
        "",
        "## 3. Policy",
        "",
        "- No LDN parameter tuning.",
        "- No selector or fusion.",
        "- No test-label checkpoint selection.",
    ])


if __name__ == "__main__":
    main()
