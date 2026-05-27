#!/usr/bin/env python
"""Stage31B: freeze guardrail sweep result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE31_ROOT_DEFAULT = "results/stage31a_guardrail_sweep_summary"
STAGE30_FREEZE_DEFAULT = "results/stage30b_guardrailed_freeze/stage30b_guardrailed_freeze_summary.json"
STAGE29_FREEZE_DEFAULT = "results/stage29b_pseudo_attribute_freeze/stage29b_pseudo_attribute_freeze_summary.json"
OUTPUT_ROOT_DEFAULT = "results/stage31b_guardrail_sweep_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE31.md"


def parse_args():
    p = argparse.ArgumentParser(description="Stage31B guardrail sweep freeze")
    p.add_argument("--stage31_root", default=STAGE31_ROOT_DEFAULT)
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


def f(row: dict, key: str) -> float:
    return float(row[key])


def main():
    args = parse_args()
    stage31_root = Path(args.stage31_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(stage31_root / "stage31a_guardrail_sweep_verdict.json")
    rows = read_csv(stage31_root / "stage31a_guardrail_sweep_table.csv")
    stage30 = read_json(Path(args.stage30_freeze_json))
    stage29 = read_json(Path(args.stage29_freeze_json))
    available = [r for r in rows if r.get("eval_available") == "True"]
    if not available:
        raise RuntimeError("No available Stage31 variants to freeze.")

    best_h = max(available, key=lambda r: f(r, "mean_hmean"))
    best_t = max(available, key=lambda r: f(r, "mean_TPR99"))
    least_anchor = min(available, key=lambda r: f(r, "last_val_anchor_loss"))
    best_fan = max(available, key=lambda r: f(r, "fan_hmean"))

    over_guarded = all(
        f(r, "last_val_anchor_loss") < float(stage30["last_epoch_val_anchor_loss"])
        and f(r, "last_val_acc") <= 0.001
        for r in available
    )
    all_below_stage30_hmean = all(f(r, "delta_hmean_vs_stage30") < 0 for r in available)
    all_below_stage30_tpr = all(f(r, "delta_TPR99_vs_stage30") < 0 for r in available)

    summary = {
        "stage": "Stage31B guardrail sweep freeze",
        "freeze_status": "over_guarded_negative_freeze",
        "available_variant_count": verdict["available_variant_count"],
        "candidate_count": verdict["candidate_count"],
        "best_hmean_variant": best_h["variant"],
        "best_hmean": f(best_h, "mean_hmean"),
        "best_hmean_delta_vs_stage30": f(best_h, "delta_hmean_vs_stage30"),
        "best_TPR99_variant": best_t["variant"],
        "best_TPR99": f(best_t, "mean_TPR99"),
        "best_TPR99_delta_vs_stage30": f(best_t, "delta_TPR99_vs_stage30"),
        "best_fan_variant": best_fan["variant"],
        "best_fan_hmean": f(best_fan, "fan_hmean"),
        "best_fan_delta_vs_stage30": f(best_fan, "fan_delta_hmean_vs_stage30"),
        "least_anchor_variant": least_anchor["variant"],
        "least_anchor_val_anchor_loss": f(least_anchor, "last_val_anchor_loss"),
        "stage30_val_anchor_loss": stage30["last_epoch_val_anchor_loss"],
        "stage29_val_anchor_loss": stage29["last_epoch_val_anchor_loss"],
        "stage30_mean_hmean": stage30["mean_hmean"],
        "stage30_mean_pAUC": stage30["mean_pAUC"],
        "stage30_mean_TPR99": stage30["mean_TPR99"],
        "stage29_mean_hmean": stage29["mean_hmean"],
        "stage29_mean_pAUC": stage29["mean_pAUC"],
        "stage29_mean_TPR99": stage29["mean_TPR99"],
        "over_guarded": over_guarded,
        "all_variants_below_stage30_hmean": all_below_stage30_hmean,
        "all_variants_below_stage30_TPR99": all_below_stage30_tpr,
        "diagnosis": "Stage31 guardrail sweep was too constrained: anchor loss became much smaller than Stage30, pseudo-attribute accuracy collapsed to zero, and tail did not recover.",
        "recommended_next": "Do not further lower anchor/pairwise weights blindly. Next try a Stage30-near variant that preserves Stage30 anchor range while adding a tail-aware normal-only objective or milder CE schedule.",
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_fusion": True,
            "no_test_label_checkpoint_selection": True,
        },
        "forbidden_followups": [
            "Do not continue lowering anchor weight based on this sweep.",
            "Do not add selector or fusion to choose Stage29/Stage30 per machine.",
            "Do not tune LDN K/r/score_k.",
            "Do not use test labels for checkpoint selection.",
        ],
    }

    write_json(out_root / "stage31b_guardrail_sweep_freeze_summary.json", summary)
    write_csv(out_root / "stage31b_guardrail_sweep_table.csv", rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE31",
        "",
        "## 1. Stage31A sweep result",
        "",
        "Stage31A tested three lower-guardrail variants after Stage30:",
        "",
        "- `stage31_anchor030`: anchor 0.30, pairwise 0.50, center 0.25",
        "- `stage31_anchor020`: anchor 0.20, pairwise 0.50, center 0.25",
        "- `stage31_anchor030_pair025`: anchor 0.30, pairwise 0.25, center 0.25",
        "",
        f"- available variants: `{summary['available_variant_count']}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- best hmean variant: `{summary['best_hmean_variant']}`",
        f"- best hmean: `{summary['best_hmean']}`",
        f"- best hmean delta vs Stage30: `{summary['best_hmean_delta_vs_stage30']}`",
        f"- best TPR@99n variant: `{summary['best_TPR99_variant']}`",
        f"- best TPR@99n: `{summary['best_TPR99']}`",
        f"- best TPR@99n delta vs Stage30: `{summary['best_TPR99_delta_vs_stage30']}`",
        f"- freeze status: `{summary['freeze_status']}`",
        "",
        "Conclusion: Stage31 is negative. None of the lower-guardrail variants stayed near Stage30 hmean or recovered Stage30/Stage29 tail.",
        "",
        "## 2. Diagnosis",
        "",
        f"- Stage30 val anchor loss: `{summary['stage30_val_anchor_loss']}`",
        f"- lowest Stage31 val anchor loss: `{summary['least_anchor_val_anchor_loss']}`",
        "",
        "The sweep became over-guarded rather than more expressive: validation anchor loss fell far below Stage30 and pseudo-attribute validation accuracy collapsed to zero. That removed useful representation shaping and did not recover TPR@99n.",
        "",
        "## 3. Current route hierarchy",
        "",
        "- hmean best: `Stage30 guardrailed pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- pAUC/TPR@99n best in this branch: `Stage29 pseudo-attribute EAT + frozen ldn_diff_K1_scorek5`",
        "- Stage31 lower-guardrail sweep: `negative freeze`",
        "",
        "## 4. Recommended next step",
        "",
        "Return to Stage30-near settings. Do not lower guardrails further. The next useful experiment should preserve Stage30's anchor-loss range while adding a normal-only tail/dispersion objective or a milder pseudo-attribute CE schedule.",
        "",
        "## 5. Forbidden next moves",
        "",
        "- Do not continue lowering anchor weight based on this sweep.",
        "- Do not add selector or fusion to choose Stage29/Stage30 per machine.",
        "- Do not tune `LDN` `K / r / score_k`.",
        "- Do not use test labels for checkpoint selection.",
    ])


if __name__ == "__main__":
    main()
