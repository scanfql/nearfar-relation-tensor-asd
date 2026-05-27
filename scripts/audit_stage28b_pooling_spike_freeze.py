#!/usr/bin/env python
"""Stage28B: freeze Stage28A EAT pooling/spike audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE28A_ROOT_DEFAULT = "results/stage28a_eat_pooling_spike_audit"
STAGE27B_ROOT_DEFAULT = "results/stage27b_tfpatch_representation_freeze"
OUTPUT_ROOT_DEFAULT = "results/stage28b_pooling_spike_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE28.md"

CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572
STAGE27_MEAN_HMEAN = 0.6168603949659727
STAGE27_MEAN_PAUC = 0.5624060150375939
STAGE27_MEAN_TPR99 = 0.11857142857142856


def parse_args():
    p = argparse.ArgumentParser(description="Stage28B pooling/spike freeze")
    p.add_argument("--stage28a_root", default=STAGE28A_ROOT_DEFAULT)
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


def safe_float(row: dict | None, key: str) -> float | None:
    if row is None or key not in row or row[key] == "":
        return None
    try:
        return float(row[key])
    except Exception:
        return None


def get_row(rows: list[dict], **filters) -> dict | None:
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return None


def main():
    args = parse_args()
    stage28a_root = Path(args.stage28a_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(stage28a_root / "stage28a_eat_pooling_spike_verdict.json")
    summary_rows = read_csv(stage28a_root / "summary.csv")
    comparison = read_csv(stage28a_root / "comparison_vs_clean_clip_ldn.csv")
    stage27b_path = Path(args.stage27b_root) / "stage27b_tfpatch_representation_freeze_summary.json"
    stage27b = read_json(stage27b_path) if stage27b_path.exists() else {}

    best_ldn = get_row(summary_rows, candidate=verdict["best_ldn_candidate"])
    best_hmean = max(summary_rows, key=lambda r: safe_float(r, "mean_hmean") or -999.0)
    best_pauc = max(summary_rows, key=lambda r: safe_float(r, "mean_pAUC") or -999.0)
    best_t99 = max(summary_rows, key=lambda r: safe_float(r, "mean_TPR@99n") or -999.0)

    best_machine_rows = [r for r in comparison if r.get("candidate") == verdict["best_ldn_candidate"]]
    ranked_best_machine = sorted(
        best_machine_rows,
        key=lambda r: safe_float(r, "delta_hmean_vs_clean_clip_ldn") if safe_float(r, "delta_hmean_vs_clean_clip_ldn") is not None else -999.0,
        reverse=True,
    )
    positive_machines = [
        r["machine"] for r in best_machine_rows
        if safe_float(r, "delta_hmean_vs_clean_clip_ldn") is not None
        and safe_float(r, "delta_hmean_vs_clean_clip_ldn") > 0
    ]
    negative_machines = [
        r["machine"] for r in best_machine_rows
        if safe_float(r, "delta_hmean_vs_clean_clip_ldn") is not None
        and safe_float(r, "delta_hmean_vs_clean_clip_ldn") < 0
    ]
    strong_machine_drops = [
        r for r in best_machine_rows
        if r.get("machine") in {"bearing", "toytrain", "valve"}
        and safe_float(r, "delta_hmean_vs_clean_clip_ldn") is not None
        and safe_float(r, "delta_hmean_vs_clean_clip_ldn") < -0.02
    ]

    best_ldn_h = verdict["best_ldn_mean_hmean"]
    best_ldn_p = verdict["best_ldn_mean_pAUC"]
    best_ldn_t = verdict["best_ldn_mean_TPR99"]
    delta_h_stage27 = best_ldn_h - float(stage27b.get("mean_hmean", STAGE27_MEAN_HMEAN))
    delta_p_stage27 = best_ldn_p - float(stage27b.get("mean_pAUC", STAGE27_MEAN_PAUC))
    delta_t_stage27 = best_ldn_t - float(stage27b.get("mean_TPR99", STAGE27_MEAN_TPR99))

    weak_hmean_probe = bool(
        best_ldn_h > CLEAN_BEST_MEAN_HMEAN
        and best_ldn_p < CLEAN_BEST_MEAN_PAUC
        and best_ldn_t < CLEAN_BEST_MEAN_TPR99
        and not verdict["promotable"]
    )
    freeze_status = "weak_hmean_probe_not_promoted" if weak_hmean_probe else "negative_or_not_promoted"

    summary = {
        "stage": "Stage28B pooling/spike freeze",
        "freeze_status": freeze_status,
        "best_ldn_candidate": verdict["best_ldn_candidate"],
        "best_ldn_mean_hmean": best_ldn_h,
        "best_ldn_mean_pAUC": best_ldn_p,
        "best_ldn_mean_TPR99": best_ldn_t,
        "delta_hmean_vs_clean_ldn_best": best_ldn_h - CLEAN_BEST_MEAN_HMEAN,
        "delta_pAUC_vs_clean_ldn_best": best_ldn_p - CLEAN_BEST_MEAN_PAUC,
        "delta_TPR99_vs_clean_ldn_best": best_ldn_t - CLEAN_BEST_MEAN_TPR99,
        "delta_hmean_vs_stage27_balanced_best": delta_h_stage27,
        "delta_pAUC_vs_stage27_balanced_best": delta_p_stage27,
        "delta_TPR99_vs_stage27_balanced_best": delta_t_stage27,
        "best_hmean_candidate": best_hmean.get("candidate"),
        "best_pAUC_candidate": best_pauc.get("candidate"),
        "best_TPR99_candidate": best_t99.get("candidate"),
        "improved_machine_count_vs_clean_clip_ldn": verdict["improved_machine_count_vs_clean_clip_ldn"],
        "positive_machines_vs_clean_clip_ldn_for_best_ldn": positive_machines,
        "negative_machines_vs_clean_clip_ldn_for_best_ldn": negative_machines,
        "best_machine_delta_for_best_ldn": ranked_best_machine[0] if ranked_best_machine else None,
        "worst_machine_delta_for_best_ldn": ranked_best_machine[-1] if ranked_best_machine else None,
        "strong_machine_drop_count_for_best_ldn": len(strong_machine_drops),
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "pooling_positive": verdict["pooling_positive"],
        "promotable": verdict["promotable"],
        "conclusion": "RDP pooling exposes a tiny clean-hmean signal but loses pAUC/TPR99 and is below Stage27; Stage28A is not a new route.",
        "recommended_next_architecture": "Move to pseudo-attribute/domain-adaptive representation; use pooling/spike suppression only as a possible auxiliary diagnostic, not as the main path to 0.7.",
        "policy": {
            "no_training": True,
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_fusion": True,
            "no_test_label_checkpoint_selection": True,
        },
        "forbidden_followups": [
            "Do not tune pooling exponents or spike thresholds as an open-ended score search.",
            "Do not promote a route that only improves hmean by noise-level margin while lowering pAUC/TPR99.",
            "Do not displace Stage27 balanced best with Stage28A.",
            "Do not tune LDN K/r/score_k.",
        ],
    }

    write_json(out_root / "stage28b_pooling_spike_freeze_summary.json", summary)
    write_csv(out_root / "stage28b_best_ldn_machine_delta.csv", best_machine_rows)
    write_csv(out_root / "stage28b_all_summary_rows.csv", summary_rows)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE28",
        "",
        "## 1. Stage28A result",
        "",
        "Stage28A tested paper-inspired clean EAT frame pooling and spike-suppression candidates with the frozen `baseline_knn_k5` and `ldn_diff_K1_scorek5` backends.",
        "",
        f"- best LDN candidate: `{summary['best_ldn_candidate']}`",
        f"- mean_hmean: `{summary['best_ldn_mean_hmean']}`",
        f"- mean_pAUC: `{summary['best_ldn_mean_pAUC']}`",
        f"- mean_TPR@99n: `{summary['best_ldn_mean_TPR99']}`",
        f"- delta_hmean_vs_clean_ldn_best: `{summary['delta_hmean_vs_clean_ldn_best']}`",
        f"- delta_pAUC_vs_clean_ldn_best: `{summary['delta_pAUC_vs_clean_ldn_best']}`",
        f"- delta_TPR99_vs_clean_ldn_best: `{summary['delta_TPR99_vs_clean_ldn_best']}`",
        f"- delta_hmean_vs_stage27_balanced_best: `{summary['delta_hmean_vs_stage27_balanced_best']}`",
        f"- delta_pAUC_vs_stage27_balanced_best: `{summary['delta_pAUC_vs_stage27_balanced_best']}`",
        f"- delta_TPR99_vs_stage27_balanced_best: `{summary['delta_TPR99_vs_stage27_balanced_best']}`",
        f"- improved_machine_count_vs_clean_clip_ldn: `{summary['improved_machine_count_vs_clean_clip_ldn']}`",
        f"- strong_machine_severe_collapse: `{summary['strong_machine_severe_collapse']}`",
        f"- promotable: `{summary['promotable']}`",
        f"- freeze_status: `{freeze_status}`",
        "",
        "Conclusion: Stage28A is a weak hmean probe, not a new route. The tiny hmean improvement over clean LDN is not balanced and remains below Stage27.",
        "",
        "## 2. Machine-level pattern for best LDN candidate",
        "",
        f"- positive machines vs clean clip LDN: `{', '.join(positive_machines)}`",
        f"- negative machines vs clean clip LDN: `{', '.join(negative_machines)}`",
        f"- best machine delta: `{summary['best_machine_delta_for_best_ldn']}`",
        f"- worst machine delta: `{summary['worst_machine_delta_for_best_ldn']}`",
        "",
        "The pooling signal is real but small. It is useful as evidence that temporal weighting can move clean EAT geometry, but it is not enough for the 0.7 target.",
        "",
        "## 3. Current route hierarchy",
        "",
        "- balanced best: `Stage27 TF-patch representation-adapted EAT + frozen ldn_diff_K1_scorek5`",
        "- hmean-only older reference: `Stage22 geometry-objective adapted EAT + frozen ldn_diff_K1_scorek5`",
        "- Stage28A: `not promoted`",
        "",
        "## 4. Recommended next architecture",
        "",
        "Move to pseudo-attribute/domain-adaptive representation. The papers point more strongly toward attribute-aware representation learning than toward further pooling-score tweaks.",
        "",
        "## 5. Forbidden next moves",
        "",
        "- Do not tune pooling exponents or spike thresholds as an open-ended score search.",
        "- Do not promote a route that only improves hmean by a noise-level margin while lowering pAUC/TPR99.",
        "- Do not displace Stage27 balanced best with Stage28A.",
        "- Do not tune `LDN` `K / r / score_k`.",
    ])


if __name__ == "__main__":
    main()
