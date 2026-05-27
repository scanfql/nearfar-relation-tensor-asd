#!/usr/bin/env python
"""Stage25B: freeze Stage25A EAT segment pseudo-boundary result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE24B_ROOT_DEFAULT = "results/stage24b_segment_evidence_freeze"
STAGE25A_ROOT_DEFAULT = "results/stage25a_eat_segment_pseudo_boundary"
OUTPUT_ROOT_DEFAULT = "results/stage25b_pseudo_boundary_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE25.md"


def parse_args():
    p = argparse.ArgumentParser(description="Stage25B pseudo-boundary freeze")
    p.add_argument("--stage24b_root", default=STAGE24B_ROOT_DEFAULT)
    p.add_argument("--stage25a_root", default=STAGE25A_ROOT_DEFAULT)
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


def maybe_float(row: dict, key: str):
    try:
        return float(row[key])
    except Exception:
        return None


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage24b = read_json(Path(args.stage24b_root) / "stage24b_segment_evidence_freeze_summary.json")
    stage25a = read_json(Path(args.stage25a_root) / "stage25a_pseudo_boundary_verdict.json")
    per_machine = read_csv(Path(args.stage25a_root) / "per_machine.csv")
    comparison = read_csv(Path(args.stage25a_root) / "comparison_vs_clean_clip_ldn.csv")

    stage24a_h = float(stage24b["mean_hmean"])
    stage25a_h = float(stage25a["mean_hmean"])
    improvement_vs_raw_segment = stage25a_h - stage24a_h
    collapse_machines = [
        row for row in comparison
        if row.get("machine") in {"bearing", "toytrain", "valve"}
        and maybe_float(row, "delta_hmean_vs_clean_clip_ldn") is not None
        and maybe_float(row, "delta_hmean_vs_clean_clip_ldn") < -0.02
    ]

    summary = {
        "stage": "Stage25B pseudo-boundary freeze",
        "stage25a_status": "negative_freeze_with_weak_learned_signal",
        "stage24a_raw_segment_hmean": stage24a_h,
        "stage25a_pseudo_boundary_hmean": stage25a_h,
        "improvement_vs_stage24a_raw_segment": improvement_vs_raw_segment,
        "stage25a_mean_pAUC": stage25a["mean_pAUC"],
        "stage25a_mean_TPR99": stage25a["mean_TPR99"],
        "delta_hmean_vs_clean_ldn_best": stage25a["delta_hmean_vs_clean_ldn_best"],
        "delta_hmean_vs_stage22_hmean_best": stage25a["delta_hmean_vs_stage22_hmean_best"],
        "improved_machine_count_vs_clean_clip_ldn": stage25a["improved_machine_count_vs_clean_clip_ldn"],
        "bottleneck_TPR99_improved_count_vs_clean_clip_ldn": stage25a["bottleneck_TPR99_improved_count_vs_clean_clip_ldn"],
        "strong_machine_severe_collapse": stage25a["strong_machine_severe_collapse"],
        "pseudo_boundary_positive": stage25a["pseudo_boundary_positive"],
        "promotable": stage25a["promotable"],
        "strong_machine_drop_count_vs_clean_clip_ldn": len(collapse_machines),
        "conclusion": "Embedding-space pseudo-boundary learning improves over raw segment kNN but remains far below clip-level LDN and is not a viable route.",
        "recommended_next_architecture": "Move pseudo-boundary generation before EAT, using real TF patch replacement / local corruption and a local EAT-frame objective, then evaluate with fixed full7 gates.",
        "forbidden_followups": [
            "Do not tune feature_patch_frac/noise/top_percent as an open-ended search.",
            "Do not add selector or fusion to combine weak pseudo-boundary scores.",
            "Do not claim Stage25A is positive; it is only a weak signal over Stage24A.",
            "Do not continue embedding-space pseudo-boundary as the main 0.7 path.",
        ],
    }

    write_json(out_root / "stage25b_pseudo_boundary_freeze_summary.json", summary)
    write_csv(out_root / "stage25b_machine_comparison_vs_clean_clip_ldn.csv", comparison)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE25",
        "",
        "## 1. Stage25A result",
        "",
        "Stage25A trained a small pseudo-boundary head on clean EAT segment embeddings. It improved over raw Stage24A segment kNN, but failed against the clip-level LDN route.",
        "",
        f"- Stage24A raw segment hmean: `{stage24a_h}`",
        f"- Stage25A pseudo-boundary hmean: `{stage25a_h}`",
        f"- improvement_vs_Stage24A: `{improvement_vs_raw_segment}`",
        f"- mean_pAUC: `{stage25a['mean_pAUC']}`",
        f"- mean_TPR@99n: `{stage25a['mean_TPR99']}`",
        f"- delta_hmean_vs_clean_ldn_best: `{stage25a['delta_hmean_vs_clean_ldn_best']}`",
        f"- delta_hmean_vs_stage22_hmean_best: `{stage25a['delta_hmean_vs_stage22_hmean_best']}`",
        f"- improved_machine_count_vs_clean_clip_ldn: `{stage25a['improved_machine_count_vs_clean_clip_ldn']}`",
        f"- bottleneck_TPR99_improved_count_vs_clean_clip_ldn: `{stage25a['bottleneck_TPR99_improved_count_vs_clean_clip_ldn']}`",
        f"- strong_machine_severe_collapse: `{stage25a['strong_machine_severe_collapse']}`",
        f"- pseudo_boundary_positive: `{stage25a['pseudo_boundary_positive']}`",
        f"- promotable: `{stage25a['promotable']}`",
        "",
        "Conclusion: Stage25A is a negative freeze with weak learned signal. Embedding-space pseudo-boundary learning is better than raw segment kNN, but far below the current clip-level LDN route.",
        "",
        "## 2. Architecture implication",
        "",
        "The weak gain from Stage24A to Stage25A suggests that learning local evidence is not hopeless. However, generating pseudo-boundaries directly in embedding space is too artificial and does not create a useful anomaly score.",
        "",
        "The next architecture should move pseudo-boundary generation before EAT:",
        "",
        "- real TF patch replacement / local corruption on audio or mel",
        "- frozen or lightly trainable EAT frame objective",
        "- local evidence head trained on clean vs pseudo-local-corrupt frames/chunks",
        "- fixed fan/valve gate first, then full7 only if the gate passes",
        "",
        "## 3. Forbidden next moves",
        "",
        "- Do not tune `feature_patch_frac / noise / top_percent` as open-ended score search.",
        "- Do not add selector or fusion to combine weak pseudo-boundary scores.",
        "- Do not claim Stage25A is positive; it is only a weak signal over Stage24A.",
        "- Do not continue embedding-space pseudo-boundary as the main 0.7 path.",
        "",
        "## 4. Recommended next step",
        "",
        "Open Stage26: real TF-patch pseudo-boundary training before EAT, starting with fan/valve gate rather than full7.",
    ])


if __name__ == "__main__":
    main()
