#!/usr/bin/env python
"""Stage24B: freeze Stage24A clean EAT segment-evidence result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE24A_ROOT_DEFAULT = "results/stage24a_eat_segment_evidence"
STAGE23C_ROOT_DEFAULT = "results/stage23c_route_diagnosis_freeze"
OUTPUT_ROOT_DEFAULT = "results/stage24b_segment_evidence_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE24.md"


def parse_args():
    p = argparse.ArgumentParser(description="Stage24B segment evidence freeze")
    p.add_argument("--stage24a_root", default=STAGE24A_ROOT_DEFAULT)
    p.add_argument("--stage23c_root", default=STAGE23C_ROOT_DEFAULT)
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


def safe_float(row: dict, key: str):
    try:
        return float(row[key])
    except Exception:
        return None


def main():
    args = parse_args()
    stage24a_root = Path(args.stage24a_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(stage24a_root / "stage24a_segment_evidence_verdict.json")
    per_machine = read_csv(stage24a_root / "per_machine.csv")
    comparison = read_csv(stage24a_root / "comparison_vs_clean_clip_ldn.csv")
    stage23c_path = Path(args.stage23c_root) / "stage23c_route_diagnosis_freeze_summary.json"
    stage23c = read_json(stage23c_path) if stage23c_path.exists() else {}

    ranked_by_delta = sorted(
        comparison,
        key=lambda r: safe_float(r, "delta_hmean_vs_clean_clip_ldn") if safe_float(r, "delta_hmean_vs_clean_clip_ldn") is not None else -999.0,
        reverse=True,
    )
    strongest_machine_drops = [
        r for r in comparison
        if r.get("machine") in {"bearing", "toytrain", "valve"}
        and safe_float(r, "delta_hmean_vs_clean_clip_ldn") is not None
        and safe_float(r, "delta_hmean_vs_clean_clip_ldn") < -0.02
    ]

    summary = {
        "stage": "Stage24B segment evidence freeze",
        "stage24a_candidate": verdict["candidate"],
        "stage24a_status": "negative_freeze",
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "delta_hmean_vs_clean_ldn_best": verdict["delta_hmean_vs_clean_ldn_best"],
        "delta_hmean_vs_stage22_hmean_best": verdict["delta_hmean_vs_stage22_hmean_best"],
        "improved_machine_count_vs_clean_clip_ldn": verdict["improved_machine_count_vs_clean_clip_ldn"],
        "bottleneck_TPR99_improved_count_vs_clean_clip_ldn": verdict["bottleneck_TPR99_improved_count_vs_clean_clip_ldn"],
        "strong_machine_severe_collapse": verdict["strong_machine_severe_collapse"],
        "segment_evidence_positive": verdict["segment_evidence_positive"],
        "promotable": verdict["promotable"],
        "best_machine_delta_vs_clean_clip_ldn": ranked_by_delta[0] if ranked_by_delta else None,
        "worst_machine_delta_vs_clean_clip_ldn": ranked_by_delta[-1] if ranked_by_delta else None,
        "strong_machine_drop_count": len(strongest_machine_drops),
        "stage23_diagnosis": stage23c.get("diagnosis"),
        "conclusion": "Clean EAT segment top-percent kNN does not expose useful local evidence; scoring-only segment architecture is not a path to 0.7.",
        "recommended_next_architecture": "pseudo-boundary or supervised-free local anomaly evidence training on EAT frames/chunks, not raw segment kNN scoring.",
        "forbidden_followups": [
            "Do not tune top_percent/segment_len/stride as an open-ended score search.",
            "Do not add per-machine selector or fusion to hide strong-machine collapse.",
            "Do not treat old transient segment-scoring success as transferable to clean EAT segment kNN.",
            "Do not continue Stage22 loss ablations for a 0.7 target.",
        ],
    }

    write_json(out_root / "stage24b_segment_evidence_freeze_summary.json", summary)
    write_csv(out_root / "stage24b_machine_delta_vs_clean_clip_ldn.csv", comparison)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE24",
        "",
        "## 1. Stage24A result",
        "",
        "Stage24A tested whether clean EAT frame chunks contain local anomaly evidence that clip-level `min_clamp__mean` averages away.",
        "",
        f"- candidate: `{verdict['candidate']}`",
        f"- mean_hmean: `{verdict['mean_hmean']}`",
        f"- mean_pAUC: `{verdict['mean_pAUC']}`",
        f"- mean_TPR@99n: `{verdict['mean_TPR99']}`",
        f"- delta_hmean_vs_clean_ldn_best: `{verdict['delta_hmean_vs_clean_ldn_best']}`",
        f"- delta_hmean_vs_stage22_hmean_best: `{verdict['delta_hmean_vs_stage22_hmean_best']}`",
        f"- improved_machine_count_vs_clean_clip_ldn: `{verdict['improved_machine_count_vs_clean_clip_ldn']}`",
        f"- bottleneck_TPR99_improved_count_vs_clean_clip_ldn: `{verdict['bottleneck_TPR99_improved_count_vs_clean_clip_ldn']}`",
        f"- strong_machine_severe_collapse: `{verdict['strong_machine_severe_collapse']}`",
        f"- segment_evidence_positive: `{verdict['segment_evidence_positive']}`",
        f"- promotable: `{verdict['promotable']}`",
        "",
        "Conclusion: Stage24A is negative. Clean EAT segment top-5% kNN does not provide a useful scoring-only path beyond clip-level LDN.",
        "",
        "## 2. Architecture implication",
        "",
        "The old V42/V43 segment lesson does not transfer directly to clean EAT segment kNN. Segment-level evidence may still matter, but raw segment kNN aggregation is too weak and causes strong-machine collapse.",
        "",
        "The current bottleneck remains architectural:",
        "",
        "- clip-level mean + LDN is near its local ceiling",
        "- Stage22 training gives only a weak hmean gain",
        "- Stage23 shows fan harm is density-backend mismatch under adapted geometry",
        "- Stage24A shows no-training clean EAT segment scoring is not enough",
        "",
        "## 3. Recommended next architecture",
        "",
        "Move to pseudo-boundary or local anomaly-evidence training on EAT frames/chunks. The next route should learn a local boundary signal rather than using raw segment kNN scores.",
        "",
        "Candidate next question:",
        "",
        "`Can same-machine pseudo-anomaly training create local EAT evidence that improves fan/toycar/gearbox without sacrificing valve?`",
        "",
        "## 4. Forbidden next moves",
        "",
        "- Do not tune `top_percent / segment_len / stride` as an open-ended score search.",
        "- Do not add per-machine selector or fusion to hide strong-machine collapse.",
        "- Do not treat old transient segment-scoring success as directly transferable to clean EAT segment kNN.",
        "- Do not continue Stage22 loss ablations for a 0.7 target.",
    ])


if __name__ == "__main__":
    main()
