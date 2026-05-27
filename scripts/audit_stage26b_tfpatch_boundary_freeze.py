#!/usr/bin/env python
"""Stage26B: freeze Stage26A TF-patch local-boundary result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE26A_ROOT_DEFAULT = "results/stage26a_tfpatch_eat_local_boundary"
OUTPUT_ROOT_DEFAULT = "results/stage26b_tfpatch_boundary_freeze"
DOC_PATH_DEFAULT = "docs/CODEX_HANDOFF_STAGE26.md"


def parse_args():
    p = argparse.ArgumentParser(description="Stage26B TF-patch boundary freeze")
    p.add_argument("--stage26a_root", default=STAGE26A_ROOT_DEFAULT)
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


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def get_machine(rows: list[dict], machine: str) -> dict:
    for row in rows:
        if row.get("machine") == machine:
            return row
    raise KeyError(machine)


def main():
    args = parse_args()
    root = Path(args.stage26a_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    verdict = read_json(root / "stage26a_tfpatch_local_boundary_verdict.json")
    per_machine = read_csv(root / "per_machine.csv")
    comparison = read_csv(root / "comparison_vs_clean_clip_ldn.csv")
    train_log = read_csv(root / "train_log.csv")

    fan = get_machine(comparison, "fan")
    valve = get_machine(comparison, "valve")
    fan_score = get_machine(per_machine, "fan")
    valve_score = get_machine(per_machine, "valve")
    last_train = {}
    for row in train_log:
        last_train[row["machine"]] = row

    summary = {
        "stage": "Stage26B TF-patch boundary freeze",
        "stage26a_status": "negative_freeze",
        "candidate": verdict["candidate"],
        "machines": verdict["machines"],
        "mean_hmean": verdict["mean_hmean"],
        "mean_pAUC": verdict["mean_pAUC"],
        "mean_TPR99": verdict["mean_TPR99"],
        "fan_valve_gate_pass": verdict["fan_valve_gate_pass"],
        "fan_hmean": as_float(fan_score, "hmean"),
        "fan_delta_hmean_vs_clean_clip_ldn": as_float(fan, "delta_hmean_vs_clean_clip_ldn"),
        "fan_q99": as_float(fan_score, "q99"),
        "fan_anomaly_median": as_float(fan_score, "anomaly_median"),
        "valve_hmean": as_float(valve_score, "hmean"),
        "valve_delta_hmean_vs_clean_clip_ldn": as_float(valve, "delta_hmean_vs_clean_clip_ldn"),
        "valve_delta_TPR99_vs_clean_clip_ldn": as_float(valve, "delta_TPR99_vs_clean_clip_ldn"),
        "valve_q99": as_float(valve_score, "q99"),
        "valve_anomaly_median": as_float(valve_score, "anomaly_median"),
        "fan_final_val_bce": float(last_train["fan"]["val_bce"]) if "fan" in last_train else None,
        "valve_final_val_bce": float(last_train["valve"]["val_bce"]) if "valve" in last_train else None,
        "final_positive_chunk_fraction": float(last_train["fan"]["val_positive_chunk_fraction"]) if "fan" in last_train else None,
        "diagnosis": "TF-patch boundary head learns the synthetic pseudo task but the learned score does not transfer to real anomalies.",
        "conclusion": "Stage26A is negative; local pseudo-boundary scoring before frozen EAT is not sufficient.",
        "recommended_next_architecture": "Stop local-head-only pseudo-boundary. To continue toward 0.7, either fine-tune the representation with a more realistic local objective or introduce a stronger representation source; do not tune patch geometry.",
        "forbidden_followups": [
            "Do not tune batch_size as if it affects route quality.",
            "Do not tune patch_time_frac/patch_freq_frac/top_percent as open-ended score search.",
            "Do not add selector or fusion to hide valve collapse.",
            "Do not call the pseudo task success a real anomaly-detection success.",
        ],
    }

    write_json(out_root / "stage26b_tfpatch_boundary_freeze_summary.json", summary)
    write_csv(out_root / "stage26b_machine_comparison_vs_clean_clip_ldn.csv", comparison)
    write_markdown(Path(args.doc_path), [
        "# CODEX HANDOFF STAGE26",
        "",
        "## 1. Stage26A result",
        "",
        "Stage26A tested real TF-patch pseudo-boundary learning before frozen EAT on `fan,valve`.",
        "",
        f"- candidate: `{verdict['candidate']}`",
        f"- machines: `{','.join(verdict['machines'])}`",
        f"- mean_hmean: `{verdict['mean_hmean']}`",
        f"- mean_pAUC: `{verdict['mean_pAUC']}`",
        f"- mean_TPR@99n: `{verdict['mean_TPR99']}`",
        f"- fan_valve_gate_pass: `{verdict['fan_valve_gate_pass']}`",
        f"- fan hmean: `{summary['fan_hmean']}`",
        f"- fan delta hmean vs clean clip LDN: `{summary['fan_delta_hmean_vs_clean_clip_ldn']}`",
        f"- valve hmean: `{summary['valve_hmean']}`",
        f"- valve delta hmean vs clean clip LDN: `{summary['valve_delta_hmean_vs_clean_clip_ldn']}`",
        f"- valve delta TPR@99n vs clean clip LDN: `{summary['valve_delta_TPR99_vs_clean_clip_ldn']}`",
        "",
        "Stage26A is negative. The head learned the synthetic pseudo task, but real anomalies scored below normal q99 and valve collapsed severely.",
        "",
        "## 2. Training diagnostic",
        "",
        f"- fan final val BCE: `{summary['fan_final_val_bce']}`",
        f"- valve final val BCE: `{summary['valve_final_val_bce']}`",
        f"- final positive chunk fraction: `{summary['final_positive_chunk_fraction']}`",
        "",
        "Interpretation: synthetic clean-vs-patch discrimination is learnable, but it does not transfer to real ASD anomalies under frozen EAT local scoring.",
        "",
        "## 3. Architecture implication",
        "",
        "The local-head-only path is exhausted:",
        "",
        "- Stage24A raw clean EAT segment kNN was negative.",
        "- Stage25A embedding-space pseudo-boundary was negative with weak learned signal.",
        "- Stage26A real TF-patch pseudo-boundary before frozen EAT was negative and collapsed valve.",
        "",
        "To continue toward 0.7, the next route must change representation learning itself or introduce a stronger representation source. Merely adding a local head on frozen EAT is not enough.",
        "",
        "## 4. Forbidden next moves",
        "",
        "- Do not tune `batch_size` as if it affects route quality.",
        "- Do not tune `patch_time_frac / patch_freq_frac / top_percent` as open-ended score search.",
        "- Do not add selector or fusion to hide valve collapse.",
        "- Do not call pseudo-task BCE success a real anomaly-detection success.",
        "",
        "## 5. Recommended next question",
        "",
        "Can a representation-level objective, not just a local head, learn anomaly-relevant local evidence without destroying the clean EAT clip-level geometry?",
    ])


if __name__ == "__main__":
    main()
