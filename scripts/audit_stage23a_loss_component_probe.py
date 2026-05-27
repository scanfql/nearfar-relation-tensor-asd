#!/usr/bin/env python
"""Stage23A: summarize loss-component ablations for fan harm / valve gain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUTPUT_ROOT_DEFAULT = "results/stage23a_loss_component_probe"
DOC_PATH_DEFAULT = "docs/CTAF_STAGE23A_LOSS_COMPONENT_PROBE.md"

REFERENCE_ROUTES_DEFAULT = (
    "stage22b=results/stage22d_adapted_full7_eval,"
    "no_ldn_proxy=results/stage23a_no_ldn_proxy_pair_eval,"
    "no_aug=results/stage23b_no_aug_pair_eval,"
    "no_dispersion=results/stage23c_no_dispersion_pair_eval,"
    "stronger_neighbor=results/stage23d_stronger_neighbor_pair_eval"
)

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
BASELINE_ROUTE = "adapted_baseline_knn_k5"

STAGE22B_FAN_LDN_HMEAN = 0.5534453058397698
STAGE22B_VALVE_LDN_HMEAN = 0.7722037044829246
STAGE18_FAN_LDN_HMEAN = 0.5527191284766243
STAGE18_FAN_BASELINE_HMEAN = 0.5820070015781968


def parse_args():
    p = argparse.ArgumentParser(description="Stage23A loss component ablation summary")
    p.add_argument("--routes", default=REFERENCE_ROUTES_DEFAULT, help="Comma list of name=eval_root entries")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=DOC_PATH_DEFAULT)
    p.add_argument("--fan_recovery_margin", type=float, default=0.010)
    p.add_argument("--valve_drop_tolerance", type=float, default=0.020)
    return p.parse_args()


def parse_routes(text: str) -> list[tuple[str, Path]]:
    routes = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Route must be name=path, got: {item}")
        name, path = item.split("=", 1)
        routes.append((name.strip(), Path(path.strip())))
    return routes


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


def find_row(rows: list[dict], candidate: str, machine: str) -> dict | None:
    for row in rows:
        if row.get("candidate") == candidate and row.get("machine") == machine:
            return row
    return None


def as_float(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        return float(row[key])
    except Exception:
        return None


def summarize_route(name: str, root: Path, fan_recovery_margin: float, valve_drop_tolerance: float) -> dict:
    per_machine = read_csv(root / "per_machine.csv")
    fan_ldn = find_row(per_machine, FROZEN_ROUTE, "fan")
    fan_base = find_row(per_machine, BASELINE_ROUTE, "fan")
    valve_ldn = find_row(per_machine, FROZEN_ROUTE, "valve")
    valve_base = find_row(per_machine, BASELINE_ROUTE, "valve")

    fan_h = as_float(fan_ldn, "hmean")
    fan_base_h = as_float(fan_base, "hmean")
    valve_h = as_float(valve_ldn, "hmean")
    valve_base_h = as_float(valve_base, "hmean")

    fan_delta_vs_stage22b = fan_h - STAGE22B_FAN_LDN_HMEAN if fan_h is not None else None
    valve_delta_vs_stage22b = valve_h - STAGE22B_VALVE_LDN_HMEAN if valve_h is not None else None
    fan_delta_vs_local_baseline = fan_h - fan_base_h if fan_h is not None and fan_base_h is not None else None
    valve_delta_vs_local_baseline = valve_h - valve_base_h if valve_h is not None and valve_base_h is not None else None

    fan_recovered = bool(fan_delta_vs_stage22b is not None and fan_delta_vs_stage22b >= fan_recovery_margin)
    fan_baseline_recovered = bool(fan_h is not None and fan_h >= STAGE18_FAN_BASELINE_HMEAN)
    valve_retained = bool(valve_delta_vs_stage22b is not None and valve_delta_vs_stage22b >= -valve_drop_tolerance)
    candidate_for_full7 = bool(name != "stage22b" and fan_recovered and valve_retained)

    return {
        "route": name,
        "eval_root": str(root),
        "eval_available": bool(per_machine),
        "fan_ldn_hmean": fan_h,
        "fan_baseline_hmean": fan_base_h,
        "fan_delta_vs_stage22b_ldn": fan_delta_vs_stage22b,
        "fan_delta_vs_local_adapted_baseline": fan_delta_vs_local_baseline,
        "fan_delta_vs_stage18_ldn": fan_h - STAGE18_FAN_LDN_HMEAN if fan_h is not None else None,
        "fan_delta_vs_stage18_baseline": fan_h - STAGE18_FAN_BASELINE_HMEAN if fan_h is not None else None,
        "fan_recovered_by_margin": fan_recovered,
        "fan_recovered_to_stage18_baseline": fan_baseline_recovered,
        "valve_ldn_hmean": valve_h,
        "valve_baseline_hmean": valve_base_h,
        "valve_delta_vs_stage22b_ldn": valve_delta_vs_stage22b,
        "valve_delta_vs_local_adapted_baseline": valve_delta_vs_local_baseline,
        "valve_retained_within_tolerance": valve_retained,
        "candidate_for_full7_followup": candidate_for_full7,
    }


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    routes = parse_routes(args.routes)
    rows = [summarize_route(name, root, args.fan_recovery_margin, args.valve_drop_tolerance) for name, root in routes]

    available_rows = [r for r in rows if r["eval_available"]]
    candidates = [r for r in available_rows if r["candidate_for_full7_followup"]]
    best_fan = max(available_rows, key=lambda r: r["fan_ldn_hmean"] if r["fan_ldn_hmean"] is not None else -1.0) if available_rows else None
    best_balanced = max(
        candidates,
        key=lambda r: r["fan_delta_vs_stage22b_ldn"] if r["fan_delta_vs_stage22b_ldn"] is not None else -1.0,
    ) if candidates else None

    verdict = {
        "stage": "Stage23A loss component probe",
        "fan_recovery_margin": args.fan_recovery_margin,
        "valve_drop_tolerance": args.valve_drop_tolerance,
        "available_route_count": len(available_rows),
        "candidate_for_full7_followup_count": len(candidates),
        "best_fan_recovery_route": best_fan["route"] if best_fan else None,
        "best_fan_recovery_hmean": best_fan["fan_ldn_hmean"] if best_fan else None,
        "best_balanced_followup_route": best_balanced["route"] if best_balanced else None,
        "interpretation": (
            "A removed component is suspicious for fan harm if fan_delta_vs_stage22b_ldn is positive "
            "while valve_delta_vs_stage22b_ldn stays within tolerance."
        ),
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_test_label_checkpoint_selection": True,
        },
    }

    write_csv(out_root / "stage23a_loss_component_probe_table.csv", rows)
    write_json(out_root / "stage23a_loss_component_probe_verdict.json", verdict)
    lines = [
        "# CTAF Stage23A Loss Component Probe",
        "",
        f"- fan recovery margin: `{args.fan_recovery_margin}`",
        f"- valve drop tolerance: `{args.valve_drop_tolerance}`",
        f"- available routes: `{len(available_rows)}`",
        f"- candidate full7 followups: `{len(candidates)}`",
        f"- best fan recovery route: `{verdict['best_fan_recovery_route']}`",
        f"- best balanced followup route: `{verdict['best_balanced_followup_route']}`",
        "",
        "## Route Table",
        "",
        "| route | fan hmean | fan delta vs Stage22B | valve hmean | valve delta vs Stage22B | full7 followup |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fan_h = "NA" if row["fan_ldn_hmean"] is None else f"{row['fan_ldn_hmean']:.12f}"
        fan_d = "NA" if row["fan_delta_vs_stage22b_ldn"] is None else f"{row['fan_delta_vs_stage22b_ldn']:+.12f}"
        valve_h = "NA" if row["valve_ldn_hmean"] is None else f"{row['valve_ldn_hmean']:.12f}"
        valve_d = "NA" if row["valve_delta_vs_stage22b_ldn"] is None else f"{row['valve_delta_vs_stage22b_ldn']:+.12f}"
        lines.append(f"| `{row['route']}` | {fan_h} | {fan_d} | {valve_h} | {valve_d} | `{row['candidate_for_full7_followup']}` |")
    lines.extend([
        "",
        "No selector, LDN parameter tuning, or test-label checkpoint selection is introduced.",
    ])
    write_markdown(Path(args.doc_path), lines)


if __name__ == "__main__":
    main()
