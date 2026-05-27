#!/usr/bin/env python
"""Stage21B: prior-year official DCASE normal-only DAP design / protocol freeze."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE21A_ROOT_DEFAULT = "results/stage21a_external_data_feasibility_audit"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
STAGE18G_ROOT_DEFAULT = "results/stage18g_ldn_diff_k1_explainability"
OUTPUT_ROOT_DEFAULT = "results/stage21b_prior_year_dcase_dap_design"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE21B_PRIOR_YEAR_DCASE_DAP_DESIGN_AUDIT.md"

CURRENT_BASELINE_ROUTE = "EAT min_clamp__mean + ldn_diff_K1_scorek5"
CURRENT_BASELINE_MEAN_HMEAN = 0.6159467288425517
TARGET_NEXT_THRESHOLD = 0.620000
LONG_TERM_TARGET = 0.650000

PRIMARY_DATASET = "DCASE 2022 Challenge Task 2 Additional Training Dataset"
PRIMARY_DEV_DATASET = "DCASE 2022 Challenge Task 2 Development Dataset"
SECONDARY_DATASETS = [
    "DCASE 2021 Challenge Task 2 Additional Training Dataset",
    "DCASE 2021 Challenge Task 2 Development Dataset",
    "DCASE 2020 Challenge Task 2 Additional Training Dataset",
    "DCASE 2020 Challenge Task 2 Development Dataset",
]
DEFERRED_DATASETS = [
    "DCASE 2023 Challenge Task 2 Development Dataset",
    "DCASE 2023 Challenge Task 2 Additional Training Dataset",
    "DCASE 2024 Challenge Task 2 Development Dataset",
    "DCASE 2024 Challenge Task 2 Additional Training Dataset",
    "DCASE 2025 Challenge Task 2 Development Dataset",
    "DCASE 2025 Challenge Task 2 Additional Training Dataset",
]


def parse_args():
    p = argparse.ArgumentParser(description="Stage21B prior-year DCASE DAP design")
    p.add_argument("--mode", required=True, choices=["check", "design"])
    p.add_argument("--stage21a_root", default=STAGE21A_ROOT_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--stage18g_root", default=STAGE18G_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    return p.parse_args()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def import_status() -> dict:
    status = {"csv_builtin": True, "json_builtin": True, "pathlib_builtin": True}
    for name in ["numpy", "pandas"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_json(path: Path):
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def required_paths(stage21a_root: Path, stage18f_root: Path, stage18g_root: Path) -> dict[str, Path]:
    return {
        "stage21a_data_feasibility_verdict.json": stage21a_root / "stage21a_data_feasibility_verdict.json",
        "recommended_safe_datasets.csv": stage21a_root / "recommended_safe_datasets.csv",
        "rejected_or_high_risk_datasets.csv": stage21a_root / "rejected_or_high_risk_datasets.csv",
        "dcase_rule_notes.md": stage21a_root / "dcase_rule_notes.md",
        "stage18f_route_freeze_verdict.json": stage18f_root / "stage18f_route_freeze_verdict.json",
        "stage18g_explainability_verdict.json": stage18g_root / "stage18g_explainability_verdict.json",
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reqs = required_paths(Path(args.stage21a_root), Path(args.stage18f_root), Path(args.stage18g_root))
    rows = []
    all_present = True
    for label, path in reqs.items():
        exists = path.exists()
        rows.append({"artifact": label, "path": str(path), "exists": exists})
        all_present = all_present and exists

    status = {
        "mode": "check",
        "stage21a_root": str(args.stage21a_root),
        "stage18f_root": str(args.stage18f_root),
        "stage18g_root": str(args.stage18g_root),
        "all_required_files_present": all_present,
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "path_check.csv", rows)


def select_dataset_rows(safe_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    safe_map = {row["dataset_name"]: row for row in safe_rows}
    selected = []
    deferred = []

    def add_selected(name: str, priority: int, stage_role: str, enabled: bool, rationale: str):
        row = safe_map[name]
        selected.append({
            "dataset_name": name,
            "year": row["year"],
            "machine_types": row["machine_types"],
            "contains_fan_like_data": row["contains_fan_like_data"],
            "contains_gearbox_like_data": row["contains_gearbox_like_data"],
            "size_estimate": row["size_estimate"],
            "priority_order": priority,
            "stage_role": stage_role,
            "enabled_by_default": enabled,
            "rationale": rationale,
        })

    add_selected(
        PRIMARY_DATASET,
        1,
        "primary_pretraining_pool",
        True,
        "Best first corpus: exact machine-family match, manageable size, and clean normal-only train clips.",
    )
    add_selected(
        PRIMARY_DEV_DATASET,
        2,
        "same_family_extension",
        False,
        "Same-family extension after the primary corpus is stable; still prior-year and machine-matched.",
    )

    for idx, name in enumerate(SECONDARY_DATASETS, start=3):
        add_selected(
            name,
            idx,
            "secondary_ablation_pool",
            False,
            "Secondary prior-year pool for incremental ablation after 2022 is understood.",
        )

    for name in DEFERRED_DATASETS:
        row = safe_map[name]
        deferred.append({
            "dataset_name": name,
            "year": row["year"],
            "machine_types": row["machine_types"],
            "contains_fan_like_data": row["contains_fan_like_data"],
            "contains_gearbox_like_data": row["contains_gearbox_like_data"],
            "size_estimate": row["size_estimate"],
            "deferred_reason": (
                "Deferred from Stage21B first implementation because the initial goal is a small, prior-year, same-family DAP sanity "
                "rather than a broad, high-volume corpus sweep."
            ),
        })
    return selected, deferred


def build_usage_policy() -> list[dict]:
    return [
        {
            "policy_area": "allowed_training_audio",
            "rule": "Use only official DCASE prior-year train-normal clips from selected corpora.",
            "status": "required",
        },
        {
            "policy_area": "forbidden_audio",
            "rule": "Never use evaluation or test audio for pretraining, adaptation, or checkpoint selection.",
            "status": "required",
        },
        {
            "policy_area": "forbidden_datasets",
            "rule": "ToyADMOS / ToyADMOS2 / MIMII / MIMII-DUE / MIMII-DG remain hard-excluded.",
            "status": "required",
        },
        {
            "policy_area": "target_task_labels",
            "rule": "Do not use anomaly labels or test labels for training or checkpoint selection.",
            "status": "required",
        },
        {
            "policy_area": "backend_freeze",
            "rule": "Keep downstream backend fixed to min_clamp__mean + ldn_diff_K1_scorek5.",
            "status": "required",
        },
        {
            "policy_area": "current_year_data",
            "rule": "Current-year DCASE 2025 corpora are deferred from the first DAP design pass to keep the external prior-year effect interpretable.",
            "status": "default_disabled",
        },
    ]


def build_schedule_rows() -> list[dict]:
    return [
        {
            "phase": "Stage21B",
            "goal": "Freeze prior-year official DCASE normal-only DAP design",
            "data_scope": "2022 additional first; 2022 dev optional extension; 2021/2020 deferred as ablations",
            "evaluation_scope": "no evaluation in this stage",
        },
        {
            "phase": "Stage21C",
            "goal": "Implement global normal-only DAP training on the primary corpus",
            "data_scope": PRIMARY_DATASET,
            "evaluation_scope": "no ASD scoring until checkpoint and extraction are available",
        },
        {
            "phase": "Stage21D",
            "goal": "Fan adapted embedding evaluation with frozen Stage18 backend",
            "data_scope": "adapted checkpoint from Stage21C",
            "evaluation_scope": "fan first, gearbox second",
        },
        {
            "phase": "Stage21E",
            "goal": "Only if fan passes, consider broader prior-year corpus extension",
            "data_scope": "2022 dev -> 2021 -> 2020 incremental ablations",
            "evaluation_scope": "fan then gearbox; no full7 until local gates pass",
        },
    ]


def build_risk_rows() -> list[dict]:
    return [
        {
            "risk": "leakage via evaluation/test clips",
            "severity": "high",
            "mitigation": "hard exclusion rule for every evaluation/test file and no test-label checkpoint selection",
        },
        {
            "risk": "fan still fails after DAP",
            "severity": "high",
            "mitigation": "fan remains the first gate; stop if fan does not beat Stage18F fan LDN route",
        },
        {
            "risk": "gearbox regression",
            "severity": "medium",
            "mitigation": "gearbox is the second gate; stop if adapted route drops below its local baseline when tested",
        },
        {
            "risk": "too-large corpus sweep obscures causal attribution",
            "severity": "medium",
            "mitigation": "freeze a small first implementation around 2022 additional only",
        },
        {
            "risk": "repeating Stage19/20 local-fix failure mode",
            "severity": "medium",
            "mitigation": "move up one level: use new prior-year normal data instead of new score variants or fan-only protocol tweaks",
        },
    ]


def build_protocol_freeze() -> dict:
    return {
        "current_baseline_route": CURRENT_BASELINE_ROUTE,
        "current_baseline_mean_hmean": CURRENT_BASELINE_MEAN_HMEAN,
        "target_next_threshold": TARGET_NEXT_THRESHOLD,
        "long_term_target": LONG_TERM_TARGET,
        "stage21b_route_name": "Prior-Year Official DCASE Normal-Only DAP",
        "primary_dataset": PRIMARY_DATASET,
        "primary_dataset_role": "first implementation corpus",
        "first_training_scope": "global normal-only DAP on prior-year official DCASE train-normal clips",
        "first_eval_machine": "fan",
        "second_eval_machine": "gearbox",
        "frozen_backend": "min_clamp__mean + ldn_diff_K1_scorek5",
        "hard_exclusions": [
            "all evaluation/test audio",
            "all anomaly labels for training",
            "ToyADMOS family",
            "MIMII family",
            "selector routes",
            "oracle routes",
            "extra fusion to chase score",
        ],
        "stage21d_gate": [
            "adapted fan ldn_diff_K1 hmean must beat Stage18F fan ldn baseline",
            "adapted fan baseline_knn should not collapse below a defensible margin",
            "gearbox should not regress below its local baseline when tested",
            "no checkpoint selection by test labels",
        ],
        "stop_conditions": [
            "fan does not improve",
            "gearbox regresses when checked",
            "checkpoint selection needs test labels",
            "full7 only looks better after selector-like logic",
        ],
    }


def build_route_plan(selected_rows: list[dict], deferred_rows: list[dict]) -> list[str]:
    lines = [
        "# Stage21B Prior-Year Official DCASE Normal-Only DAP Design",
        "",
        "## First implementation target",
        "",
        f"- primary corpus: `{PRIMARY_DATASET}`",
        "- objective: conservative domain-adaptive pretraining on normal-only machine audio",
        "- backend remains frozen: `min_clamp__mean + ldn_diff_K1_scorek5`",
        "- first evaluation machine: `fan`",
        "- second evaluation machine: `gearbox`",
        "",
        "## Enabled-by-default corpora",
        "",
    ]
    for row in selected_rows:
        if row["enabled_by_default"]:
            lines.append(f"- {row['dataset_name']}: {row['rationale']}")
    lines.extend([
        "",
        "## Deferred corpora",
        "",
    ])
    for row in deferred_rows:
        lines.append(f"- {row['dataset_name']}: {row['deferred_reason']}")
    lines.extend([
        "",
        "## Why this ordering",
        "",
        "- 2022 additional is the cleanest first external lever: same family, prior-year, normal-only train clips, moderate size.",
        "- 2021/2020 remain valuable but are staged as ablations so we can attribute gains instead of mixing everything at once.",
        "- 2023/2024/2025 are deferred to avoid turning the first Stage21 pass into a high-volume corpus sweep.",
        "",
        "## Guardrails",
        "",
        "- no test/eval audio in pretraining",
        "- no anomaly labels in training",
        "- no selector / oracle / extra fusion",
        "- no changes to the Stage18 backend during the first Stage21 pass",
    ])
    return lines


def build_audit_doc() -> list[str]:
    return [
        "# CTAF Stage21B Prior-Year Official DCASE DAP Design Audit",
        "",
        f"- current baseline route: `{CURRENT_BASELINE_ROUTE}`",
        f"- current baseline mean_hmean: `{CURRENT_BASELINE_MEAN_HMEAN:.6f}`",
        f"- target next threshold: `{TARGET_NEXT_THRESHOLD:.6f}`",
        f"- primary corpus: `{PRIMARY_DATASET}`",
        "- first design principle: small, prior-year, same-family, normal-only corpus before any broad corpus sweep",
        "- first evaluation machine: `fan`",
        "- second evaluation machine: `gearbox`",
        "- backend remains frozen to `min_clamp__mean + ldn_diff_K1_scorek5`",
        "",
        "No training, no forward, no embedding extraction, and no score recomputation occurred in this stage.",
    ]


def run_design(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage21a_root = Path(args.stage21a_root)
    safe_rows = load_csv_rows(stage21a_root / "recommended_safe_datasets.csv")
    verdict = load_json(stage21a_root / "stage21a_data_feasibility_verdict.json")

    if verdict["recommended_first_dataset"] != PRIMARY_DATASET:
        raise RuntimeError(
            f"Stage21A recommended first dataset is {verdict['recommended_first_dataset']}, "
            f"but Stage21B is frozen around {PRIMARY_DATASET}."
        )

    selected_rows, deferred_rows = select_dataset_rows(safe_rows)
    write_csv(out_root / "selected_prior_year_corpora.csv", selected_rows)
    write_csv(out_root / "deferred_safe_corpora.csv", deferred_rows)
    write_csv(out_root / "data_usage_policy.csv", build_usage_policy())
    write_csv(out_root / "stage21b_training_schedule.csv", build_schedule_rows())
    write_csv(out_root / "stage21b_risk_register.csv", build_risk_rows())
    write_json(out_root / "stage21b_protocol_freeze.json", build_protocol_freeze())
    write_markdown(out_root / "stage21b_training_route_plan.md", build_route_plan(selected_rows, deferred_rows))
    write_markdown(Path(args.doc_path), build_audit_doc())


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "design":
        run_design(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
