#!/usr/bin/env python
"""Stage21A: external machine-sound data feasibility audit for future DAP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
STAGE18G_ROOT_DEFAULT = "results/stage18g_ldn_diff_k1_explainability"
OUTPUT_ROOT_DEFAULT = "results/stage21a_external_data_feasibility_audit"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE21A_EXTERNAL_DATA_FEASIBILITY_AUDIT.md"

CURRENT_BASELINE_ROUTE = "EAT min_clamp__mean + ldn_diff_K1_scorek5"
CURRENT_BASELINE_MEAN_HMEAN = 0.6159467288425517
TARGET_NEXT_THRESHOLD = 0.620000
LONG_TERM_TARGET = 0.650000


DATASET_CANDIDATES = [
    {
        "dataset_name": "DCASE 2020 Challenge Task 2 Development Dataset",
        "source_url": "https://zenodo.org/records/3678171",
        "official_or_third_party": "official challenge dataset",
        "year": 2020,
        "machine_types": "ToyCar|ToyConveyor|Valve|Pump|Fan|Slide rail",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "8.1 GB",
        "sample_rate_if_known": "16 kHz",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Official machine-ASD corpus with direct normal-only train split and the same family of benchmark machines used in this project.",
        "_fan_relevance_score": 7.5,
    },
    {
        "dataset_name": "DCASE 2020 Challenge Task 2 Additional Training Dataset",
        "source_url": "https://zenodo.org/records/3727685",
        "official_or_third_party": "official challenge dataset",
        "year": 2020,
        "machine_types": "ToyCar|ToyConveyor|Valve|Pump|Fan|Slide rail",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "5.5 GB",
        "sample_rate_if_known": "16 kHz",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Official extra normal corpus released for the challenge; directly usable as normal-only pretraining data.",
        "_fan_relevance_score": 7.7,
    },
    {
        "dataset_name": "DCASE 2021 Challenge Task 2 Development Dataset",
        "source_url": "https://zenodo.org/records/4562016",
        "official_or_third_party": "official challenge dataset",
        "year": 2021,
        "machine_types": "Fan|Gearbox|Pump|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "7.5 GB",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Official machine-ASD corpus with train-only normal clips and the exact kind of domain-shift setup we care about.",
        "_fan_relevance_score": 8.9,
    },
    {
        "dataset_name": "DCASE 2021 Challenge Task 2 Additional Training Dataset",
        "source_url": "https://zenodo.org/records/4660992",
        "official_or_third_party": "official challenge dataset",
        "year": 2021,
        "machine_types": "Fan|Gearbox|Pump|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "5.5 GB",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Official supplementary normal corpus aligned with DCASE-style domain shift and normal-only training.",
        "_fan_relevance_score": 9.0,
    },
    {
        "dataset_name": "DCASE 2022 Challenge Task 2 Development Dataset",
        "source_url": "https://zenodo.org/records/6355122",
        "official_or_third_party": "official challenge dataset",
        "year": 2022,
        "machine_types": "Fan|Gearbox|Bearing|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "6.2 GB",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Best machine-family match to the current project; train split is normal-only and includes fan/gearbox/bearing/slider/toy/valve.",
        "_fan_relevance_score": 9.3,
    },
    {
        "dataset_name": "DCASE 2022 Challenge Task 2 Additional Training Dataset",
        "source_url": "https://zenodo.org/records/6462969",
        "official_or_third_party": "official challenge dataset",
        "year": 2022,
        "machine_types": "Fan|Gearbox|Bearing|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "5.5 GB",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Best first candidate for domain-adaptive pretraining because it matches the current machine families and provides normal-only train clips.",
        "_fan_relevance_score": 9.4,
    },
    {
        "dataset_name": "DCASE 2023 Challenge Task 2 Development Dataset",
        "source_url": "https://zenodo.org/records/7690148",
        "official_or_third_party": "official challenge dataset",
        "year": 2023,
        "machine_types": "Fan|Gearbox|Bearing|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "5.7 TB (Zenodo data volume)",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Official current-era machine-ASD corpus; useful as a larger normal corpus if we want more domain breadth.",
        "_fan_relevance_score": 9.1,
    },
    {
        "dataset_name": "DCASE 2023 Challenge Task 2 Additional Training Dataset",
        "source_url": "https://zenodo.org/records/7830345",
        "official_or_third_party": "official challenge dataset",
        "year": 2023,
        "machine_types": "Vacuum|ToyTank|ToyNscale|ToyDrone|bandsaw|grinder|shaker",
        "contains_fan_like_data": "no",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "2.0 GB",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Normal-only training split is available and the toy/industrial mix is useful for broader audio pretraining.",
        "_fan_relevance_score": 5.6,
    },
    {
        "dataset_name": "DCASE 2024 Challenge Task 2 Development Dataset",
        "source_url": "https://zenodo.org/records/10850879",
        "official_or_third_party": "official challenge dataset",
        "year": 2024,
        "machine_types": "Fan|Gearbox|Bearing|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "multi-TB (official Zenodo record)",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Same machine family as the current project and suitable for normal-only pretraining on the train split.",
        "_fan_relevance_score": 9.2,
    },
    {
        "dataset_name": "DCASE 2024 Challenge Task 2 Additional Training Dataset",
        "source_url": "https://zenodo.org/records/11259435",
        "official_or_third_party": "official challenge dataset",
        "year": 2024,
        "machine_types": "3DPrinter|AirCompressor|BrushlessMotor|HairDryer|HoveringDrone|RoboticArm|Scanner|ToothBrush|ToyCircuit",
        "contains_fan_like_data": "no",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "multi-TB (official Zenodo record)",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "allowed in current DCASE 2025 rules if only train normals are used; never use test/eval audio",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Broader machine/audio coverage with normal-only training clips; useful for auxiliary pretraining once the current-family corpora are exhausted.",
        "_fan_relevance_score": 4.8,
    },
    {
        "dataset_name": "DCASE 2025 Challenge Task 2 Development Dataset",
        "source_url": "https://zenodo.org/records/15097779",
        "official_or_third_party": "official challenge dataset",
        "year": 2025,
        "machine_types": "Fan|Gearbox|Bearing|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "multi-GB (official Zenodo record)",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "current challenge data; do not use any evaluation/test audio for training",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Most direct data-layer match to the current benchmark family; keep to the normal train split only.",
        "_fan_relevance_score": 9.2,
    },
    {
        "dataset_name": "DCASE 2025 Challenge Task 2 Additional Training Dataset",
        "source_url": "https://zenodo.org/records/15392814",
        "official_or_third_party": "official challenge dataset",
        "year": 2025,
        "machine_types": "AutoTrash|HomeCamera|ToyPet|ToyRCCar|BandSealer|Polisher|ScrewFeeder|CoffeeGrinder",
        "contains_fan_like_data": "no",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-NC-SA 4.0",
        "size_estimate": "1.4 TB",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "current challenge data; do not use any evaluation/test audio for training",
        "leakage_risk": "low",
        "recommended_use": "safe_pretraining",
        "reason": "Newest official normal-heavy corpus; good for broad domain-adaptive pretraining with clean machine/noise-only supplemental clips.",
        "_fan_relevance_score": 5.4,
    },
    {
        "dataset_name": "ToyADMOS dataset",
        "source_url": "https://zenodo.org/records/3351307",
        "official_or_third_party": "official dataset release",
        "year": 2019,
        "machine_types": "ToyCar|ToyConveyor|ToyTrain",
        "contains_fan_like_data": "no",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "see LICENSE.pdf",
        "size_estimate": "197.1 GB",
        "sample_rate_if_known": "48 kHz",
        "DCASE_rule_compatibility": "explicitly forbidden by current DCASE 2025 rules",
        "leakage_risk": "high",
        "recommended_use": "reject",
        "reason": "Toy-domain benchmark with anomalies; explicitly banned by the current DCASE challenge rule set.",
        "_fan_relevance_score": 0.0,
    },
    {
        "dataset_name": "ToyADMOS2 dataset",
        "source_url": "https://zenodo.org/records/4580270",
        "official_or_third_party": "official dataset release",
        "year": 2021,
        "machine_types": "ToyCar|ToyTrain",
        "contains_fan_like_data": "no",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "see LICENSE.pdf",
        "size_estimate": "169.4 GB",
        "sample_rate_if_known": "48 kHz",
        "DCASE_rule_compatibility": "explicitly forbidden by current DCASE 2025 rules",
        "leakage_risk": "high",
        "recommended_use": "reject",
        "reason": "Toy-domain benchmark with anomalies; explicitly banned by the current DCASE challenge rule set.",
        "_fan_relevance_score": 0.0,
    },
    {
        "dataset_name": "MIMII Dataset",
        "source_url": "https://zenodo.org/records/3384388",
        "official_or_third_party": "official dataset release",
        "year": 2019,
        "machine_types": "Valve|Pump|Fan|Slide rail",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "no",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY-SA 4.0",
        "size_estimate": "100.2 GB",
        "sample_rate_if_known": "16 kHz",
        "DCASE_rule_compatibility": "explicitly forbidden by current DCASE 2025 rules",
        "leakage_risk": "high",
        "recommended_use": "reject",
        "reason": "Industrial ASD benchmark with anomalies; explicitly banned by the current DCASE challenge rule set.",
        "_fan_relevance_score": 0.0,
    },
    {
        "dataset_name": "MIMII DUE dataset",
        "source_url": "https://zenodo.org/records/4740355",
        "official_or_third_party": "official dataset release",
        "year": 2021,
        "machine_types": "Fan|Gearbox|Pump|Slide rail|ToyCar|ToyTrain|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "not re-verified here",
        "size_estimate": "not re-verified here",
        "sample_rate_if_known": "",
        "DCASE_rule_compatibility": "explicitly forbidden by current DCASE 2025 rules",
        "leakage_risk": "high",
        "recommended_use": "reject",
        "reason": "Domain-shifted benchmark with anomaly labels; explicitly banned by the current DCASE challenge rule set.",
        "_fan_relevance_score": 0.0,
    },
    {
        "dataset_name": "MIMII DG dataset",
        "source_url": "https://zenodo.org/records/6529888",
        "official_or_third_party": "official dataset release",
        "year": 2022,
        "machine_types": "Fan|Gearbox|Bearing|Slide rail|Valve",
        "contains_fan_like_data": "yes",
        "contains_gearbox_like_data": "yes",
        "contains_slider_or_toy_like_data": "yes",
        "normal_only_available": "yes",
        "anomaly_labels_present": "yes",
        "can_use_normal_subset_only": "yes",
        "license": "CC BY 4.0",
        "size_estimate": "825.4 MB",
        "sample_rate_if_known": "not stated on page",
        "DCASE_rule_compatibility": "explicitly forbidden by current DCASE 2025 rules",
        "leakage_risk": "high",
        "recommended_use": "reject",
        "reason": "Subset of DCASE 2022 benchmark data with anomalies; explicitly banned by the current DCASE challenge rule set.",
        "_fan_relevance_score": 0.0,
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Stage21A external data feasibility audit")
    p.add_argument("--mode", required=True, choices=["check", "audit"])
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
            if key.startswith("_"):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if not k.startswith("_")})


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


def stage18_required_paths(stage18f_root: Path, stage18g_root: Path) -> dict[str, Path]:
    return {
        "stage18f_route_freeze_verdict.json": stage18f_root / "stage18f_route_freeze_verdict.json",
        "route_summary.csv": stage18f_root / "route_summary.csv",
        "stage18g_explainability_verdict.json": stage18g_root / "stage18g_explainability_verdict.json",
        "machine_delta_summary.csv": stage18g_root / "machine_delta_summary.csv",
        "fan_failure_analysis.csv": stage18g_root / "fan_failure_analysis.csv",
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage18f_root = Path(args.stage18f_root)
    stage18g_root = Path(args.stage18g_root)
    required = stage18_required_paths(stage18f_root, stage18g_root)

    path_rows = []
    all_present = True
    for label, path in required.items():
        exists = path.exists()
        path_rows.append({"artifact": label, "path": str(path), "exists": exists})
        all_present = all_present and exists

    status = {
        "mode": "check",
        "stage18f_root": str(stage18f_root),
        "stage18g_root": str(stage18g_root),
        "output_root": str(out_root),
        "required_inputs_present": all_present,
        "candidate_dataset_count": len(DATASET_CANDIDATES),
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "path_check.csv", path_rows)


def build_rule_notes() -> list[str]:
    return [
        "# DCASE Rule Notes",
        "",
        "- Safe direction: only use official DCASE development / additional-training corpora and keep to normal-only train clips.",
        "- Hard exclusion: never use evaluation or test audio for pretraining or checkpoint selection.",
        "- Current challenge rules explicitly forbid ToyADMOS / ToyADMOS2 / MIMII / MIMII-DUE / MIMII-DG family datasets.",
        "- If a corpus contains anomalies but also provides train-normal clips, only the normal train subset is eligible.",
        "- Any transductive use of current test audio should stay disabled by default.",
    ]


def build_audit_doc(best_safe: dict, safe_rows: list[dict], reject_rows: list[dict]) -> list[str]:
    lines = [
        "# CTAF Stage21A External Normal-Only Data Feasibility Audit",
        "",
        "## Goal",
        "",
        "Find official, public, machine-audio datasets that can support future normal-only pretraining.",
        "",
        "## Bottom line",
        "",
        f"- Best first dataset: `{best_safe['dataset_name']}`",
        "- Best use: train on source-normal and target-normal only",
        "- Why: exact machine-family match to the current benchmark and clean normal-only training splits",
        "",
        "## Safe candidates",
        "",
    ]
    lines.extend([f"- {row['dataset_name']}" for row in safe_rows])
    lines.extend([
        "",
        "## Rejected or high risk",
        "",
    ])
    lines.extend([f"- {row['dataset_name']}" for row in reject_rows])
    lines.extend([
        "",
        "## Rule note",
        "",
        "- The current DCASE 2025 rule set allows prior DCASE training/additional-training data if test/evaluation audio is excluded, but explicitly forbids the ToyADMOS/MIMII-family datasets listed above.",
    ])
    return lines


def build_verdict(best_safe: dict) -> dict:
    safe_rows = [row for row in DATASET_CANDIDATES if row["leakage_risk"] == "low"]
    return {
        "safe_dataset_count": len(safe_rows),
        "recommended_first_dataset": best_safe["dataset_name"],
        "recommended_first_use": "Pretrain on source-normal and target-normal clips only, then reuse the backbone for domain-adaptive normal-only pretraining before any ASD head design.",
        "fan_relevance_score": best_safe["_fan_relevance_score"],
        "leakage_risk_summary": "Low for official DCASE training/additional-training normal-only splits when evaluation/test audio is excluded; high for ToyADMOS/MIMII-family datasets because the current DCASE 2025 rules explicitly forbid them; medium for any balanced auxiliary corpus that is only normal after manual filtering.",
        "whether_stage21b_training_design_is_justified": True,
        "recommended_next_step": "Design Stage21B around the DCASE 2022 additional-training corpus first, then ablate 2021/2020 official corpora as secondary normal-only pretraining sources; keep a hard exclusion rule for every evaluation/test clip and for all banned external families.",
        "top_safe_candidates": [
            "DCASE 2022 Challenge Task 2 Additional Training Dataset",
            "DCASE 2022 Challenge Task 2 Development Dataset",
            "DCASE 2025 Challenge Task 2 Development Dataset",
        ],
        "notes": [
            "No dataset was downloaded.",
            "No training or model code was executed.",
            "No experiment results were modified.",
        ],
    }


def run_audit(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage18f_root = Path(args.stage18f_root)
    stage18g_root = Path(args.stage18g_root)

    stage18f_verdict = load_json(stage18f_root / "stage18f_route_freeze_verdict.json")
    stage18g_verdict = load_json(stage18g_root / "stage18g_explainability_verdict.json")

    if not stage18f_verdict.get("freeze_accepted"):
        raise RuntimeError("Stage18F frozen route is not accepted; Stage21A assumes Stage18F is the current anchor.")
    if not stage18g_verdict.get("explanation_audit_complete"):
        raise RuntimeError("Stage18G explanation audit is incomplete; Stage21A assumes Stage18G context is available.")

    safe_rows = [row for row in DATASET_CANDIDATES if row["leakage_risk"] == "low"]
    reject_rows = [row for row in DATASET_CANDIDATES if row["leakage_risk"] == "high"]
    best_safe = max(safe_rows, key=lambda row: row["_fan_relevance_score"])

    write_csv(out_root / "external_dataset_candidates.csv", DATASET_CANDIDATES)
    write_csv(out_root / "recommended_safe_datasets.csv", safe_rows)
    write_csv(out_root / "rejected_or_high_risk_datasets.csv", reject_rows)
    write_markdown(out_root / "dcase_rule_notes.md", build_rule_notes())
    write_json(out_root / "stage21a_data_feasibility_verdict.json", build_verdict(best_safe))
    write_markdown(Path(args.doc_path), build_audit_doc(best_safe, safe_rows, reject_rows))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "audit":
        run_audit(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
