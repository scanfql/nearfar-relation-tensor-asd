#!/usr/bin/env python
"""V60B cross-year normal-only manifest builder."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

from src.data.dataset import scan_dataset, AudioItem
from src.features.logmel import load_audio, extract_logmel
from src.features.tensor_builder import build_tensor


TARGET_7 = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


def parse_args():
    p = argparse.ArgumentParser(description="Build clean cross-year normal-only manifest")
    p.add_argument("--root_2023", required=True)
    p.add_argument("--root_2024", required=True)
    p.add_argument("--root_2025", required=True)
    p.add_argument("--root_2026", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--max_tensor_checks", type=int, default=100)
    p.add_argument("--max_meta_checks", type=int, default=200)
    return p.parse_args()


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def infer_split_rule(year: str, item: AudioItem):
    path_str = str(item.path)
    machine_ok = item.machine in TARGET_7

    if year in ("2023", "2024"):
        if machine_ok and item.split == "train" and item.label in (0, None):
            return True, "include_train_normal"
        if item.label == 1:
            return False, "exclude_anomaly_clip"
        if item.split != "train":
            return False, "exclude_test_split"
        if not machine_ok:
            return False, "exclude_non_target_machine"
        return False, "exclude_other"

    if year == "2025":
        if "/15097779/" in path_str and machine_ok and item.label in (0, None):
            if item.split == "train" or "supplemental" in path_str:
                return True, "include_2025_current_dev_normal"
        if item.label == 1:
            return False, "exclude_anomaly_clip"
        if "/15392814/" in path_str or "/15519362/" in path_str:
            return False, "exclude_2025_eval_data_branch"
        if item.split != "train" and "supplemental" not in path_str:
            return False, "exclude_test_split"
        if not machine_ok:
            return False, "exclude_non_target_machine"
        return False, "exclude_other"

    if year == "2026":
        return False, "exclude_2026_reserved_for_future_protocol"

    return False, "exclude_unknown_year"


def sample_rate_duration(path: Path):
    import soundfile as sf
    info = sf.info(str(path))
    return int(info.samplerate), float(info.frames / info.samplerate)


def tensor_ok(path: Path):
    try:
        audio = load_audio(path)
        logmel = extract_logmel(audio)
        tensor = build_tensor("full_5view", logmel, audio=audio)
        return bool(tensor.shape[0] == 5 and np.isfinite(tensor).all())
    except Exception:
        return False


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    roots = {
        "2023": Path(args.root_2023),
        "2024": Path(args.root_2024),
        "2025": Path(args.root_2025),
        "2026": Path(args.root_2026),
    }

    manifest_rows = []
    excluded_rows = []
    year_machine_summary = []
    tensor_summary = []
    leakage_rows = []

    for year, root in roots.items():
        items = scan_dataset(root)
        include_items = []
        exclude_counter = Counter()
        machine_counter = defaultdict(int)
        tensor_checked = 0
        tensor_failed = 0

        for item in items:
            keep, reason = infer_split_rule(year, item)
            if keep:
                include_items.append(item)
                machine_counter[item.machine] += 1
            else:
                exclude_counter[reason] += 1
                excluded_rows.append({
                    "year": year,
                    "absolute_path": str(item.path),
                    "machine_raw": item.path.parts[-3] if len(item.path.parts) >= 3 else item.machine,
                    "machine_canonical": item.machine,
                    "split": item.split,
                    "domain": item.domain,
                    "section": item.section,
                    "label": item.label,
                    "exclude_reason": reason,
                })

        # enrich included rows
        for item in include_items:
            sr, dur = sample_rate_duration(item.path)
            ok = tensor_ok(item.path)
            tensor_checked += 1
            if not ok:
                tensor_failed += 1
            _keep, include_reason = infer_split_rule(year, item)
            manifest_rows.append({
                "year": year,
                "absolute_path": str(item.path),
                "machine_raw": item.path.parts[-3] if len(item.path.parts) >= 3 else item.machine,
                "machine_canonical": item.machine,
                "split": item.split,
                "domain": item.domain,
                "section": item.section,
                "label_status": "normal",
                "duration": f"{dur:.6f}",
                "sample_rate": sr,
                "usable_for_v61": bool(ok),
                "include_reason": include_reason,
            })

        for machine in sorted(machine_counter):
            year_machine_summary.append({
                "year": year,
                "machine_canonical": machine,
                "usable_normal_count": machine_counter[machine],
            })

        tensor_summary.append({
            "year": year,
            "selected_manifest_count": len(include_items),
            "tensor_checked": tensor_checked,
            "tensor_failed": tensor_failed,
            "full_5view_tensor_feasible": tensor_failed == 0,
        })

        risks = []
        if year == "2025":
            risks.append("contains_eval_data_branches_excluded")
        if year == "2026":
            risks.append("2026_reserved_for_future_noise_aware_protocol")
        leakage_rows.append({
            "year": year,
            "risk_summary": ";".join(risks) if risks else "none",
        })

    # final clean manifest must keep only usable_for_v61 = true
    clean_manifest = [r for r in manifest_rows if r["usable_for_v61"]]

    write_csv(output_root / "manifest_v61_clean.csv", clean_manifest)
    write_csv(output_root / "excluded_files.csv", excluded_rows)
    write_csv(output_root / "year_machine_summary.csv", year_machine_summary)
    write_csv(output_root / "tensor_feasibility_summary.csv", tensor_summary)
    write_csv(output_root / "leakage_risk_report.csv", leakage_rows)

    usable_count = len(clean_manifest)
    excluded_count = len(excluded_rows)
    by_reason = Counter(r["exclude_reason"] for r in excluded_rows)

    verdict = "V60B pass"
    if any(r["tensor_failed"] > 0 for r in tensor_summary):
        verdict = "V60B fail"
    elif any(r["risk_summary"] != "none" for r in leakage_rows):
        verdict = "V60B limited"

    lines = [
        "# CTAF V60B Cross-Year Normal Manifest",
        "",
        "## 1. Why V60B is needed after V60 limited",
        "",
        "V60 showed that cross-year data exists and is mostly compatible, but the protocol was still limited because 2025 contains eval-related branches and 2026 contains Emu/new-setting taxonomy.",
        "V60B converts that audit into a reproducible manifest lock: exactly which files enter V61-clean, and exactly which files are excluded.",
        "",
        "## 2. Exact include/exclude rules",
        "",
        "Included:",
        "",
        "- 2023 train normal only",
        "- 2024 train normal only",
        "- 2025 current/dev train normal only",
        "- metadata fields retained: year, path, machine_raw, machine_canonical, split, domain, section, duration, sample_rate",
        "",
        "Excluded:",
        "",
        "- all test split files",
        "- all anomaly clips",
        "- all 2025 eval_data_* branches",
        "- all 2026 Emu / new-setting taxonomy files",
        "- ambiguous or failed files",
        "",
        "## 3. Per-year usable normal count",
        "",
    ]
    by_year = defaultdict(int)
    for r in clean_manifest:
        by_year[r["year"]] += 1
    for year in ["2023", "2024", "2025", "2026"]:
        lines.append(f"- {year}: `{by_year[year]}`")

    lines.extend([
        "",
        "## 4. Per-machine usable normal count",
        "",
        "| year | machine | usable normal count |",
        "|---|---|---:|",
    ])
    for row in year_machine_summary:
        lines.append(f"| {row['year']} | {row['machine_canonical']} | {row['usable_normal_count']} |")

    lines.extend([
        "",
        "## 5. Excluded count by reason",
        "",
        "| reason | count |",
        "|---|---:|",
    ])
    for reason, count in sorted(by_reason.items()):
        lines.append(f"| {reason} | {count} |")

    lines.extend([
        "",
        "## 6. full_5view tensor feasibility on selected manifest",
        "",
        "| year | selected count | tensor checked | tensor failed | full_5view feasible |",
        "|---|---:|---:|---:|---|",
    ])
    for row in tensor_summary:
        lines.append(
            f"| {row['year']} | {row['selected_manifest_count']} | {row['tensor_checked']} | "
            f"{row['tensor_failed']} | {row['full_5view_tensor_feasible']} |"
        )

    lines.extend([
        "",
        "## 7. Leakage risk assessment",
        "",
        "| year | risk summary |",
        "|---|---|",
    ])
    for row in leakage_rows:
        lines.append(f"| {row['year']} | {row['risk_summary']} |")

    lines.extend([
        "",
        "## 8. Final verdict",
        "",
        f"`{verdict}`",
        "",
    ])
    if verdict == "V60B pass":
        lines.append("Manifest is safe for V61 prereg / implementation review.")
    elif verdict == "V60B limited":
        lines.append("Manifest exists, but manual cleaning / mapping is still required before training.")
    else:
        lines.append("Safe manifest cannot be produced under the current rules.")

    lines.extend([
        "",
        "## 9. Confirmation that no training occurred",
        "",
        "No model was trained. No checkpoint was modified. No anomaly label was used beyond exclusion filtering.",
    ])

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
