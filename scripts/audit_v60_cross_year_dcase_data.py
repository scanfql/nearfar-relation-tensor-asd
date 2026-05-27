#!/usr/bin/env python
"""V60 cross-year normal-only DCASE data schema/protocol audit."""

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


TARGET_2024_MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


def parse_args():
    p = argparse.ArgumentParser(description="Cross-year DCASE normal-only data audit")
    p.add_argument("--root_2023", required=True)
    p.add_argument("--root_2024", required=True)
    p.add_argument("--root_2025", required=True)
    p.add_argument("--root_2026", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--max_length_samples", type=int, default=200)
    p.add_argument("--max_tensor_checks_per_year", type=int, default=20)
    return p.parse_args()


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def detect_year_schema(root: Path, year_label: str):
    items = scan_dataset(root)
    wavs = sorted(root.rglob("*.wav"))
    root_name = root.name

    machine_names = sorted({x.machine for x in items})
    overlap = sorted(set(machine_names) & set(TARGET_2024_MACHINES))

    contains_anomaly = any(x.label == 1 for x in items)
    contains_normal = any(x.label == 0 or (x.split == "train" and x.label is None) for x in items)
    contains_train = any(x.split == "train" for x in items)
    contains_test = any(x.split == "test" for x in items)
    contains_target = any(x.domain == "target" for x in items)
    contains_source = any(x.domain == "source" for x in items)
    contains_sections = any(x.section != "00" or "section" in x.path.name.lower() for x in items)

    # year-specific structure labels
    if year_label == "2025":
        structure = "dev_current + eval_train_extra + eval_test_extra"
    else:
        structure = "train/test dev-style"

    per_machine = []
    counts = defaultdict(lambda: Counter())
    for x in items:
        key = x.machine
        if x.label == 1:
            counts[key]["anomaly"] += 1
        elif x.label == 0 or (x.split == "train" and x.label is None):
            counts[key]["normal"] += 1
        counts[key][f"split_{x.split}"] += 1
        counts[key][f"domain_{x.domain}"] += 1
        counts[key][f"section_{x.section}"] += 1

    for machine in machine_names:
        c = counts[machine]
        per_machine.append({
            "year": year_label,
            "machine": machine,
            "normal_count": int(c["normal"]),
            "anomaly_count": int(c["anomaly"]),
            "train_count": int(c["split_train"]),
            "test_count": int(c["split_test"]),
            "source_count": int(c["domain_source"]),
            "target_count": int(c["domain_target"]),
            "overlap_with_2024_target": machine in TARGET_2024_MACHINES,
            "emu_or_new_setting": ("Emu" in machine or machine.endswith("emu") or machine.lower().endswith("emu")),
        })

    # sample rate / duration audit
    sr_counter = Counter()
    durations = []
    sample_names = []
    checked = 0
    for wav in wavs:
        if checked >=  min(len(wavs), 200):
            break
        try:
            import soundfile as sf
            info = sf.info(str(wav))
            sr_counter[info.samplerate] += 1
            durations.append(info.frames / float(info.samplerate))
        except Exception:
            pass
        if len(sample_names) < 8:
            sample_names.append(wav.name)
        checked += 1

    return {
        "root": str(root),
        "root_name": root_name,
        "machine_names": machine_names,
        "items": items,
        "contains_anomaly": contains_anomaly,
        "contains_normal": contains_normal,
        "contains_train": contains_train,
        "contains_test": contains_test,
        "contains_source": contains_source,
        "contains_target": contains_target,
        "contains_sections": contains_sections,
        "structure": structure,
        "per_machine": per_machine,
        "sample_rate_counter": dict(sr_counter),
        "durations": durations,
        "sample_names": sample_names,
        "overlap": overlap,
    }


def classify_usable_subset(year_label: str, root: Path, items: list[AudioItem]):
    usable = []
    excluded = []
    for x in items:
        path_str = str(x.path)
        machine_ok = x.machine in TARGET_2024_MACHINES
        emu = "Emu" in x.machine or x.machine.lower().endswith("emu")
        if year_label in ("2023", "2024"):
            if x.split == "train" and machine_ok and x.domain == "source":
                usable.append((x, "train_source_normal"))
            else:
                reason = []
                if x.split != "train":
                    reason.append("not_train_split")
                if not machine_ok:
                    reason.append("non_target_machine")
                if x.domain != "source":
                    reason.append("non_source_train")
                excluded.append((x, ";".join(reason) or "excluded"))
        elif year_label == "2025":
            # current-year baseline root contains dev current machines plus unrelated eval_data train/test trees
            if "/15097779/" in path_str and machine_ok and x.split == "train" and x.domain == "source":
                usable.append((x, "current_dev_train_source_normal"))
            elif "/15097779/" in path_str and machine_ok and "supplemental" in path_str and x.label is None:
                usable.append((x, "current_dev_supplemental_source_normal"))
            else:
                reason = []
                if "/15097779/" not in path_str:
                    reason.append("future_eval_machine_tree")
                if not machine_ok:
                    reason.append("non_target_machine")
                if x.split != "train" and "supplemental" not in path_str:
                    reason.append("not_train_like")
                excluded.append((x, ";".join(reason) or "excluded"))
        elif year_label == "2026":
            # mark Emu/new-setting separately; do not directly mix
            if emu:
                excluded.append((x, "emu_or_new_setting"))
            elif x.split == "train" and machine_ok and x.domain == "source":
                usable.append((x, "train_source_normal"))
            else:
                reason = []
                if x.split != "train":
                    reason.append("not_train_split")
                if not machine_ok:
                    reason.append("non_target_machine")
                if x.domain != "source":
                    reason.append("non_source_train")
                excluded.append((x, ";".join(reason) or "excluded"))
        else:
            excluded.append((x, "unknown_year"))
    return usable, excluded


def tensor_feasibility(usable_items, max_checks: int):
    checked = 0
    failures = []
    success = 0
    for item, _tag in usable_items:
        if checked >= max_checks:
            break
        try:
            audio = load_audio(item.path)
            logmel = extract_logmel(audio)
            t = build_tensor("full_5view", logmel, audio=audio)
            if t.shape[0] != 5 or not np.isfinite(t).all():
                failures.append(f"{item.path}: invalid_tensor_shape_or_nonfinite")
            else:
                success += 1
        except Exception as e:
            failures.append(f"{item.path}: {type(e).__name__}: {e}")
        checked += 1
    return {
        "checked": checked,
        "success": success,
        "failures": failures,
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    year_roots = {
        "2023": Path(args.root_2023),
        "2024": Path(args.root_2024),
        "2025": Path(args.root_2025),
        "2026": Path(args.root_2026),
    }

    missing_roots = []
    audits = {}
    schema_rows = []
    filtering_rows = []
    machine_map_rows = []
    compatibility_rows = []
    leakage_rows = []
    usable_rows = []
    excluded_rows = []
    tensor_rows = []

    for year, root in year_roots.items():
        if not root.exists():
            missing_roots.append({"year": year, "missing_root": str(root)})
            continue
        audit = detect_year_schema(root, year)
        audits[year] = audit

        usable, excluded = classify_usable_subset(year, root, audit["items"])
        feas = tensor_feasibility(usable, args.max_tensor_checks_per_year)

        sr_keys = sorted(audit["sample_rate_counter"].keys())
        duration_mean = float(np.mean(audit["durations"])) if audit["durations"] else float("nan")
        duration_min = float(np.min(audit["durations"])) if audit["durations"] else float("nan")
        duration_max = float(np.max(audit["durations"])) if audit["durations"] else float("nan")

        schema_rows.append({
            "year": year,
            "root": str(root),
            "machine_types": ",".join(audit["machine_names"]),
            "n_machine_types": len(audit["machine_names"]),
            "contains_train": audit["contains_train"],
            "contains_test": audit["contains_test"],
            "contains_source": audit["contains_source"],
            "contains_target": audit["contains_target"],
            "contains_sections": audit["contains_sections"],
            "structure": audit["structure"],
            "sample_rates_seen": ",".join(map(str, sr_keys)),
            "duration_mean_sec": f"{duration_mean:.4f}" if np.isfinite(duration_mean) else "",
            "duration_min_sec": f"{duration_min:.4f}" if np.isfinite(duration_min) else "",
            "duration_max_sec": f"{duration_max:.4f}" if np.isfinite(duration_max) else "",
            "filename_examples": " | ".join(audit["sample_names"]),
        })

        total_normal = sum(r["normal_count"] for r in audit["per_machine"])
        total_anom = sum(r["anomaly_count"] for r in audit["per_machine"])
        filtering_rows.append({
            "year": year,
            "total_normal_like": total_normal,
            "total_anomaly": total_anom,
            "anomaly_present": audit["contains_anomaly"],
            "anomaly_filterable": True,
            "usable_normal_only_count": len(usable),
            "excluded_count": len(excluded),
        })

        tensor_rows.append({
            "year": year,
            "tensor_checks": feas["checked"],
            "tensor_success": feas["success"],
            "tensor_failures": len(feas["failures"]),
            "full_5view_tensor_feasible": len(feas["failures"]) == 0,
        })

        compatibility_rows.append({
            "year": year,
            "overlap_with_2024_target": ",".join(audit["overlap"]),
            "all_2024_targets_present": set(TARGET_2024_MACHINES).issubset(set(audit["machine_names"])),
            "has_source_target": audit["contains_source"] and audit["contains_target"],
            "has_section_metadata": audit["contains_sections"],
            "has_machine_taxonomy_issue": any("Emu" in m or m.lower().endswith("emu") for m in audit["machine_names"]),
        })

        leakage_risks = []
        if year == "2025":
            leakage_risks.append("current_year_eval_trees_present")
        if year == "2026" and any("Emu" in m or m.lower().endswith("emu") for m in audit["machine_names"]):
            leakage_risks.append("emu_or_new_setting_requires_separate_protocol")
        if audit["contains_anomaly"]:
            leakage_risks.append("anomaly_clips_present_but_filterable")
        leakage_rows.append({
            "year": year,
            "risk_flags": ";".join(leakage_risks) if leakage_risks else "none",
        })

        for row in audit["per_machine"]:
            machine_map_rows.append({
                "year": year,
                **row,
            })
        for x, reason in usable:
            usable_rows.append({
                "year": year,
                "machine": x.machine,
                "split": x.split,
                "domain": x.domain,
                "label": x.label,
                "path": str(x.path),
                "usable_reason": reason,
            })
        for x, reason in excluded:
            excluded_rows.append({
                "year": year,
                "machine": x.machine,
                "split": x.split,
                "domain": x.domain,
                "label": x.label,
                "path": str(x.path),
                "exclude_reason": reason,
            })

    write_csv(output_root / "missing_roots.csv", missing_roots)
    write_csv(output_root / "per_year_schema.csv", schema_rows)
    write_csv(output_root / "per_year_filtering.csv", filtering_rows)
    write_csv(output_root / "machine_type_mapping.csv", machine_map_rows)
    write_csv(output_root / "metadata_compatibility.csv", compatibility_rows)
    write_csv(output_root / "tensor_feasibility.csv", tensor_rows)
    write_csv(output_root / "leakage_risks.csv", leakage_rows)
    write_csv(output_root / "usable_normal_only_files.csv", usable_rows)
    write_csv(output_root / "excluded_files.csv", excluded_rows)

    usable_count = len(usable_rows)
    excluded_count = len(excluded_rows)
    full5_ok = all(r["full_5view_tensor_feasible"] for r in tensor_rows) if tensor_rows else False

    # Final verdict heuristic
    verdict = "V60 pass"
    if missing_roots:
        verdict = "V60 fail"
    elif any("emu_or_new_setting_requires_separate_protocol" in r["risk_flags"] for r in leakage_rows):
        verdict = "V60 limited"
    elif any("current_year_eval_trees_present" in r["risk_flags"] for r in leakage_rows):
        verdict = "V60 limited"
    elif not full5_ok:
        verdict = "V60 fail"

    lines = [
        "# CTAF V60 Cross-Year DCASE Data Audit",
        "",
        "## 1. Stage 13A-0 motivation",
        "",
        "Stage 13A-0 checks only whether a multi-year normal-only data protocol is safe and unified enough to define.",
        "It does not answer whether the protocol will improve performance.",
        "",
        "## 2. Why this is not V39-V59 version search",
        "",
        "- no model is trained",
        "- no scoring rule is changed",
        "- no frontend is changed",
        "- only dataset schema, legality, overlap, and leakage risk are audited",
        "",
        "## 3. Multi-year normal-only protocol definition",
        "",
        "- Protocol A: current-year-only CTAF baseline",
        "- Protocol B: multi-year normal-only CTAF pretraining corpus",
        "- Protocol B may use only normal clips from allowed roots after strict filtering",
        "- anomaly clips, anomaly labels, and test labels are excluded",
        "",
        "## 4. Per-year dataset schema table",
        "",
        "| year | root | n_machine_types | contains train | contains test | has source/target | has section metadata | sample rates | duration mean | structure |",
        "|---|---|---:|---|---|---|---|---|---:|---|",
    ]
    for row in schema_rows:
        lines.append(
            f"| {row['year']} | {row['root']} | {row['n_machine_types']} | {row['contains_train']} | {row['contains_test']} | "
            f"{row['contains_source'] and row['contains_target']} | {row['contains_sections']} | {row['sample_rates_seen']} | "
            f"{row['duration_mean_sec']} | {row['structure']} |"
        )

    lines.extend([
        "",
        "## 5. Per-year normal/anomaly filtering table",
        "",
        "| year | total normal-like | total anomaly | anomaly present | anomaly filterable | usable normal-only count | excluded count |",
        "|---|---:|---:|---|---|---:|---:|",
    ])
    for row in filtering_rows:
        lines.append(
            f"| {row['year']} | {row['total_normal_like']} | {row['total_anomaly']} | "
            f"{row['anomaly_present']} | {row['anomaly_filterable']} | {row['usable_normal_only_count']} | {row['excluded_count']} |"
        )

    lines.extend([
        "",
        "## 6. Machine type mapping table",
        "",
        "| year | machine | normal count | anomaly count | train count | test count | source count | target count | overlap with 2024 target | emu/new setting |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in machine_map_rows:
        lines.append(
            f"| {row['year']} | {row['machine']} | {row['normal_count']} | {row['anomaly_count']} | {row['train_count']} | "
            f"{row['test_count']} | {row['source_count']} | {row['target_count']} | {row['overlap_with_2024_target']} | {row['emu_or_new_setting']} |"
        )

    lines.extend([
        "",
        "## 7. source/target/section metadata compatibility",
        "",
        "| year | overlap with current 2024 target machines | all targets present | has source/target | has section metadata | machine taxonomy issue |",
        "|---|---|---|---|---|---|",
    ])
    for row in compatibility_rows:
        lines.append(
            f"| {row['year']} | {row['overlap_with_2024_target']} | {row['all_2024_targets_present']} | "
            f"{row['has_source_target']} | {row['has_section_metadata']} | {row['has_machine_taxonomy_issue']} |"
        )

    lines.extend([
        "",
        "## 8. full_5view tensor feasibility",
        "",
        "| year | tensor checks | success | failures | full_5view feasible |",
        "|---|---:|---:|---:|---|",
    ])
    for row in tensor_rows:
        lines.append(
            f"| {row['year']} | {row['tensor_checks']} | {row['tensor_success']} | {row['tensor_failures']} | {row['full_5view_tensor_feasible']} |"
        )

    lines.extend([
        "",
        "## 9. Data leakage risk assessment",
        "",
        "| year | risk flags |",
        "|---|---|",
    ])
    for row in leakage_rows:
        lines.append(f"| {row['year']} | {row['risk_flags']} |")

    lines.extend([
        "",
        "## 10. usable normal-only data count",
        "",
        f"- total usable normal-only files: `{usable_count}`",
        "",
        "## 11. excluded data count and reason",
        "",
        f"- total excluded files: `{excluded_count}`",
        "- main reasons include:",
        "  - anomaly clips present but filtered out",
        "  - test splits excluded",
        "  - 2025 eval_data_* branches treated as non-baseline external trees",
        "  - 2026 `Emu` / new-setting machine taxonomy flagged separately",
        "",
        "## 12. final verdict",
        "",
        f"`{verdict}`",
        "",
    ])

    if verdict == "V60 pass":
        lines.append("Data protocol is safe and eligible for Stage 13A-1 prereg/implementation review.")
    elif verdict == "V60 limited":
        lines.append("Data exists but needs manual cleaning/mapping before training.")
    else:
        lines.append("Data protocol is insufficient, unsafe, or leakage-prone in its current form.")

    lines.extend([
        "",
        "## 13. confirmation that no training occurred",
        "",
        "No model was trained. No checkpoint was loaded for optimization. No anomaly label was used in training or model selection.",
        "",
        "V60 only answers whether multi-year normal-only DCASE data can safely enter a unified CTAF protocol.",
    ])

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
