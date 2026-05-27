#!/usr/bin/env python
"""Stage21C: DCASE 2022 additional data acquisition and integrity audit."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATA_ROOTS_DEFAULT = "/data1/test/LAES_2025/data,/data1/test/LAES_2025/data/dcase"
STAGE21B_ROOT_DEFAULT = "results/stage21b_prior_year_dcase_dap_design"
OUTPUT_ROOT_DEFAULT = "results/stage21c_dcase2022_additional_integrity_audit"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE21C_DCASE2022_ADDITIONAL_INTEGRITY_AUDIT.md"

PRIMARY_DATASET = "DCASE 2022 Challenge Task 2 Additional Training Dataset"
PRIMARY_DATASET_ID = "6462969"
EXPECTED_MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
DATASET_DIR_ALIASES = [
    "6462969",
    "dcase2022_additional",
    "dcase_2022_additional",
    "2022_additional",
    "additional_training_2022",
    "task2_additional_2022",
]
ARCHIVE_EXTENSIONS = [".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"]
SECTION_RE = re.compile(r"section[_-]?(\d+)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Stage21C DCASE2022 additional integrity audit")
    p.add_argument("--mode", required=True, choices=["check", "audit"])
    p.add_argument("--data_roots", default=DATA_ROOTS_DEFAULT)
    p.add_argument("--stage21b_root", default=STAGE21B_ROOT_DEFAULT)
    p.add_argument("--dataset_root_override", default="")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    return p.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def required_paths(stage21b_root: Path) -> dict[str, Path]:
    return {
        "stage21b_protocol_freeze.json": stage21b_root / "stage21b_protocol_freeze.json",
        "selected_prior_year_corpora.csv": stage21b_root / "selected_prior_year_corpora.csv",
        "data_usage_policy.csv": stage21b_root / "data_usage_policy.csv",
    }


def looks_like_machine_dir(name: str) -> str | None:
    lowered = name.lower().replace(" ", "")
    if "bearing" in lowered:
        return "bearing"
    if "gearbox" in lowered:
        return "gearbox"
    if lowered == "fan" or "fan" in lowered:
        return "fan"
    if "slider" in lowered or "sliderail" in lowered or "sliderail" in lowered.replace("-", ""):
        return "slider"
    if "slide" in lowered and "rail" in lowered:
        return "slider"
    if "toycar" in lowered:
        return "toycar"
    if "toytrain" in lowered:
        return "toytrain"
    if "valve" in lowered:
        return "valve"
    return None


def infer_machine(path: Path) -> str:
    for part in path.parts:
        machine = looks_like_machine_dir(part)
        if machine:
            return machine
    return "unknown"


def infer_split_group(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    name = path.name.lower()
    if "eval" in parts or "eval_data_" in joined or "eval_" in name:
        return "eval"
    if "test" in parts or "_test_" in name:
        return "test"
    if "train" in parts or "_train_" in name or joined.endswith("_train"):
        return "train"
    return "unknown"


def infer_label(path: Path) -> str:
    joined = "/".join(p.lower() for p in path.parts)
    if "anomaly" in joined:
        return "anomaly"
    if "normal" in joined:
        return "normal"
    return "unknown"


def infer_domain(path: Path) -> str:
    joined = "/".join(p.lower() for p in path.parts)
    if "source" in joined:
        return "source"
    if "target" in joined:
        return "target"
    return "unknown"


def infer_section(path: Path) -> str:
    joined = "/".join(path.parts)
    match = SECTION_RE.search(joined)
    return f"{int(match.group(1)):02d}" if match else "unknown"


def risk_and_allow(split_group: str, label: str, machine: str, section: str) -> tuple[bool, str]:
    if machine == "unknown":
        return False, "unknown_machine"
    if split_group == "eval":
        return False, "eval_audio"
    if split_group == "test":
        return False, "test_audio"
    if split_group != "train":
        return False, "non_train_split"
    if label == "anomaly":
        return False, "anomaly_file"
    if label != "normal":
        return False, "unknown_label_pattern"
    if section == "unknown":
        return False, "missing_section"
    return True, "allowed_train_normal"


def path_alias_match(path: Path) -> bool:
    lowered = str(path).lower()
    return any(alias in lowered for alias in DATASET_DIR_ALIASES)


def find_archive_candidates(data_roots: list[Path]) -> list[Path]:
    hits = []
    for root in data_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lowered = str(path).lower()
            if not any(alias in lowered for alias in DATASET_DIR_ALIASES):
                continue
            if any(lowered.endswith(ext) for ext in ARCHIVE_EXTENSIONS):
                hits.append(path)
    return sorted(set(hits))


def candidate_score(path: Path) -> int:
    score = 0
    lowered = str(path).lower()
    if PRIMARY_DATASET_ID in lowered:
        score += 100
    if "additional" in lowered:
        score += 40
    if "2022" in lowered:
        score += 20

    child_names = [child.name for child in path.iterdir() if child.is_dir()] if path.exists() else []
    matched_machines = {looks_like_machine_dir(name) for name in child_names}
    matched_machines.discard(None)
    score += 5 * len(matched_machines)

    has_train = any("train" in name.lower() for name in child_names)
    has_test = any("test" in name.lower() for name in child_names)
    if has_train:
        score += 10
    if has_test:
        score -= 5
    return score


def find_dataset_root(data_roots: list[Path], override: str) -> tuple[Path | None, list[dict]]:
    candidates = []
    if override:
        path = Path(override)
        candidates.append({
            "path": str(path),
            "exists": path.exists(),
            "selected_by_override": True,
            "score": candidate_score(path) if path.exists() else -1,
        })
        return (path if path.exists() else None, candidates)

    candidate_paths: set[Path] = set()
    for root in data_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            try:
                if len(path.relative_to(root).parts) > 4:
                    continue
            except ValueError:
                continue
            if path_alias_match(path):
                candidate_paths.add(path)

    for path in sorted(candidate_paths):
        candidates.append({
            "path": str(path),
            "exists": path.exists(),
            "selected_by_override": False,
            "score": candidate_score(path),
        })
    if not candidates:
        return None, candidates
    best = max(candidates, key=lambda row: row["score"])
    return Path(best["path"]), candidates


def scan_dataset_files(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    unexpected = []
    for wav_path in sorted(dataset_root.rglob("*.wav")):
        rel_path = wav_path.relative_to(dataset_root)
        machine = infer_machine(rel_path)
        split_group = infer_split_group(rel_path)
        label = infer_label(rel_path)
        domain = infer_domain(rel_path)
        section = infer_section(rel_path)
        allowed, risk_tag = risk_and_allow(split_group, label, machine, section)
        row = {
            "path": str(wav_path),
            "machine": machine,
            "split_group": split_group,
            "label_inferred": label,
            "domain_inferred": domain,
            "section": section,
            "allowed_train_normal": allowed,
            "risk_tag": risk_tag,
        }
        rows.append(row)

        issues = []
        if machine == "unknown":
            issues.append("unknown_machine")
        if split_group == "unknown":
            issues.append("unknown_split")
        if label == "unknown":
            issues.append("unknown_label")
        if section == "unknown":
            issues.append("missing_section")
        if issues:
            unexpected.append({
                "path": str(wav_path),
                "issues": "|".join(issues),
            })
    return rows, unexpected


def summarize_by_machine_domain_section(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["machine"], r["domain_inferred"], r["section"]) for r in rows})
    out = []
    for machine, domain, section in keys:
        rr = [r for r in rows if (r["machine"], r["domain_inferred"], r["section"]) == (machine, domain, section)]
        out.append({
            "machine": machine,
            "domain_inferred": domain,
            "section": section,
            "total_count": len(rr),
            "allowed_train_normal_count": sum(bool(r["allowed_train_normal"]) for r in rr),
            "rejected_or_risky_count": sum(not bool(r["allowed_train_normal"]) for r in rr),
            "train_count": sum(r["split_group"] == "train" for r in rr),
            "test_count": sum(r["split_group"] == "test" for r in rr),
            "eval_count": sum(r["split_group"] == "eval" for r in rr),
            "normal_count": sum(r["label_inferred"] == "normal" for r in rr),
            "anomaly_count": sum(r["label_inferred"] == "anomaly" for r in rr),
            "unknown_label_count": sum(r["label_inferred"] == "unknown" for r in rr),
        })
    return out


def build_presence_json(
    data_roots: list[Path],
    dataset_root: Path | None,
    candidate_rows: list[dict],
    archive_candidates: list[Path],
    rows: list[dict],
) -> dict:
    allowed_rows = [r for r in rows if r.get("allowed_train_normal")]
    return {
        "primary_dataset_name": PRIMARY_DATASET,
        "primary_dataset_id": PRIMARY_DATASET_ID,
        "data_roots": [str(p) for p in data_roots],
        "candidate_dataset_dirs_found": candidate_rows,
        "archive_candidates_found": [str(p) for p in archive_candidates],
        "selected_dataset_root": str(dataset_root) if dataset_root else "",
        "dataset_downloaded": bool(dataset_root or archive_candidates),
        "dataset_unpacked": bool(dataset_root),
        "wav_file_count": len(rows),
        "allowed_train_normal_count": len(allowed_rows),
        "expected_machine_presence": {
            machine: any(r["machine"] == machine for r in rows) for machine in EXPECTED_MACHINES
        },
    }


def build_integrity_verdict(dataset_root: Path | None, rows: list[dict], unexpected: list[dict]) -> dict:
    allowed_rows = [r for r in rows if r["allowed_train_normal"]]
    rejected_rows = [r for r in rows if not r["allowed_train_normal"]]
    fan_allowed = sum(r["machine"] == "fan" for r in allowed_rows)
    gearbox_allowed = sum(r["machine"] == "gearbox" for r in allowed_rows)
    no_test_eval_in_allowlist = all(r["split_group"] == "train" for r in allowed_rows)
    no_anomaly_in_allowlist = all(r["label_inferred"] == "normal" for r in allowed_rows)

    ready = (
        dataset_root is not None
        and len(allowed_rows) > 0
        and fan_allowed > 0
        and gearbox_allowed > 0
        and no_test_eval_in_allowlist
        and no_anomaly_in_allowlist
    )
    return {
        "primary_dataset_name": PRIMARY_DATASET,
        "dataset_found": dataset_root is not None,
        "selected_dataset_root": str(dataset_root) if dataset_root else "",
        "allowed_train_normal_count": len(allowed_rows),
        "rejected_or_risky_count": len(rejected_rows),
        "unexpected_structure_issue_count": len(unexpected),
        "fan_allowed_count": fan_allowed,
        "gearbox_allowed_count": gearbox_allowed,
        "no_test_eval_in_allowlist": no_test_eval_in_allowlist,
        "no_anomaly_in_allowlist": no_anomaly_in_allowlist,
        "ready_for_stage21d": ready,
        "notes": [
            "Only allowed_train_normal_filelist.csv should be consumed by the next training stage.",
            "Rejected or risky files are kept for traceability and leakage defense.",
            "No training or model code was executed in Stage21C.",
        ],
    }


def build_audit_doc(presence: dict, verdict: dict) -> list[str]:
    lines = [
        "# CTAF Stage21C DCASE 2022 Additional Data Acquisition & Integrity Audit",
        "",
        f"- primary dataset: `{PRIMARY_DATASET}`",
        f"- dataset found: `{presence['dataset_unpacked']}`",
        f"- selected dataset root: `{presence['selected_dataset_root']}`",
        f"- wav file count: `{presence['wav_file_count']}`",
        f"- allowed train-normal count: `{presence['allowed_train_normal_count']}`",
        f"- ready for Stage21D: `{verdict['ready_for_stage21d']}`",
        "",
        "This stage is file-level integrity audit only.",
        "",
        "- no training",
        "- no forward",
        "- no embedding extraction",
        "- no scoring",
        "- no Stage18/Stage19/Stage21B result files modified",
    ]
    return lines


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_roots = [Path(x) for x in parse_csv_list(args.data_roots)]
    stage21b_root = Path(args.stage21b_root)
    reqs = required_paths(stage21b_root)
    rows = []
    all_present = True
    for root in data_roots:
        exists = root.exists()
        rows.append({"artifact": "data_root", "path": str(root), "exists": exists})
        all_present = all_present and exists
    for label, path in reqs.items():
        exists = path.exists()
        rows.append({"artifact": label, "path": str(path), "exists": exists})
        all_present = all_present and exists

    status = {
        "mode": "check",
        "data_roots": [str(p) for p in data_roots],
        "stage21b_root": str(stage21b_root),
        "dataset_root_override": args.dataset_root_override,
        "all_required_files_present": all_present,
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "path_check.csv", rows)


def run_audit(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_roots = [Path(x) for x in parse_csv_list(args.data_roots)]
    stage21b_root = Path(args.stage21b_root)
    protocol_freeze = load_json(stage21b_root / "stage21b_protocol_freeze.json")
    if protocol_freeze.get("primary_dataset") != PRIMARY_DATASET:
        raise RuntimeError(
            f"Stage21B primary dataset is {protocol_freeze.get('primary_dataset')}, expected {PRIMARY_DATASET}."
        )

    dataset_root, candidate_rows = find_dataset_root(data_roots, args.dataset_root_override)
    archive_candidates = find_archive_candidates(data_roots)

    rows: list[dict] = []
    unexpected: list[dict] = []
    if dataset_root is not None:
        rows, unexpected = scan_dataset_files(dataset_root)

    presence = build_presence_json(data_roots, dataset_root, candidate_rows, archive_candidates, rows)
    summary_rows = summarize_by_machine_domain_section(rows)
    allowed_rows = [r for r in rows if r["allowed_train_normal"]]
    rejected_rows = [r for r in rows if not r["allowed_train_normal"]]
    verdict = build_integrity_verdict(dataset_root, rows, unexpected)

    write_json(out_root / "dataset_presence_check.json", presence)
    write_csv(out_root / "raw_file_inventory.csv", rows)
    write_csv(out_root / "summary_by_machine_domain_section.csv", summary_rows)
    write_csv(out_root / "allowed_train_normal_filelist.csv", allowed_rows)
    write_csv(out_root / "rejected_or_risky_filelist.csv", rejected_rows)
    write_csv(out_root / "unexpected_structure_report.csv", unexpected)
    write_json(out_root / "integrity_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_doc(presence, verdict))


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
