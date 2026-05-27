#!/usr/bin/env python
"""Stage 19A: high-ceiling training route design / data audit / protocol freeze."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATA_ROOTS_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
STAGE18G_ROOT_DEFAULT = "results/stage18g_ldn_diff_k1_explainability"
OUTPUT_ROOT_DEFAULT = "results/stage19a_training_route_design"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19A_TRAINING_ROUTE_DESIGN_AUDIT.md"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"

CURRENT_BASELINE_ROUTE = "EAT min_clamp__mean + ldn_diff_K1_scorek5"
CURRENT_BASELINE_MEAN_HMEAN = 0.6159467288425517
TARGET_NEXT_THRESHOLD = 0.620000
LONG_TERM_TARGET = 0.650000

MACHINE_PATTERNS = [
    (r"\bgearbox\b", "gearbox"),
    (r"\bbearing\b", "bearing"),
    (r"\bfan\b", "fan"),
    (r"\bslider\b", "slider"),
    (r"\bvalve\b", "valve"),
    (r"\btoycar\b", "toycar"),
    (r"\btoytrain\b", "toytrain"),
]
SECTION_RE = re.compile(r"section[_]?(\d+)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Stage19A training route design / data audit")
    p.add_argument("--mode", required=True, choices=["check", "inventory"])
    p.add_argument("--data_roots", default=DATA_ROOTS_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--stage18g_root", default=STAGE18G_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
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


def import_status():
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


def stage18_required_paths(stage18f_root: Path, stage18g_root: Path) -> dict[str, Path]:
    return {
        "stage18f_route_freeze_verdict.json": stage18f_root / "stage18f_route_freeze_verdict.json",
        "route_summary.csv": stage18f_root / "route_summary.csv",
        "per_machine_delta_paper.csv": stage18f_root / "per_machine_delta_paper.csv",
        "stage18g_explainability_verdict.json": stage18g_root / "stage18g_explainability_verdict.json",
        "machine_delta_summary.csv": stage18g_root / "machine_delta_summary.csv",
        "fan_failure_analysis.csv": stage18g_root / "fan_failure_analysis.csv",
    }


def infer_machine(path: Path) -> str:
    joined = "/".join(p.lower() for p in path.parts)
    for pattern, name in MACHINE_PATTERNS:
        if re.search(pattern, joined):
            return name
    return "unknown"


def infer_split_group(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    if "/train/" in joined or "train" in parts:
        return "train"
    if "/test/" in joined or "test" in parts:
        return "test"
    if "/dev/" in joined or "dev" in parts:
        return "dev"
    if "/eval/" in joined or "eval" in parts:
        return "eval"
    return "unknown"


def infer_label(path: Path) -> str:
    name = path.name.lower()
    if "normal" in name:
        return "normal"
    if "anomaly" in name:
        return "anomaly"
    return "unknown"


def infer_domain(path: Path) -> str:
    name = path.name.lower()
    if "source" in name:
        return "source"
    if "target" in name:
        return "target"
    return "unknown"


def infer_section(path: Path) -> str:
    joined = "/".join(path.parts)
    m = SECTION_RE.search(joined)
    return f"{int(m.group(1)):02d}" if m else "unknown"


def infer_root_id(data_root: Path) -> str:
    parts = [p for p in data_root.parts if p]
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return data_root.name


def infer_year_like(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"20\d{2}", part):
            return part
    joined = "/".join(path.parts)
    m = re.search(r"20\d{2}", joined)
    return m.group(0) if m else "unknown"


def scan_data_roots(data_roots: list[Path], machines_filter: set[str]) -> list[dict]:
    rows = []
    for data_root in data_roots:
        root_id = infer_root_id(data_root)
        for wav_path in sorted(data_root.rglob("*.wav")):
            machine = infer_machine(wav_path)
            if machine not in machines_filter:
                continue
            split_group = infer_split_group(wav_path)
            label = infer_label(wav_path)
            domain = infer_domain(wav_path)
            section = infer_section(wav_path)
            year_like = infer_year_like(wav_path)

            safe_training = label == "normal" and split_group == "train"
            transductive_candidate = split_group == "test" and label == "unknown"
            forbidden = split_group == "test" and label in {"normal", "anomaly"}
            risk = (
                "forbidden_test_labeled"
                if forbidden
                else "transductive_test_unlabeled" if transductive_candidate
                else "safe_train_normal" if safe_training
                else "needs_manual_review"
            )

            rows.append({
                "path": str(wav_path),
                "root_id": root_id,
                "year_like": year_like,
                "machine": machine,
                "split_group": split_group,
                "label": label,
                "domain": domain,
                "section": section,
                "safe_training_candidate": safe_training,
                "possible_unsupervised_adaptation": transductive_candidate,
                "forbidden_or_leakage_risk": forbidden,
                "risk_tag": risk,
            })
    return rows


def summarize_by_machine(rows: list[dict]) -> list[dict]:
    out = []
    for machine in sorted({r["machine"] for r in rows}):
        rr = [r for r in rows if r["machine"] == machine]
        out.append({
            "machine": machine,
            "total_count": len(rr),
            "normal_count": sum(r["label"] == "normal" for r in rr),
            "anomaly_count": sum(r["label"] == "anomaly" for r in rr),
            "unknown_label_count": sum(r["label"] == "unknown" for r in rr),
            "source_count": sum(r["domain"] == "source" for r in rr),
            "target_count": sum(r["domain"] == "target" for r in rr),
            "train_normal_count": sum(r["split_group"] == "train" and r["label"] == "normal" for r in rr),
            "test_count": sum(r["split_group"] == "test" for r in rr),
            "safe_training_candidate_count": sum(bool(r["safe_training_candidate"]) for r in rr),
            "possible_unsupervised_adaptation_count": sum(bool(r["possible_unsupervised_adaptation"]) for r in rr),
            "forbidden_or_leakage_risk_count": sum(bool(r["forbidden_or_leakage_risk"]) for r in rr),
        })
    return out


def summarize_by_root(rows: list[dict]) -> list[dict]:
    out = []
    for root_id in sorted({r["root_id"] for r in rows}):
        rr = [r for r in rows if r["root_id"] == root_id]
        out.append({
            "root_id": root_id,
            "total_count": len(rr),
            "normal_count": sum(r["label"] == "normal" for r in rr),
            "anomaly_count": sum(r["label"] == "anomaly" for r in rr),
            "unknown_label_count": sum(r["label"] == "unknown" for r in rr),
            "safe_training_candidate_count": sum(bool(r["safe_training_candidate"]) for r in rr),
            "possible_unsupervised_adaptation_count": sum(bool(r["possible_unsupervised_adaptation"]) for r in rr),
            "forbidden_or_leakage_risk_count": sum(bool(r["forbidden_or_leakage_risk"]) for r in rr),
        })
    return out


def build_route_plan_md(machine_rows: list[dict]) -> list[str]:
    fan_row = next((r for r in machine_rows if r["machine"] == "fan"), None)
    gearbox_row = next((r for r in machine_rows if r["machine"] == "gearbox"), None)
    fan_safe = fan_row["safe_training_candidate_count"] if fan_row else 0
    gearbox_safe = gearbox_row["safe_training_candidate_count"] if gearbox_row else 0
    return [
        "# Stage19 Training Route Plan",
        "",
        "## Route A: Conservative Domain-Adaptive SSL Continuation",
        "",
        "- backbone: clean EAT",
        "- train data: safe normal / unlabeled machine audio only",
        "- objective: teacher-anchored consistency or masked acoustic reconstruction",
        "- no anomaly labels",
        "- no machine-specific selector",
        "- output: adapted EAT embeddings",
        "- backend: same `min_clamp__mean + ldn_diff_K1_scorek5`",
        "- first sanity machine: `fan`",
        f"- fan safe train-normal count available: `{fan_safe}`",
        "- second sanity machine: `gearbox`",
        f"- gearbox safe train-normal count available: `{gearbox_safe}`",
        "",
        "## Route B: Pseudo-Attribute Clustering + Metric Adaptation",
        "",
        "- generate pseudo attributes from safe normal embeddings",
        "- train with ArcFace / MAC-style objective",
        "- high risk",
        "- not first implementation target",
        "",
        "## Route C: Teacher-Student / Consistency Adaptation",
        "",
        "- teacher = frozen Stage18F route representation",
        "- student = lightly adapted encoder",
        "- weak augmentation consistency",
        "- high risk due to previous LoRA-light near-null result",
        "- not first implementation target unless Route A fails",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_roots = [Path(x) for x in parse_csv_list(args.data_roots)]
    stage18f_root = Path(args.stage18f_root)
    stage18g_root = Path(args.stage18g_root)
    reqs = stage18_required_paths(stage18f_root, stage18g_root)

    rows = []
    all_present = all(root.exists() for root in data_roots) and stage18f_root.exists() and stage18g_root.exists()
    for root in data_roots:
        rows.append({"artifact": "data_root", "path": str(root), "exists": root.exists()})
        all_present = all_present and root.exists()
    for label, path in reqs.items():
        rows.append({"artifact": label, "path": str(path), "exists": path.exists()})
        all_present = all_present and path.exists()

    status = {
        "mode": "check",
        "data_roots": [str(x) for x in data_roots],
        "stage18f_root": str(stage18f_root),
        "stage18g_root": str(stage18g_root),
        "all_required_files_present": all_present,
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "path_check.csv", rows)


def run_inventory(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_roots = [Path(x) for x in parse_csv_list(args.data_roots)]
    stage18f_root = Path(args.stage18f_root)
    stage18g_root = Path(args.stage18g_root)
    machines = set(parse_csv_list(args.machines))

    rows = scan_data_roots(data_roots, machines)
    machine_summary = summarize_by_machine(rows)
    root_summary = summarize_by_root(rows)
    usable_rows = [r for r in rows if r["safe_training_candidate"] or r["possible_unsupervised_adaptation"]]
    risk_rows = [r for r in rows if r["forbidden_or_leakage_risk"] or r["risk_tag"] != "safe_train_normal"]

    stage18f_verdict = load_json(stage18f_root / "stage18f_route_freeze_verdict.json")
    stage18g_verdict = load_json(stage18g_root / "stage18g_explainability_verdict.json")
    stage18f_route_summary = load_csv_rows(stage18f_root / "route_summary.csv")
    stage18g_machine_delta = load_csv_rows(stage18g_root / "machine_delta_summary.csv")
    fan_failure = load_csv_rows(stage18g_root / "fan_failure_analysis.csv")

    anchor_rows = [{
        "current_baseline_route": CURRENT_BASELINE_ROUTE,
        "current_baseline_mean_hmean": CURRENT_BASELINE_MEAN_HMEAN,
        "fan_degradation_confirmed": stage18g_verdict.get("fan_degradation_confirmed"),
        "main_positive_machines": ",".join(stage18g_verdict.get("main_positive_machines", [])),
        "degradation_machines": ",".join(stage18g_verdict.get("degradation_machines", [])),
        "strong_machine_guardrail_pass": stage18g_verdict.get("strong_machine_guardrail_pass"),
    }]

    protocol_freeze = {
        "current_baseline_route": CURRENT_BASELINE_ROUTE,
        "current_baseline_mean_hmean": CURRENT_BASELINE_MEAN_HMEAN,
        "target_next_threshold": TARGET_NEXT_THRESHOLD,
        "long_term_target": LONG_TERM_TARGET,
        "first_training_stage": "Stage19B",
        "first_stage_goal": "safe one-machine training sanity",
        "first_sanity_machine": "fan",
        "second_sanity_machine": "gearbox",
        "required_gate_for_stage19B": [
            "adapted route must beat Stage18F on fan without degrading hmean/pAUC catastrophically",
            "adapted route must not reduce gearbox below Stage18F baseline if tested",
            "no label leakage",
            "no selector",
        ],
        "stop_conditions": [
            "fan does not improve",
            "training improves fan but collapses strong machines in later full7",
            "checkpoint selection requires test labels",
            "training instability or embedding collapse",
        ],
    }

    write_csv(out_root / "data_inventory_filelist.csv", rows)
    write_csv(out_root / "data_inventory_summary_by_machine.csv", machine_summary)
    write_csv(out_root / "data_inventory_summary_by_root.csv", root_summary)
    write_csv(out_root / "usable_data_for_dap.csv", usable_rows)
    write_csv(out_root / "excluded_data_risk_table.csv", risk_rows)
    write_csv(out_root / "stage18_anchor_summary.csv", anchor_rows)
    write_json(out_root / "stage19_protocol_freeze.json", protocol_freeze)
    write_markdown(out_root / "stage19_training_route_plan.md", build_route_plan_md(machine_summary))

    audit_lines = [
        "# CTAF Stage19A Training Route Design Audit",
        "",
        f"- current baseline route: `{CURRENT_BASELINE_ROUTE}`",
        f"- current baseline mean_hmean: `{CURRENT_BASELINE_MEAN_HMEAN:.6f}`",
        f"- fan degradation confirmed: `{stage18g_verdict.get('fan_degradation_confirmed')}`",
        f"- valve / gearbox / slider gains retained in Stage18G summary",
        f"- first training stage: `Stage19B`",
        f"- first sanity machine: `fan`",
        f"- second sanity machine: `gearbox`",
        "",
        "No training, no forward, no embedding extraction, and no pseudo-label generation occurred in this stage.",
    ]
    write_markdown(Path(args.doc_path), audit_lines)


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inventory":
        run_inventory(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
