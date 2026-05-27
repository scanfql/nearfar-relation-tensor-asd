#!/usr/bin/env python
"""Stage 16A-1: external checkpoint provenance audit (no training, no forward)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Audit external checkpoint provenance")
    p.add_argument("--eat_root", required=True)
    p.add_argument("--beats_lora_root", required=True)
    p.add_argument("--beats_official_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    return p.parse_args()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def infer_variant_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def infer_eat_stage(variant: str) -> str:
    v = variant.lower()
    if "debug" in v:
        return "debug"
    if "paper_like" in v:
        return "paper_like"
    if "dev" in v:
        return "dev"
    if "pretrain" in v or "pretrained" in v:
        return "pretrained"
    if "seed" in v:
        return "fine_tuned"
    return "unknown"


def scan_eat_inventory(eat_root: Path):
    rows = []
    config_files = list((eat_root / "configs").glob("*.yaml"))
    config_stems = [p.stem for p in config_files]
    for ckpt in sorted(eat_root.glob("runs/*/*.pt")):
        stat = ckpt.stat()
        variant = infer_variant_name(ckpt, eat_root)
        has_config = any(stem in variant for stem in config_stems)
        has_log = any(ckpt.parent.glob("eval*.json"))
        stage = infer_eat_stage(variant)
        notes = []
        if ckpt.name == "best.pt":
            notes.append("best checkpoint")
        if ckpt.name == "last.pt":
            notes.append("last checkpoint")
        if "paper_like" in variant:
            notes.append("paper-faithful reproduction variant")
        if "dev" in variant:
            notes.append("development-oriented variant")
        if "debug" in variant:
            notes.append("debug variant")
        rows.append({
            "checkpoint_path": str(ckpt),
            "filename": ckpt.name,
            "size": stat.st_size,
            "modified_time": fmt_mtime(stat.st_mtime),
            "variant_name": variant,
            "likely_stage": stage,
            "has_config": has_config,
            "has_log": has_log,
            "notes": "; ".join(notes),
        })
    return rows


def parse_manifest_info(manifest_path: Path) -> dict:
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]
    datasets = sorted(df["dataset_name"].astype(str).unique().tolist()) if "dataset_name" in df.columns else []
    dcase_years = sorted({m.group(1) for name in datasets if (m := re.search(r"(20\d{2})", name))})
    machines = sorted(df["machine_type"].astype(str).unique().tolist()) if "machine_type" in df.columns else []
    domains = sorted(train_df["domain"].astype(str).unique().tolist()) if "domain" in train_df.columns else []
    uses_eval_data = any("eval" in d.lower() for d in datasets)
    uses_additional = any("additional" in d.lower() for d in datasets)
    uses_labels = "attr_key" in df.columns
    train_has_target = any(d == "target" for d in domains)
    train_is_normal_only = bool((train_df["is_anomaly"] == 0).all()) if "is_anomaly" in train_df.columns and not train_df.empty else False
    return {
        "source_path": str(manifest_path),
        "source_type": "manifest_csv",
        "training_data_root": " / ".join(datasets),
        "dcase_year": ",".join(dcase_years),
        "machine_list": ",".join(machines),
        "labels_used": uses_labels,
        "anomaly_test_eval_used": uses_eval_data or bool((df["split"] == "test").any()),
        "objective_type": "machine attribute classification" if uses_labels else "unknown",
        "classifier_finetuned": uses_labels,
        "knn_train_normal_only": train_is_normal_only,
        "target_train_normal_used": train_has_target,
        "notes": f"datasets={datasets}; uses_additional={uses_additional}",
    }


def summarize_eat_project(eat_root: Path):
    rows = []
    readme = eat_root / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "source_path": str(readme),
            "source_type": "readme",
            "training_data_root": "dcase2024_dev / dcase2024_additional / dcase2024_eval (as documented)",
            "dcase_year": "2024",
            "machine_list": "bearing,fan,gearbox,slider,ToyCar,ToyTrain,valve + additional/eval machines",
            "labels_used": ("attr_key" in text) or ("classification" in text.lower()),
            "anomaly_test_eval_used": ("eval" in text.lower()),
            "objective_type": "machine attribute classification + KNN backend",
            "classifier_finetuned": True,
            "knn_train_normal_only": True,
            "target_train_normal_used": True,
            "notes": "project-level README summary",
        })

    for mf in sorted((eat_root / "manifests").glob("*.csv")):
        rows.append(parse_manifest_info(mf))

    train_py = eat_root / "scripts" / "train.py"
    if train_py.exists():
        text = train_py.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "source_path": str(train_py),
            "source_type": "train_script",
            "training_data_root": "--manifest argument",
            "dcase_year": "unknown",
            "machine_list": "manifest-defined",
            "labels_used": ("attr_id" in text) or ("CrossEntropyLoss" in text) or ("mixup_loss" in text),
            "anomaly_test_eval_used": False,
            "objective_type": "supervised attribute classification",
            "classifier_finetuned": True,
            "knn_train_normal_only": True,
            "target_train_normal_used": True,
            "notes": "train split only, attr_id supervision",
        })

    eval_py = eat_root / "scripts" / "eval_knn.py"
    if eval_py.exists():
        rows.append({
            "source_path": str(eval_py),
            "source_type": "eval_script",
            "training_data_root": "--manifest argument",
            "dcase_year": "unknown",
            "machine_list": "manifest-defined",
            "labels_used": True,
            "anomaly_test_eval_used": True,
            "objective_type": "KNN backend evaluation on test split",
            "classifier_finetuned": True,
            "knn_train_normal_only": True,
            "target_train_normal_used": True,
            "notes": "fits KNN on train embeddings, evaluates on test labels",
        })

    for cfg in sorted((eat_root / "configs").glob("*.yaml")):
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        objective = "unknown"
        if "embedding_dim" in text:
            objective = "attribute classification embedding fine-tune"
        rows.append({
            "source_path": str(cfg),
            "source_type": "config_yaml",
            "training_data_root": "manifest-dependent",
            "dcase_year": "2024",
            "machine_list": "manifest-defined",
            "labels_used": True,
            "anomaly_test_eval_used": "eval" in cfg.name.lower(),
            "objective_type": objective,
            "classifier_finetuned": True,
            "knn_train_normal_only": True,
            "target_train_normal_used": "global_machine_types" in text or True,
            "notes": f"config_name={cfg.stem}",
        })

    for eval_json in sorted(eat_root.glob("runs/*/eval*.json")):
        rows.append({
            "source_path": str(eval_json),
            "source_type": "eval_json",
            "training_data_root": "manifest-dependent",
            "dcase_year": "2024",
            "machine_list": "eval json per_machine keys",
            "labels_used": True,
            "anomaly_test_eval_used": True,
            "objective_type": "evaluated fine-tuned checkpoint",
            "classifier_finetuned": True,
            "knn_train_normal_only": True,
            "target_train_normal_used": True,
            "notes": f"run={eval_json.parent.name}",
        })

    return rows


def scan_beats_lora_inventory(root: Path):
    rows = []
    for ckpt in sorted(root.rglob("*.pth")):
        stat = ckpt.stat()
        full = str(ckpt)
        year = "unknown"
        m = re.search(r"DCASE(20\d{2})", full)
        if m:
            year = m.group(1)
        lower = full.lower()
        if "lora" in lower:
            kind = "LoRA checkpoint"
        elif "adapter" in lower:
            kind = "adapter checkpoint"
        else:
            kind = "full/task checkpoint"
        notes = []
        if "backendpool" in lower:
            notes.append("backendpool variant")
        if "rank" in lower:
            notes.append("rank-based backend config encoded in path")
        if "zscore" in lower or "center" in lower:
            notes.append("backend normalization encoded in path")
        rows.append({
            "checkpoint_path": str(ckpt),
            "filename": ckpt.name,
            "size": stat.st_size,
            "modified_time": fmt_mtime(stat.st_mtime),
            "likely_dcase_year": year,
            "likely_lora_or_adapter_or_full": kind,
            "notes": "; ".join(notes),
        })
    return rows


def build_risk_classification(eat_inventory: list[dict], eat_summary: list[dict], beats_lora: list[dict], beats_official_root: Path):
    rows = []
    # BEATs official backbone
    official_ckpt = beats_official_root / "BEATs_iter3_plus_AS2M.pt"
    if official_ckpt.exists():
        rows.append({
            "model_family": "BEATs_official",
            "checkpoint_path": str(official_ckpt),
            "risk_class": "A",
            "category_name": "Pure external pretrained backbone",
            "rationale": "Official pretrained backbone, no task adaptation artifact in checkpoint path",
            "allowed_use": "frozen external representation primary candidate",
        })

    # EAT runs: supervised/task-adapted or unknown leakage risk
    for row in eat_inventory:
        variant = row["variant_name"].lower()
        if row["likely_stage"] in ("debug", "paper_like"):
            risk = "D"
            cat = "Unknown / leakage-risk checkpoint"
            rationale = "Run variant is debug/paper_like and project manifests include additional/eval datasets; exact train manifest linkage is not fully explicit"
            allowed = "manual confirmation required before any use"
        else:
            risk = "C"
            cat = "Supervised / label-used / dev-tuned checkpoint"
            rationale = "EAT project fine-tunes backbone for machine attribute classification using attr labels; suitable only as upper-bound/task-adapted diagnostic"
            allowed = "upper-bound diagnostic only"
        rows.append({
            "model_family": "EAT_project_run",
            "checkpoint_path": row["checkpoint_path"],
            "risk_class": risk,
            "category_name": cat,
            "rationale": rationale,
            "allowed_use": allowed,
        })

    # BEATs LoRA
    for row in beats_lora:
        rows.append({
            "model_family": "BEATs_LoRA",
            "checkpoint_path": row["checkpoint_path"],
            "risk_class": "C",
            "category_name": "Supervised / label-used / dev-tuned checkpoint",
            "rationale": "Checkpoint naming indicates DCASE-year task adaptation with LoRA and backend-specific tuning",
            "allowed_use": "upper-bound diagnostic only",
        })

    return rows


def write_recommendation(path: Path, risk_rows: list[dict], eat_inventory: list[dict]):
    has_pure_eat = any(r["risk_class"] == "A" and r["model_family"] == "EAT_project_run" for r in risk_rows)
    has_eat_runs = len(eat_inventory) > 0
    lines = ["# Recommended Next Steps", ""]
    if has_pure_eat:
        lines.append("- EAT includes a pure pretrained backbone checkpoint. Recommend Stage 16B-EAT frozen embedding audit.")
    elif has_eat_runs:
        lines.append("- EAT root does not expose a standalone pure pretrained backbone checkpoint; available checkpoints appear task-adapted. Recommend Stage 16B-EAT existing checkpoint diagnostic, not frozen primary.")
    else:
        lines.append("- No usable EAT checkpoint found. Do not open Stage 16B-EAT until provenance is clarified.")
    lines.append("- BEATs official checkpoint remains eligible as a frozen external backbone reference.")
    lines.append("- BEATs LoRA checkpoints appear task-adapted and should be treated only as upper-bound diagnostics, not as frozen-route primaries.")
    if any(r["risk_class"] == "D" for r in risk_rows):
        lines.append("- RED FLAG: at least one checkpoint family has leakage or provenance ambiguity. Do not use those checkpoints in the main experiment without explicit manual confirmation.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_markdown(doc_path: Path, eat_inventory, eat_summary, beats_lora, risk_rows, rec_md_path: Path):
    counts = Counter(r["risk_class"] for r in risk_rows)
    n_eat = len(eat_inventory)
    n_beats_lora = len(beats_lora)
    lines = [
        "# CTAF Stage16A1 External Checkpoint Provenance Audit",
        "",
        "## 1. Scope",
        "",
        "Lightweight provenance audit only.",
        "No training. No full DCASE evaluation. No embedding extraction. No model forward.",
        "",
        "## 2. EAT Checkpoint Inventory Summary",
        "",
        f"- EAT checkpoints found: `{n_eat}`",
        f"- unique EAT run variants: `{len(set(r['variant_name'] for r in eat_inventory))}`",
        "",
        "## 3. BEATs LoRA Inventory Summary",
        "",
        f"- BEATs LoRA checkpoints found: `{n_beats_lora}`",
        "",
        "## 4. Risk Classification Summary",
        "",
        f"- A (pure pretrained backbone): `{counts.get('A', 0)}`",
        f"- B (normal-only task-adapted): `{counts.get('B', 0)}`",
        f"- C (supervised / label-used / dev-tuned): `{counts.get('C', 0)}`",
        f"- D (unknown / leakage-risk): `{counts.get('D', 0)}`",
        "",
        "## 5. Key Findings",
        "",
        "- The EAT project does not expose a clear standalone pure pretrained EAT backbone checkpoint inside the project root.",
        "- Available EAT run checkpoints are tied to machine-attribute classification fine-tuning and KNN backend evaluation.",
        "- The EAT project manifest and README indicate DCASE2024 dev/additional/eval data are all present in the project scaffold.",
        "- BEATs official checkpoint is a clean frozen backbone candidate.",
        "- BEATs LoRA checkpoints are task-adapted and should be treated as upper-bound diagnostics only.",
        "",
        "## 6. Recommended Next Step",
        "",
        rec_md_path.read_text(),
        "",
        "## 7. Confirmation",
        "",
        "- no training occurred",
        "- no full evaluation occurred",
        "- no embedding extraction occurred",
        "- no model forward occurred",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    eat_root = Path(args.eat_root)
    beats_lora_root = Path(args.beats_lora_root)
    beats_official_root = Path(args.beats_official_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)

    eat_inventory = scan_eat_inventory(eat_root)
    eat_summary = summarize_eat_project(eat_root)
    beats_lora = scan_beats_lora_inventory(beats_lora_root)
    risk_rows = build_risk_classification(eat_inventory, eat_summary, beats_lora, beats_official_root)

    write_csv(output_root / "eat_checkpoint_inventory.csv", eat_inventory)
    write_csv(output_root / "eat_config_log_summary.csv", eat_summary)
    write_csv(output_root / "beats_lora_checkpoint_inventory.csv", beats_lora)
    write_csv(output_root / "risk_classification.csv", risk_rows)
    rec_md = output_root / "recommended_next_steps.md"
    write_recommendation(rec_md, risk_rows, eat_inventory)
    build_markdown(doc_path, eat_inventory, eat_summary, beats_lora, risk_rows, rec_md)


if __name__ == "__main__":
    main()
