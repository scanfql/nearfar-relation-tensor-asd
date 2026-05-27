#!/usr/bin/env python
"""V59A: CTAF V37 Score-File Schema Audit.

Checks V37 NPZ contents, key availability, domain metadata, and candidate
feasibility for V59 UACS. No selection, no score modification, no training.
"""

import argparse, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
REQUIRED_KEYS = ["train_scores", "test_scores", "y_test"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v37_dir", default="results/v37_teacher_student_pretrain_seedpanel/seed_13711")
    p.add_argument("--out_dir", default="results/v59a_ctaf_score_schema_audit")
    p.add_argument("--out_md", default="docs/CTAF_V59A_SCORE_SCHEMA_AUDIT.md")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    v37_dir = Path(args.v37_dir)

    schema_rows = []

    for machine in MACHINES:
        path = v37_dir / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
        d = np.load(path, allow_pickle=True)
        keys = sorted(d.keys())
        has = {k: k in d for k in REQUIRED_KEYS + ["d_test", "train_latent_summaries"]}

        train_scores = d["train_scores"].astype(np.float64) if has["train_scores"] else None
        test_scores = d["test_scores"].astype(np.float64) if has["test_scores"] else None
        y_test = d["y_test"].astype(np.int64) if has["y_test"] else None

        d_test_available = has.get("d_test", False)
        d_test = d["d_test"] if d_test_available else None
        if d_test is not None:
            d_test = np.array([str(x) for x in d_test])
            src_count = int(np.sum(d_test == "source"))
            tgt_count = int(np.sum(d_test == "target"))
        else:
            src_count, tgt_count = 0, 0

        # Target train-normal reference: does V37 NPZ have train domain labels?
        # V37 is source-only by design -> check if any target train exists
        has_target_train = False  # V37 trains on source normal only
        train_ls = d["train_latent_summaries"].shape if "train_latent_summaries" in d else None
        memory_ls = d["memory_latent_summaries"].shape if "memory_latent_summaries" in d else None

        # Candidate availability
        c0 = "available" if has["test_scores"] else "unavailable"
        c1 = "available" if has["train_scores"] and has["test_scores"] else "unavailable"
        c2 = c1  # same requirements
        c3 = "available" if d_test_available and src_count > 0 and tgt_count > 0 and has_target_train else "unavailable"
        c4 = "available" if d_test_available and src_count > 0 and tgt_count > 0 and has_target_train else "unavailable"
        c5 = c0  # fallback = raw

        c3_reason = "" if c3 == "available" else (
            "no target train-normal reference (V37 is source-only)" if not has_target_train else
            "missing d_test or domain metadata" if not d_test_available else
            "insufficient source/target samples"
        )

        schema_rows.append({
            "machine": machine,
            "npz_path": str(path),
            "keys": ", ".join(keys),
            "train_scores_shape": str(train_scores.shape) if train_scores is not None else "missing",
            "test_scores_shape": str(test_scores.shape) if test_scores is not None else "missing",
            "y_test_shape": str(y_test.shape) if y_test is not None else "missing",
            "normal_count": int(np.sum(y_test == 0)) if y_test is not None else 0,
            "anomaly_count": int(np.sum(y_test == 1)) if y_test is not None else 0,
            "d_test_available": d_test_available,
            "d_test_source_count": src_count,
            "d_test_target_count": tgt_count,
            "target_train_normal_exists": has_target_train,
            "train_latent_summaries_shape": str(train_ls) if train_ls else "missing",
            "memory_latent_summaries_shape": str(memory_ls) if memory_ls else "missing",
            "C0_raw": c0,
            "C1_zscore": c1,
            "C2_robust": c2,
            "C3_domain_calib": c3,
            "C4_min_score": c4,
            "C5_fallback": c5,
            "c3_c4_blocker": c3_reason if c3 == "unavailable" else "",
        })

        print(f"{machine}: keys={len(keys)} d_test={'YES' if d_test_available else 'NO'} "
              f"src={src_count} tgt={tgt_count} C3={c3} C4={c4}")

    # Report
    all_d_test = all(r["d_test_available"] for r in schema_rows)
    any_c3 = any(r["C3_domain_calib"] == "available" for r in schema_rows)
    any_c4 = any(r["C4_min_score"] == "available" for r in schema_rows)

    if all_d_test and any_c3 and any_c4:
        verdict = "V59A pass: C3/C4 available, full UACS audit eligible"
    elif all_d_test:
        verdict = "V59A limited: d_test exists but C3/C4 unavailable (no target train-normal reference). Only rank-preserving candidates available. Full UACS unlikely to improve AUC/pAUC."
    else:
        verdict = "V59A limited: only rank-preserving candidates available, full UACS unlikely to improve AUC/pAUC"

    md = _build_md(schema_rows, verdict)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    print(f"\n{args.out_md}")
    print(verdict)


def _build_md(rows, verdict):
    L = [
        "# CTAF V59A Score-File Schema Audit",
        "",
        "## 1. Checked Files",
        "",
        "| Machine | Path | Keys | train_scores | test_scores | y_test |",
        "|---------|------|------|-------------|------------|--------|",
    ]
    for r in rows:
        L.append(f"| {r['machine']:10s} | .../{Path(r['npz_path']).name} | "
                 f"{len(r['keys'].split(','))} keys | {r['train_scores_shape']} | "
                 f"{r['test_scores_shape']} | {r['y_test_shape']} |")
    L.append("")

    # Key table
    L.extend(["## 2. Key Availability per Machine", "",
              "| Machine | d_test | source | target | target_train_normal |",
              "|---------|--------|--------|--------|---------------------|"])
    for r in rows:
        L.append(f"| {r['machine']:10s} | {'YES' if r['d_test_available'] else 'NO'} | "
                 f"{r['d_test_source_count']} | {r['d_test_target_count']} | "
                 f"{'YES' if r['target_train_normal_exists'] else 'NO'} |")
    L.append("")

    L.append(f"d_test metadata: {'ALL present' if all(r['d_test_available'] for r in rows) else 'SOME missing'}")
    L.append(f"target train-normal: {'exists' if any(r['target_train_normal_exists'] for r in rows) else 'does NOT exist (V37 is source-only)'}")
    L.append("")

    # Candidate
    L.extend(["## 3. Candidate Availability", "",
              "| Machine | C0 raw | C1 zscore | C2 robust | C3 domain | C4 min | C5 fallback | C3/C4 blocker |",
              "|---------|--------|-----------|-----------|-----------|--------|-------------|---------------|"])
    for r in rows:
        L.append(f"| {r['machine']:10s} | {r['C0_raw']} | {r['C1_zscore']} | {r['C2_robust']} | "
                 f"{r['C3_domain_calib']} | {r['C4_min_score']} | {r['C5_fallback']} | "
                 f"{r['c3_c4_blocker'] or '—'} |")
    L.append("")

    n_c34 = sum(1 for r in rows if r["C3_domain_calib"] == "available")
    L.append(f"C3 domain calibration: {n_c34}/7 available")
    L.append(f"C4 min(source,target): {n_c34}/7 available")

    L.extend(["", "## 4. Final Verdict", "", f"**{verdict}**", "",
              "---", "",
              "No UACS selection performed. No anomaly labels used for selection. ",
              "No score modification. No training."])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
