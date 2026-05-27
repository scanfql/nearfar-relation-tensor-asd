#!/usr/bin/env python
"""Stage 16C-3: clean EAT segment-chunk fixed fusion one-machine audit.

This script is strictly score-level only.

It does NOT:
- load EAT
- run model forward
- extract embeddings
- train models
- fine-tune
- run LoRA / adapter
- search alpha
- do per-machine selection
- use oracle routing
- open 7-machine runs

It only:
- loads existing score bundles
- checks score alignment
- computes fixed rank-normalized fusion candidates
- reports one-machine gearbox metrics and verdict
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINE_DEFAULT = "gearbox"
SEED_DEFAULT = 13711
OUTPUT_ROOT_DEFAULT = "results/stage16c3_clean_eat_chunk_fixed_fusion_one_machine_gearbox"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE16C3_CLEAN_EAT_CHUNK_FIXED_FUSION_ONE_MACHINE_GEARBOX_AUDIT.md"

P1_SCORE_DEFAULT = "results/stage16b_clean_eat_pooling_fusion/scores/P1_mean_std/gearbox_scores.npz"
A1_SCORE_DEFAULT = "results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz"
C16_SCORE_DEFAULT = "results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C16S08_top1/gearbox_scores.npz"
C08_SCORE_DEFAULT = "results/stage16c2_clean_eat_segment_chunk_one_machine_gearbox/scores/C08S04_top1/gearbox_scores.npz"
FRAW_SCORE_DEFAULT = "results/stage16c_clean_eat_segment_one_machine_gearbox/scores/F_P1_S3_050/gearbox_scores.npz"

REGISTERED_REFERENCES = {
    "P1_mean_std": {"hmean": 0.561801, "pAUC": 0.493158, "TPR@99n": 0.020000},
    "F_P1_A1_065": {"hmean": 0.572219, "pAUC": 0.519474, "TPR@99n": 0.030000},
    "F_P1_S3_050": {"hmean": 0.575941, "pAUC": 0.525789, "TPR@99n": 0.080000},
    "C16S08_top1": {"hmean": 0.560879, "pAUC": 0.544211, "TPR@99n": 0.090000},
    "C08S04_top1": {"hmean": 0.551612, "pAUC": 0.546842, "TPR@99n": 0.110000},
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage16C-3 clean EAT chunk fixed fusion one-machine audit")
    p.add_argument("--mode", required=True, choices=["check", "fixed_fusion"])
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--p1_score_path", default=P1_SCORE_DEFAULT)
    p.add_argument("--a1_score_path", default=A1_SCORE_DEFAULT)
    p.add_argument("--c16_score_path", default=C16_SCORE_DEFAULT)
    p.add_argument("--c08_score_path", default=C08_SCORE_DEFAULT)
    p.add_argument("--fraw_score_path", default=FRAW_SCORE_DEFAULT)
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


def import_status():
    status = {}
    for name in ["numpy", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def load_score_bundle(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing score file: {npz_path}")
    arr = np.load(npz_path, allow_pickle=True)
    if "test_scores" in arr.files:
        scores = arr["test_scores"].astype(np.float64)
    elif "scores" in arr.files:
        scores = arr["scores"].astype(np.float64)
    else:
        raise KeyError(f"{npz_path} has no test_scores/scores key; keys={arr.files}")
    if "y_test" not in arr.files:
        raise KeyError(f"{npz_path} has no y_test key; keys={arr.files}")
    if "d_test" in arr.files:
        domains = arr["d_test"].astype(str)
    elif "domains" in arr.files:
        domains = arr["domains"].astype(str)
    else:
        raise KeyError(f"{npz_path} has no d_test/domains key; keys={arr.files}")
    paths = arr["test_paths"].astype(str) if "test_paths" in arr.files else None
    return {
        "path": str(npz_path),
        "scores": scores,
        "y": arr["y_test"].astype(np.int64),
        "d": domains,
        "paths": paths,
    }


def alignment_check(candidate: str, ref: dict, bundle: dict) -> dict:
    y_match = np.array_equal(bundle["y"], ref["y"])
    d_match = np.array_equal(bundle["d"], ref["d"])
    if ref["paths"] is not None and bundle["paths"] is not None:
        p_match = np.array_equal(bundle["paths"], ref["paths"])
    elif ref["paths"] is None and bundle["paths"] is None:
        p_match = bool(y_match and d_match)
    else:
        p_match = False
    return {
        "candidate": candidate,
        "score_path": bundle["path"],
        "y_test_match": bool(y_match),
        "domains_match": bool(d_match),
        "test_paths_match": bool(p_match),
        "alignment_ok": bool(y_match and d_match and p_match),
    }


def tail_metrics(scores: np.ndarray, y_test: np.ndarray) -> dict:
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95)) if len(normals) else 0.0
    q99 = float(np.quantile(normals, 0.99)) if len(normals) else 0.0
    anomaly_median = float(np.quantile(anomalies, 0.50)) if len(anomalies) else 0.0
    anomaly_q75 = float(np.quantile(anomalies, 0.75)) if len(anomalies) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": anomaly_median,
        "anomaly_q75": anomaly_q75,
        "TPR@95n": float(np.mean(anomalies > q95)) if len(anomalies) else 0.0,
        "TPR@99n": float(np.mean(anomalies > q99)) if len(anomalies) else 0.0,
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    summary = {
        "hmean": float(met["hmean"]),
        "pAUC": float(met["pauc"]),
        "KS": float(geom["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "anomaly_median": float(tail["anomaly_median"]),
        "anomaly_q75": float(tail["anomaly_q75"]),
    }
    return summary, tail, geom


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage16C3 Clean EAT Chunk Fixed Fusion One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no scoring",
        "- no fusion",
        "- no model load",
        "- no forward",
        "- no extraction",
        "",
        "## Status",
        "",
        f"- machine: `{status['machine']}`",
        f"- required files present: `{status['required_files_present']}`",
        f"- optional raw-frame score exists: `{status['optional_raw_frame_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_fusion_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage16C3 Clean EAT Chunk Fixed Fusion One-Machine Audit",
        "",
        "## Best Fixed Fusion Candidate",
        "",
        f"- machine: `{verdict['machine']}`",
        f"- best candidate: `{verdict['best_candidate']}`",
        f"- hmean: `{verdict['best_hmean']:.6f}`",
        f"- pAUC: `{verdict['best_pAUC']:.6f}`",
        f"- TPR@99n: `{verdict['best_TPR@99n']:.6f}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- no alpha sweep",
        "- no per-machine selection",
        "- gearbox only",
        "- human review required before any promotion",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    p1_path = Path(args.p1_score_path)
    a1_path = Path(args.a1_score_path)
    c16_path = Path(args.c16_score_path)
    c08_path = Path(args.c08_score_path)
    fraw_path = Path(args.fraw_score_path)

    required_paths = {
        "P1_mean_std": p1_path,
        "A1": a1_path,
        "C16S08_top1": c16_path,
        "C08S04_top1": c08_path,
    }
    optional_exists = fraw_path.exists()

    status = {
        "mode": "check",
        "machine": args.machine,
        "seed": int(args.seed),
        "required_paths": {k: str(v) for k, v in required_paths.items()},
        "required_exists": {k: v.exists() for k, v in required_paths.items()},
        "required_files_present": bool(all(v.exists() for v in required_paths.values())),
        "optional_raw_frame_path": str(fraw_path),
        "optional_raw_frame_exists": bool(optional_exists),
        "imports": import_status(),
        "notes": "No scoring, no fusion, no model load, no forward, no extraction.",
    }

    write_json(out_root / "check_status.json", status)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_fixed_fusion(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    bundles = {
        "P1_mean_std": load_score_bundle(Path(args.p1_score_path)),
        "A1": load_score_bundle(Path(args.a1_score_path)),
        "C16S08_top1": load_score_bundle(Path(args.c16_score_path)),
        "C08S04_top1": load_score_bundle(Path(args.c08_score_path)),
    }
    optional_fraw_exists = Path(args.fraw_score_path).exists()
    if optional_fraw_exists:
        bundles["F_P1_S3_050"] = load_score_bundle(Path(args.fraw_score_path))

    ref = bundles["P1_mean_std"]
    y = ref["y"]
    d = ref["d"]
    paths = ref["paths"]

    alignment_rows = []
    for candidate, bundle in bundles.items():
        row = alignment_check(candidate, ref, bundle)
        alignment_rows.append(row)
        if not row["alignment_ok"]:
            write_csv(out_root / "alignment_summary.csv", alignment_rows)
            raise RuntimeError(f"Alignment failed for {candidate}: {row}")
    write_csv(out_root / "alignment_summary.csv", alignment_rows)

    p1_r = rank_norm(bundles["P1_mean_std"]["scores"])
    a1_r = rank_norm(bundles["A1"]["scores"])
    c16_r = rank_norm(bundles["C16S08_top1"]["scores"])
    c08_r = rank_norm(bundles["C08S04_top1"]["scores"])

    candidates = {
        "F_P1_C16S08_top1_050": 0.5 * p1_r + 0.5 * c16_r,
        "F_P1_C08S04_top1_050": 0.5 * p1_r + 0.5 * c08_r,
        "F_A1_C16S08_top1_050": 0.5 * a1_r + 0.5 * c16_r,
        "F_A1_C08S04_top1_050": 0.5 * a1_r + 0.5 * c08_r,
        "F_P1_A1_C16S08_top1_TRI": (p1_r + a1_r + c16_r) / 3.0,
        "F_P1_A1_C08S04_top1_TRI": (p1_r + a1_r + c08_r) / 3.0,
    }

    if optional_fraw_exists:
        fraw_r = rank_norm(bundles["F_P1_S3_050"]["scores"])
        candidates["F_P1_Fraw_C16S08_top1_TRI"] = (p1_r + fraw_r + c16_r) / 3.0
        candidates["F_P1_Fraw_C08S04_top1_TRI"] = (p1_r + fraw_r + c08_r) / 3.0

    summary_rows = []
    tail_rows = []
    geom_rows = []

    best_row = None
    for candidate, scores in candidates.items():
        summary, tail, geom = evaluate(scores, y, d)
        row = {"candidate": candidate, "machine": args.machine, **summary}
        summary_rows.append(row)
        tail_rows.append({"candidate": candidate, "machine": args.machine, **tail})
        geom_rows.append({"candidate": candidate, "machine": args.machine, **geom})

        score_dir = scores_root / candidate
        score_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            score_dir / f"{args.machine}_scores.npz",
            test_scores=scores.astype(np.float64),
            y_test=y.astype(np.int64),
            d_test=d.astype("U16"),
            test_paths=np.asarray(paths, dtype=object) if paths is not None else None,
            score_mode=candidate,
            machine=args.machine,
            seed=int(args.seed),
        )

        if best_row is None or row["hmean"] > best_row["hmean"]:
            best_row = row

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    promising = bool(
        best_row["hmean"] > 0.575941
        and best_row["pAUC"] >= 0.525789
        and best_row["TPR@99n"] >= 0.080000
    )
    strong_positive = bool(
        best_row["hmean"] >= 0.585000
        and best_row["pAUC"] >= 0.535000
        and best_row["TPR@99n"] >= 0.090000
    )
    negative = bool(
        best_row["hmean"] <= 0.575941
        or best_row["pAUC"] < 0.525789
        or best_row["TPR@99n"] < 0.080000
    )

    comparison_row = {
        "machine": args.machine,
        "best_candidate": best_row["candidate"],
        "best_hmean": best_row["hmean"],
        "best_pAUC": best_row["pAUC"],
        "best_TPR@99n": best_row["TPR@99n"],
        "registered_P1_hmean": REGISTERED_REFERENCES["P1_mean_std"]["hmean"],
        "registered_P1_pAUC": REGISTERED_REFERENCES["P1_mean_std"]["pAUC"],
        "registered_P1_TPR@99n": REGISTERED_REFERENCES["P1_mean_std"]["TPR@99n"],
        "registered_F_P1_A1_065_hmean": REGISTERED_REFERENCES["F_P1_A1_065"]["hmean"],
        "registered_F_P1_A1_065_pAUC": REGISTERED_REFERENCES["F_P1_A1_065"]["pAUC"],
        "registered_F_P1_A1_065_TPR@99n": REGISTERED_REFERENCES["F_P1_A1_065"]["TPR@99n"],
        "registered_F_P1_S3_050_hmean": REGISTERED_REFERENCES["F_P1_S3_050"]["hmean"],
        "registered_F_P1_S3_050_pAUC": REGISTERED_REFERENCES["F_P1_S3_050"]["pAUC"],
        "registered_F_P1_S3_050_TPR@99n": REGISTERED_REFERENCES["F_P1_S3_050"]["TPR@99n"],
        "registered_C16S08_top1_hmean": REGISTERED_REFERENCES["C16S08_top1"]["hmean"],
        "registered_C16S08_top1_pAUC": REGISTERED_REFERENCES["C16S08_top1"]["pAUC"],
        "registered_C16S08_top1_TPR@99n": REGISTERED_REFERENCES["C16S08_top1"]["TPR@99n"],
        "registered_C08S04_top1_hmean": REGISTERED_REFERENCES["C08S04_top1"]["hmean"],
        "registered_C08S04_top1_pAUC": REGISTERED_REFERENCES["C08S04_top1"]["pAUC"],
        "registered_C08S04_top1_TPR@99n": REGISTERED_REFERENCES["C08S04_top1"]["TPR@99n"],
        "optional_raw_frame_score_loaded": bool(optional_fraw_exists),
        "delta_hmean_vs_F_P1_S3_050": best_row["hmean"] - REGISTERED_REFERENCES["F_P1_S3_050"]["hmean"],
        "delta_pAUC_vs_F_P1_S3_050": best_row["pAUC"] - REGISTERED_REFERENCES["F_P1_S3_050"]["pAUC"],
        "delta_TPR99_vs_F_P1_S3_050": best_row["TPR@99n"] - REGISTERED_REFERENCES["F_P1_S3_050"]["TPR@99n"],
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
    }
    write_csv(out_root / "reference_comparison.csv", [comparison_row])

    verdict = {
        "machine": args.machine,
        "best_candidate": best_row["candidate"],
        "best_hmean": float(best_row["hmean"]),
        "best_pAUC": float(best_row["pAUC"]),
        "best_TPR@99n": float(best_row["TPR@99n"]),
        "optional_raw_frame_score_loaded": bool(optional_fraw_exists),
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "registered_references": REGISTERED_REFERENCES,
        "notes": [
            "Score-level only.",
            "No alpha sweep.",
            "No per-machine selection.",
            "Gearbox only.",
            "Human review required before any promotion.",
        ],
    }
    write_json(out_root / "fixed_fusion_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_fusion_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "fixed_fusion":
        run_fixed_fusion(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
