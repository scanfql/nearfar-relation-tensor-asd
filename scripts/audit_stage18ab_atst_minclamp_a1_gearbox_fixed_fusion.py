#!/usr/bin/env python
"""Stage 18A-B: ATST + min_clamp / A1 gearbox fixed fusion audit.

This script is strictly score-level only.

It does NOT:
- load models
- run model forward
- extract embeddings
- train models
- fine-tune
- run LoRA / adapter
- search alpha
- do per-machine selection
- use oracle routing

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
OUTPUT_ROOT_DEFAULT = "results/stage18ab_atst_minclamp_a1_gearbox_fixed_fusion"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18AB_ATST_MINCLAMP_A1_GEARBOX_FIXED_FUSION_AUDIT.md"

ATST_SCORE_DEFAULT = "results/stage18a_atst_one_machine_knn/scores/gearbox_atst_knn_scores.npz"
MINCLAMP_SCORE_DEFAULT = "results/stage16d_minclamp_mean_full7/scores/gearbox_minclamp_mean_scores.npz"
A1_SCORE_DEFAULT = "results/stage14a_frozen_beats_v63_fusion/a1_v63_beats_fusion/seed_13711/scores/gearbox_v63_beats_rankfusion_scores.npz"

REGISTERED_REFERENCES = {
    "P1_mean_std": {"hmean": 0.561801, "pAUC": 0.493158, "TPR@99n": 0.020000},
    "F_P1_A1_065": {"hmean": 0.572219, "pAUC": 0.519474, "TPR@99n": 0.030000},
    "F_P1_S3_050": {"hmean": 0.575941, "pAUC": 0.525789, "TPR@99n": 0.080000},
    "min_clamp__mean": {"hmean": 0.577034, "pAUC": 0.504211, "TPR@99n": 0.030000},
    "ATST_knn": {"hmean": 0.565605, "pAUC": 0.533684, "TPR@99n": 0.030000},
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage18A-B ATST + min_clamp / A1 gearbox fixed fusion audit")
    p.add_argument("--mode", required=True, choices=["check", "fixed_fusion"])
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--atst_score_path", default=ATST_SCORE_DEFAULT)
    p.add_argument("--minclamp_score_path", default=MINCLAMP_SCORE_DEFAULT)
    p.add_argument("--a1_score_path", default=A1_SCORE_DEFAULT)
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
    if "d_test" in arr.files:
        domains = arr["d_test"].astype(str)
    elif "domains" in arr.files:
        domains = arr["domains"].astype(str)
    else:
        raise KeyError(f"{npz_path} has no d_test/domains key; keys={arr.files}")
    return {
        "path": str(npz_path),
        "scores": scores,
        "y": arr["y_test"].astype(np.int64),
        "d": domains,
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def alignment_row(candidate: str, ref: dict, bundle: dict) -> dict:
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
        "# CTAF Stage18A-B ATST + MinClamp / A1 Gearbox Fixed Fusion Audit",
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
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_fusion_markdown(best: dict, verdict: dict) -> list[str]:
    return [
        "# CTAF Stage18A-B ATST + MinClamp / A1 Gearbox Fixed Fusion Audit",
        "",
        "## Best Candidate",
        "",
        f"- machine: `{verdict['machine']}`",
        f"- best candidate: `{best['candidate']}`",
        f"- hmean: `{best['hmean']:.6f}`",
        f"- pAUC: `{best['pAUC']:.6f}`",
        f"- TPR@99n: `{best['TPR@99n']:.6f}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- fixed score-level only",
        "- no alpha sweep",
        "- no per-machine selector",
        "- no oracle",
        "- gearbox only",
        "- human review required",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    atst_path = Path(args.atst_score_path)
    minclamp_path = Path(args.minclamp_score_path)
    a1_path = Path(args.a1_score_path)

    status = {
        "mode": "check",
        "machine": args.machine,
        "required_paths": {
            "ATST_knn": str(atst_path),
            "min_clamp__mean": str(minclamp_path),
            "A1": str(a1_path),
        },
        "required_exists": {
            "ATST_knn": atst_path.exists(),
            "min_clamp__mean": minclamp_path.exists(),
            "A1": a1_path.exists(),
        },
        "required_files_present": bool(atst_path.exists() and minclamp_path.exists() and a1_path.exists()),
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
        "ATST_knn": load_score_bundle(Path(args.atst_score_path)),
        "min_clamp__mean": load_score_bundle(Path(args.minclamp_score_path)),
        "A1": load_score_bundle(Path(args.a1_score_path)),
    }
    ref = bundles["ATST_knn"]

    alignment_rows = []
    for cand, bundle in bundles.items():
        row = alignment_row(cand, ref, bundle)
        alignment_rows.append(row)
        if not row["alignment_ok"]:
            write_csv(out_root / "alignment_summary.csv", alignment_rows)
            raise RuntimeError(f"Alignment failed for {cand}: {row}")
    write_csv(out_root / "alignment_summary.csv", alignment_rows)

    y = ref["y"]
    d = ref["d"]
    paths = ref["paths"]

    atst_r = rank_norm(bundles["ATST_knn"]["scores"])
    minc_r = rank_norm(bundles["min_clamp__mean"]["scores"])
    a1_r = rank_norm(bundles["A1"]["scores"])

    candidates = {
        "ATST_knn": bundles["ATST_knn"]["scores"],
        "min_clamp__mean": bundles["min_clamp__mean"]["scores"],
        "A1": bundles["A1"]["scores"],
        "F_ATST_minclamp_050": 0.5 * atst_r + 0.5 * minc_r,
        "F_ATST_A1_050": 0.5 * atst_r + 0.5 * a1_r,
        "F_minclamp_A1_050": 0.5 * minc_r + 0.5 * a1_r,
        "F_ATST_minclamp_A1_TRI": (atst_r + minc_r + a1_r) / 3.0,
    }

    summary_rows = []
    tail_rows = []
    geom_rows = []
    best = None
    for cand, scores in candidates.items():
        summary, tail, geom = evaluate(scores, y, d)
        row = {"candidate": cand, "machine": args.machine, **summary}
        summary_rows.append(row)
        tail_rows.append({"candidate": cand, "machine": args.machine, **tail})
        geom_rows.append({"candidate": cand, "machine": args.machine, **geom})

        if cand.startswith("F_"):
            score_dir = scores_root / cand
            score_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                score_dir / f"{args.machine}_scores.npz",
                test_scores=scores.astype(np.float64),
                y_test=y.astype(np.int64),
                d_test=d.astype("U16"),
                test_paths=np.asarray(paths, dtype=object) if paths is not None else None,
                score_mode=cand,
                machine=args.machine,
                seed=int(args.seed),
            )
        if best is None or row["hmean"] > best["hmean"]:
            best = row

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    reference_row = {
        "machine": args.machine,
        "best_candidate": best["candidate"],
        "best_hmean": best["hmean"],
        "best_pAUC": best["pAUC"],
        "best_TPR@99n": best["TPR@99n"],
        "ATST_knn_hmean": REGISTERED_REFERENCES["ATST_knn"]["hmean"],
        "ATST_knn_pAUC": REGISTERED_REFERENCES["ATST_knn"]["pAUC"],
        "ATST_knn_TPR@99n": REGISTERED_REFERENCES["ATST_knn"]["TPR@99n"],
        "min_clamp__mean_hmean": REGISTERED_REFERENCES["min_clamp__mean"]["hmean"],
        "min_clamp__mean_pAUC": REGISTERED_REFERENCES["min_clamp__mean"]["pAUC"],
        "min_clamp__mean_TPR@99n": REGISTERED_REFERENCES["min_clamp__mean"]["TPR@99n"],
        "A1_registered_hmean": REGISTERED_REFERENCES["F_P1_A1_065"]["hmean"],
        "A1_registered_pAUC_proxy": REGISTERED_REFERENCES["F_P1_A1_065"]["pAUC"],
        "A1_registered_TPR@99n_proxy": REGISTERED_REFERENCES["F_P1_A1_065"]["TPR@99n"],
        "F_P1_S3_050_hmean": REGISTERED_REFERENCES["F_P1_S3_050"]["hmean"],
        "F_P1_S3_050_pAUC": REGISTERED_REFERENCES["F_P1_S3_050"]["pAUC"],
        "F_P1_S3_050_TPR@99n": REGISTERED_REFERENCES["F_P1_S3_050"]["TPR@99n"],
        "delta_hmean_vs_minclamp": best["hmean"] - REGISTERED_REFERENCES["min_clamp__mean"]["hmean"],
        "delta_pAUC_vs_minclamp": best["pAUC"] - REGISTERED_REFERENCES["min_clamp__mean"]["pAUC"],
        "delta_TPR99_vs_minclamp": best["TPR@99n"] - REGISTERED_REFERENCES["min_clamp__mean"]["TPR@99n"],
    }
    write_csv(out_root / "reference_comparison.csv", [reference_row])

    improved_vs_atst = best["hmean"] > REGISTERED_REFERENCES["ATST_knn"]["hmean"]
    improved_vs_minclamp = best["hmean"] > REGISTERED_REFERENCES["min_clamp__mean"]["hmean"]
    promising = bool(
        best["hmean"] > REGISTERED_REFERENCES["min_clamp__mean"]["hmean"]
        and best["pAUC"] >= REGISTERED_REFERENCES["min_clamp__mean"]["pAUC"]
    )
    strong_positive = bool(
        best["hmean"] >= REGISTERED_REFERENCES["min_clamp__mean"]["hmean"] + 0.005
        and best["pAUC"] >= REGISTERED_REFERENCES["min_clamp__mean"]["pAUC"] + 0.005
        and best["TPR@99n"] >= REGISTERED_REFERENCES["min_clamp__mean"]["TPR@99n"]
    )
    negative = bool(not improved_vs_atst and not improved_vs_minclamp)

    verdict = {
        "machine": args.machine,
        "best_candidate": best["candidate"],
        "best_hmean": float(best["hmean"]),
        "best_pAUC": float(best["pAUC"]),
        "best_TPR@99n": float(best["TPR@99n"]),
        "improved_vs_atst": bool(improved_vs_atst),
        "improved_vs_minclamp": bool(improved_vs_minclamp),
        "promising": bool(promising),
        "strong_positive": bool(strong_positive),
        "negative": bool(negative),
        "registered_references": REGISTERED_REFERENCES,
        "notes": [
            "Pure score-level gearbox fixed fusion.",
            "No alpha sweep.",
            "No per-machine selector.",
            "Human review required after gearbox.",
        ],
    }
    write_json(out_root / "fusion_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_fusion_markdown(best, verdict))


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
