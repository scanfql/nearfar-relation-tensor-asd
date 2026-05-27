#!/usr/bin/env python
"""Stage 16D fixed full7 fusion audit for min_clamp__mean + A1.

This script is strictly score-level only.

It does NOT:
- load EAT
- run model forward
- extract embeddings
- train models
- fine-tune
- run LoRA / adapter
- run Conformer
- search alpha
- do per-machine selection
- use oracle routing

It only:
- loads existing score bundles
- checks alignment
- computes fixed rank-normalized fusion candidates
- reports full7 metrics and verdict
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

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]

DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
MINCLAMP_ROOT_DEFAULT = "results/stage16d_minclamp_mean_full7"
STAGE14A_ROOT_DEFAULT = "results/stage14a_frozen_beats_v63_fusion"
OUTPUT_ROOT_DEFAULT = "results/stage16d_minclamp_mean_a1_fixed_fusion_full7"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE16D_MINCLAMP_MEAN_A1_FIXED_FUSION_FULL7_AUDIT.md"

CURRENT_BEST_REGISTERED = {
    "candidate": "F_P1_mean_std_A1_065_REGISTERED",
    "mean_hmean": 0.607175,
    "std_hmean": 0.001630,
    "mean_pAUC": 0.558321,
    "mean_TPR@99n": 0.098095,
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage16D min_clamp__mean + A1 fixed fusion full7 audit")
    p.add_argument("--mode", required=True, choices=["check", "fixed_fusion"])
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--minclamp_root", default=MINCLAMP_ROOT_DEFAULT)
    p.add_argument("--stage14a_root", default=STAGE14A_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--seeds", default="13711,13721,13731")
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
    return {
        "path": str(npz_path),
        "scores": scores,
        "y": arr["y_test"].astype(np.int64),
        "d": domains,
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def expected_test_paths(data_root: Path, machine: str) -> np.ndarray:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return np.asarray([str(x.path) for x in test_items], dtype=str)


def minclamp_score_path(minclamp_root: Path, machine: str) -> Path:
    return minclamp_root / "scores" / f"{machine}_minclamp_mean_scores.npz"


def a1_score_path(stage14a_root: Path, seed: str, machine: str) -> Path:
    return (
        stage14a_root
        / "a1_v63_beats_fusion"
        / f"seed_{seed}"
        / "scores"
        / f"{machine}_v63_beats_rankfusion_scores.npz"
    )


def alignment_row(machine: str, seed: str, candidate: str, ref: dict, bundle: dict, expected_paths: np.ndarray) -> dict:
    y_match = np.array_equal(bundle["y"], ref["y"])
    d_match = np.array_equal(bundle["d"], ref["d"])
    if bundle["paths"] is not None:
        p_match = np.array_equal(bundle["paths"], expected_paths)
    else:
        p_match = bool(y_match and d_match)
    return {
        "machine": machine,
        "seed": seed,
        "candidate": candidate,
        "score_path": bundle["path"],
        "y_test_match": bool(y_match),
        "domains_match": bool(d_match),
        "sample_order_match": bool(p_match),
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


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    return {
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


def summarize_seed(per_seed_rows: list[dict]) -> list[dict]:
    out = []
    seeds = sorted({str(r["seed"]) for r in per_seed_rows})
    candidates = sorted({r["candidate"] for r in per_seed_rows})
    for seed in seeds:
        for cand in candidates:
            rows = [r for r in per_seed_rows if str(r["seed"]) == seed and r["candidate"] == cand]
            if not rows:
                continue
            out.append({
                "seed": seed,
                "candidate": cand,
                "hmean": float(np.mean([r["hmean"] for r in rows])),
                "pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "KS": float(np.mean([r["KS"] for r in rows])),
                "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            })
    return out


def summarize_candidate(seed_rows: list[dict]) -> list[dict]:
    out = []
    for cand in sorted({r["candidate"] for r in seed_rows}):
        rows = [r for r in seed_rows if r["candidate"] == cand]
        h = np.asarray([r["hmean"] for r in rows], dtype=np.float64)
        p = np.asarray([r["pAUC"] for r in rows], dtype=np.float64)
        t = np.asarray([r["TPR@99n"] for r in rows], dtype=np.float64)
        out.append({
            "candidate": cand,
            "mean_hmean": float(np.mean(h)),
            "std_hmean": float(np.std(h, ddof=0)),
            "mean_pAUC": float(np.mean(p)),
            "std_pAUC": float(np.std(p, ddof=0)),
            "mean_TPR@99n": float(np.mean(t)),
            "std_TPR@99n": float(np.std(t, ddof=0)),
        })
    return out


def summarize_per_machine(per_seed_rows: list[dict]) -> list[dict]:
    out = []
    machines = sorted({r["machine"] for r in per_seed_rows})
    candidates = sorted({r["candidate"] for r in per_seed_rows})
    for machine in machines:
        for cand in candidates:
            rows = [r for r in per_seed_rows if r["machine"] == machine and r["candidate"] == cand]
            if not rows:
                continue
            out.append({
                "machine": machine,
                "candidate": cand,
                "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
                "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "mean_KS": float(np.mean([r["KS"] for r in rows])),
                "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            })
    return out


def summarize_tail(tail_rows: list[dict]) -> list[dict]:
    out = []
    machines = sorted({r["machine"] for r in tail_rows})
    candidates = sorted({r["candidate"] for r in tail_rows})
    for machine in machines:
        for cand in candidates:
            rows = [r for r in tail_rows if r["machine"] == machine and r["candidate"] == cand]
            if not rows:
                continue
            out.append({
                "machine": machine,
                "candidate": cand,
                "mean_q95": float(np.mean([r["q95"] for r in rows])),
                "mean_q99": float(np.mean([r["q99"] for r in rows])),
                "mean_anomaly_median": float(np.mean([r["anomaly_median"] for r in rows])),
                "mean_anomaly_q75": float(np.mean([r["anomaly_q75"] for r in rows])),
                "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            })
    return out


def get_summary(summary_rows: list[dict], candidate: str) -> dict:
    return next(r for r in summary_rows if r["candidate"] == candidate)


def get_pm(per_machine_rows: list[dict], machine: str, candidate: str) -> dict:
    return next(r for r in per_machine_rows if r["machine"] == machine and r["candidate"] == candidate)


def build_check_markdown(status: dict) -> list[str]:
    return [
        "# CTAF Stage16D MinClamp Mean A1 Fixed Fusion Full7 Audit",
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
        f"- min_clamp root exists: `{status['minclamp_root_exists']}`",
        f"- stage14a root exists: `{status['stage14a_root_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
    ]


def build_fusion_markdown(best: dict, verdict: dict) -> list[str]:
    return [
        "# CTAF Stage16D MinClamp Mean A1 Fixed Fusion Full7 Audit",
        "",
        "## Best Candidate",
        "",
        f"- best candidate: `{best['candidate']}`",
        f"- mean hmean: `{best['mean_hmean']:.6f}`",
        f"- std hmean: `{best['std_hmean']:.6f}`",
        f"- mean pAUC: `{best['mean_pAUC']:.6f}`",
        f"- mean TPR@99n: `{best['mean_TPR@99n']:.6f}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- fixed fusion only",
        "- no alpha sweep",
        "- no per-machine selector",
        "- no oracle",
        "- no 7-machine promotion without human review beyond this audit",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    minclamp_root = Path(args.minclamp_root)
    stage14a_root = Path(args.stage14a_root)
    machines = parse_csv_list(args.machines)
    seeds = parse_csv_list(args.seeds)

    rows = []
    for machine in machines:
        rows.append({
            "candidate": "min_clamp__mean",
            "machine": machine,
            "seed": "shared",
            "path": str(minclamp_score_path(minclamp_root, machine)),
            "exists": minclamp_score_path(minclamp_root, machine).exists(),
        })
        for seed in seeds:
            rows.append({
                "candidate": "A1",
                "machine": machine,
                "seed": seed,
                "path": str(a1_score_path(stage14a_root, seed, machine)),
                "exists": a1_score_path(stage14a_root, seed, machine).exists(),
            })
    write_csv(out_root / "check_files.csv", rows)

    status = {
        "mode": "check",
        "minclamp_root_exists": minclamp_root.exists(),
        "stage14a_root_exists": stage14a_root.exists(),
        "data_root_exists": data_root.exists(),
        "machines": machines,
        "seeds": seeds,
    }
    write_json(out_root / "check_status.json", status)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_fixed_fusion(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    minclamp_root = Path(args.minclamp_root)
    stage14a_root = Path(args.stage14a_root)
    machines = parse_csv_list(args.machines)
    seeds = parse_csv_list(args.seeds)

    alignment_rows = []
    per_seed_rows = []
    tail_rows = []
    score_geom_rows = []

    for machine in machines:
        expected_paths = expected_test_paths(data_root, machine)
        minclamp_bundle = load_score_bundle(minclamp_score_path(minclamp_root, machine))
        ref = {
            "y": minclamp_bundle["y"],
            "d": minclamp_bundle["d"],
            "paths": minclamp_bundle["paths"] if minclamp_bundle["paths"] is not None else expected_paths,
        }

        minclamp_align = alignment_row(machine, "shared", "min_clamp__mean", ref, minclamp_bundle, expected_paths)
        alignment_rows.append(minclamp_align)
        if not minclamp_align["alignment_ok"]:
            write_csv(out_root / "alignment_summary.csv", alignment_rows)
            raise RuntimeError(f"Alignment failed for min_clamp__mean on machine={machine}")

        for seed in seeds:
            a1_bundle = load_score_bundle(a1_score_path(stage14a_root, seed, machine))
            a1_align = alignment_row(machine, seed, "A1", ref, a1_bundle, expected_paths)
            alignment_rows.append(a1_align)
            if not a1_align["alignment_ok"]:
                write_csv(out_root / "alignment_summary.csv", alignment_rows)
                raise RuntimeError(f"Alignment failed for A1 on machine={machine}, seed={seed}")

            y_test = minclamp_bundle["y"]
            domains = minclamp_bundle["d"]
            paths = ref["paths"]
            minc = minclamp_bundle["scores"]
            a1 = a1_bundle["scores"]
            minc_r = rank_norm(minc)
            a1_r = rank_norm(a1)

            candidates = {
                "min_clamp__mean": minc,
                "A1": a1,
                "F_minclampmean_A1_050": 0.5 * minc_r + 0.5 * a1_r,
                "F_minclampmean_A1_065": 0.65 * minc_r + 0.35 * a1_r,
            }

            for cand, scores in candidates.items():
                ev = evaluate(scores, y_test, domains)
                per_seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, **ev})
                tail_rows.append({"seed": seed, "machine": machine, "candidate": cand, **tail_metrics(scores, y_test)})
                score_geom_rows.append({"seed": seed, "machine": machine, "candidate": cand, "KS": ev["KS"]})

                if cand.startswith("F_"):
                    score_dir = scores_root / cand / f"seed_{seed}"
                    score_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        score_dir / f"{machine}_scores.npz",
                        test_scores=scores.astype(np.float64),
                        y_test=y_test.astype(np.int64),
                        d_test=domains.astype("U16"),
                        test_paths=np.asarray(paths, dtype=object),
                        score_mode=cand,
                        seed=seed,
                        machine=machine,
                    )

    seed_summary = summarize_seed(per_seed_rows)
    summary = summarize_candidate(seed_summary)
    per_machine = summarize_per_machine(per_seed_rows)
    tail_table = summarize_tail(tail_rows)

    write_csv(out_root / "alignment_summary.csv", alignment_rows)
    write_csv(out_root / "per_machine_seed_metrics.csv", per_seed_rows)
    write_csv(out_root / "seed_summary.csv", seed_summary)
    write_csv(out_root / "summary.csv", summary)
    write_csv(out_root / "per_machine.csv", per_machine)
    write_csv(out_root / "tail_table.csv", tail_table)
    write_csv(out_root / "score_geometry.csv", score_geom_rows)

    a1_sum = get_summary(summary, "A1")
    minc_sum = get_summary(summary, "min_clamp__mean")
    f050_sum = get_summary(summary, "F_minclampmean_A1_050")
    f065_sum = get_summary(summary, "F_minclampmean_A1_065")
    best = max([f050_sum, f065_sum], key=lambda r: r["mean_hmean"])

    collapse_rows = []
    severe_collapse = False
    for machine in ["bearing", "valve", "toytrain"]:
        a1_pm = get_pm(per_machine, machine, "A1")
        f_pm = get_pm(per_machine, machine, best["candidate"])
        gap = float(f_pm["mean_hmean"] - a1_pm["mean_hmean"])
        collapse = gap < -0.020
        severe_collapse = severe_collapse or collapse
        collapse_rows.append({
            "machine": machine,
            "A1_mean_hmean": a1_pm["mean_hmean"],
            f"{best['candidate']}_mean_hmean": f_pm["mean_hmean"],
            "hmean_gap_vs_A1": gap,
            "severe_collapse_threshold": -0.020,
            "severe_collapse": collapse,
        })
    write_csv(out_root / "collapse_check.csv", collapse_rows)

    improved_vs_a1 = bool(best["mean_hmean"] > a1_sum["mean_hmean"])
    improved_vs_minc = bool(best["mean_hmean"] > minc_sum["mean_hmean"])
    improved_vs_current_best = bool(best["mean_hmean"] > CURRENT_BEST_REGISTERED["mean_hmean"])
    promising = bool(improved_vs_current_best and not severe_collapse)
    strong_positive = bool(
        best["mean_hmean"] >= CURRENT_BEST_REGISTERED["mean_hmean"] + 0.005
        and best["mean_pAUC"] >= CURRENT_BEST_REGISTERED["mean_pAUC"] - 0.005
        and best["mean_TPR@99n"] >= CURRENT_BEST_REGISTERED["mean_TPR@99n"] - 0.010
        and not severe_collapse
    )
    negative = bool((not improved_vs_a1) or (not improved_vs_minc) or severe_collapse)

    reference_rows = [
        {
            "candidate": "A1",
            "mean_hmean": a1_sum["mean_hmean"],
            "std_hmean": a1_sum["std_hmean"],
            "mean_pAUC": a1_sum["mean_pAUC"],
            "mean_TPR@99n": a1_sum["mean_TPR@99n"],
        },
        {
            "candidate": "min_clamp__mean",
            "mean_hmean": minc_sum["mean_hmean"],
            "std_hmean": minc_sum["std_hmean"],
            "mean_pAUC": minc_sum["mean_pAUC"],
            "mean_TPR@99n": minc_sum["mean_TPR@99n"],
        },
        {
            "candidate": "F_minclampmean_A1_050",
            "mean_hmean": f050_sum["mean_hmean"],
            "std_hmean": f050_sum["std_hmean"],
            "mean_pAUC": f050_sum["mean_pAUC"],
            "mean_TPR@99n": f050_sum["mean_TPR@99n"],
        },
        {
            "candidate": "F_minclampmean_A1_065",
            "mean_hmean": f065_sum["mean_hmean"],
            "std_hmean": f065_sum["std_hmean"],
            "mean_pAUC": f065_sum["mean_pAUC"],
            "mean_TPR@99n": f065_sum["mean_TPR@99n"],
        },
        {
            "candidate": CURRENT_BEST_REGISTERED["candidate"],
            "mean_hmean": CURRENT_BEST_REGISTERED["mean_hmean"],
            "std_hmean": CURRENT_BEST_REGISTERED["std_hmean"],
            "mean_pAUC": CURRENT_BEST_REGISTERED["mean_pAUC"],
            "mean_TPR@99n": CURRENT_BEST_REGISTERED["mean_TPR@99n"],
        },
    ]
    write_csv(out_root / "reference_comparison.csv", reference_rows)

    verdict = {
        "best_candidate": best["candidate"],
        "best_mean_hmean": best["mean_hmean"],
        "best_std_hmean": best["std_hmean"],
        "best_mean_pAUC": best["mean_pAUC"],
        "best_mean_TPR@99n": best["mean_TPR@99n"],
        "a1_mean_hmean": a1_sum["mean_hmean"],
        "minclamp_mean_hmean": minc_sum["mean_hmean"],
        "current_best_registered_mean_hmean": CURRENT_BEST_REGISTERED["mean_hmean"],
        "improved_vs_a1": improved_vs_a1,
        "improved_vs_minclamp_mean": improved_vs_minc,
        "improved_vs_current_best_registered": improved_vs_current_best,
        "severe_collapse": severe_collapse,
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "notes": [
            "Fixed score-level fusion only.",
            "Only 050 and 065 are tested; no alpha sweep.",
            "Human review required before any promotion.",
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
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
