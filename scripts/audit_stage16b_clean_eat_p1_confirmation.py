#!/usr/bin/env python
"""Stage 16B-6: clean EAT P1_mean_std confirmation / 3-seed fusion validation.

This script is score-level only.

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

It only:
- loads existing P0/P1 pooling scores from Stage16B-5A
- loads existing A1 / V63 / A0 scores
- checks alignment
- computes fixed rank fusions
- reports 3-seed confirmation metrics
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

BASE_CANDIDATES = [
    "P0_mean",
    "P1_mean_std",
    "A0_raw_BEATs",
    "A1",
    "V63",
]

FUSION_CANDIDATES = [
    "F_P1_mean_std_A1_050",
    "F_P1_mean_std_A1_065",
    "F_P1_mean_std_V63_050",
    "F_P1_mean_std_TRI",
]

ALL_CANDIDATES = BASE_CANDIDATES + FUSION_CANDIDATES


def parse_args():
    p = argparse.ArgumentParser(description="Stage16B clean EAT P1 confirmation audit")
    p.add_argument("--mode", required=True, choices=["check", "confirm_fusion"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--pooling_root", default="results/stage16b_clean_eat_pooling_fusion")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--stage14a_root", default="results/stage14a_frozen_beats_v63_fusion")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--seeds", default="13711,13721,13731")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    return p.parse_args()


def parse_list_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray):
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95))
    q99 = float(np.quantile(normals, 0.99))
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "TPR@95n": float(np.mean(anomalies > q95)),
        "TPR@99n": float(np.mean(anomalies > q99)),
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray):
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


def load_bundle(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing score file: {npz_path}")
    arr = np.load(npz_path, allow_pickle=True)

    if "test_scores" in arr.files:
        scores = arr["test_scores"]
    elif "scores" in arr.files:
        scores = arr["scores"]
    else:
        raise KeyError(f"{npz_path} has no test_scores/scores key; keys={arr.files}")

    if "y_test" not in arr.files:
        raise KeyError(f"{npz_path} has no y_test key; keys={arr.files}")

    if "d_test" in arr.files:
        domains = arr["d_test"]
    elif "domains" in arr.files:
        domains = arr["domains"]
    else:
        raise KeyError(f"{npz_path} has no d_test/domains key; keys={arr.files}")

    test_paths = arr["test_paths"].astype(str) if "test_paths" in arr.files else None

    return {
        "path": str(npz_path),
        "scores": scores.astype(np.float64),
        "y": arr["y_test"].astype(np.int64),
        "d": domains.astype(str),
        "paths": test_paths,
    }


def expected_test_paths(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return np.asarray([str(x.path) for x in test_items], dtype=str)


def pooling_score_path(pooling_root: Path, pooling: str, machine: str) -> Path:
    return pooling_root / "scores" / pooling / f"{machine}_scores.npz"


def v63_score_path(v63_root: Path, seed: int, machine: str) -> Path:
    return v63_root / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"


def a0_score_path(stage14a_root: Path, machine: str) -> Path:
    return stage14a_root / "a0_frozen_beats" / "scores" / f"{machine}_frozen_beats_knn_scores.npz"


def a1_score_path(stage14a_root: Path, seed: int, machine: str) -> Path:
    return (
        stage14a_root
        / "a1_v63_beats_fusion"
        / f"seed_{seed}"
        / "scores"
        / f"{machine}_v63_beats_rankfusion_scores.npz"
    )


def alignment_check(machine: str, seed: int, candidate: str, ref: dict, bundle: dict, expected_paths: np.ndarray):
    y_match = np.array_equal(bundle["y"], ref["y"])
    d_match = np.array_equal(bundle["d"], ref["d"])

    if bundle["paths"] is not None:
        p_match = np.array_equal(bundle["paths"], expected_paths)
    else:
        p_match = bool(y_match and d_match)

    alignment_ok = bool(y_match and d_match and p_match)

    return {
        "machine": machine,
        "seed": seed,
        "candidate": candidate,
        "score_path": bundle["path"],
        "y_test_match": y_match,
        "domains_match": d_match,
        "sample_order_match": p_match,
        "alignment_ok": alignment_ok,
    }


def summarize_seed(per_seed_rows):
    seed_rows = []
    for seed in sorted({int(r["seed"]) for r in per_seed_rows}):
        for cand in sorted({r["candidate"] for r in per_seed_rows}):
            rows = [r for r in per_seed_rows if int(r["seed"]) == seed and r["candidate"] == cand]
            if not rows:
                continue
            seed_rows.append({
                "seed": seed,
                "candidate": cand,
                "hmean": float(np.mean([r["hmean"] for r in rows])),
                "pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "KS": float(np.mean([r["KS"] for r in rows])),
                "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            })
    return seed_rows


def summarize_candidate(seed_rows):
    summary = []
    for cand in sorted({r["candidate"] for r in seed_rows}):
        rows = [r for r in seed_rows if r["candidate"] == cand]
        if not rows:
            continue
        h = np.asarray([r["hmean"] for r in rows], dtype=np.float64)
        p = np.asarray([r["pAUC"] for r in rows], dtype=np.float64)
        t99 = np.asarray([r["TPR@99n"] for r in rows], dtype=np.float64)
        summary.append({
            "candidate": cand,
            "mean_hmean": float(np.mean(h)),
            "std_hmean": float(np.std(h, ddof=0)),
            "mean_pAUC": float(np.mean(p)),
            "std_pAUC": float(np.std(p, ddof=0)),
            "mean_TPR@99n": float(np.mean(t99)),
            "std_TPR@99n": float(np.std(t99, ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "seeds_ge_0p6": int(np.sum(h >= 0.600)),
            "num_seeds": int(len(rows)),
        })
    return summary


def summarize_per_machine(per_seed_rows):
    out = []
    for machine in sorted({r["machine"] for r in per_seed_rows}):
        for cand in sorted({r["candidate"] for r in per_seed_rows}):
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


def summarize_tail(tail_rows):
    out = []
    for machine in sorted({r["machine"] for r in tail_rows}):
        for cand in sorted({r["candidate"] for r in tail_rows}):
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


def get_summary(summary_rows, candidate: str):
    for row in summary_rows:
        if row["candidate"] == candidate:
            return row
    raise KeyError(f"Missing candidate in summary: {candidate}")


def get_pm(per_machine_rows, machine: str, candidate: str):
    for row in per_machine_rows:
        if row["machine"] == machine and row["candidate"] == candidate:
            return row
    raise KeyError(f"Missing per-machine row: {machine}/{candidate}")


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    pooling_root = Path(args.pooling_root)
    v63_root = Path(args.v63_root)
    stage14a_root = Path(args.stage14a_root)

    machines = parse_list_csv(args.machines)
    seeds = [int(x) for x in parse_list_csv(args.seeds)]

    missing_rows = []
    check_rows = []

    for machine in machines:
        for pooling in ["P0_mean", "P1_mean_std"]:
            path = pooling_score_path(pooling_root, pooling, machine)
            check_rows.append({
                "machine": machine,
                "seed": "deterministic",
                "candidate": pooling,
                "path": str(path),
                "exists": path.exists(),
            })
            if not path.exists():
                missing_rows.append(check_rows[-1])

        a0_path = a0_score_path(stage14a_root, machine)
        check_rows.append({
            "machine": machine,
            "seed": "deterministic",
            "candidate": "A0_raw_BEATs",
            "path": str(a0_path),
            "exists": a0_path.exists(),
        })
        if not a0_path.exists():
            missing_rows.append(check_rows[-1])

        for seed in seeds:
            for cand, path in [
                ("V63", v63_score_path(v63_root, seed, machine)),
                ("A1", a1_score_path(stage14a_root, seed, machine)),
            ]:
                check_rows.append({
                    "machine": machine,
                    "seed": seed,
                    "candidate": cand,
                    "path": str(path),
                    "exists": path.exists(),
                })
                if not path.exists():
                    missing_rows.append(check_rows[-1])

    write_csv(out_root / "check_files.csv", check_rows)
    write_csv(out_root / "missing_candidates.csv", missing_rows)

    status = {
        "mode": "check",
        "data_root_exists": data_root.exists(),
        "pooling_root_exists": pooling_root.exists(),
        "v63_root_exists": v63_root.exists(),
        "stage14a_root_exists": stage14a_root.exists(),
        "machines": machines,
        "seeds": seeds,
        "num_checked_files": len(check_rows),
        "num_missing_files": len(missing_rows),
        "all_required_files_present": len(missing_rows) == 0,
        "notes": "No scoring, no fusion, no model load, no forward, no embedding extraction.",
    }
    write_json(out_root / "check_status.json", status)


def run_confirm_fusion(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    pooling_root = Path(args.pooling_root)
    v63_root = Path(args.v63_root)
    stage14a_root = Path(args.stage14a_root)

    machines = parse_list_csv(args.machines)
    seeds = [int(x) for x in parse_list_csv(args.seeds)]

    alignment_rows = []
    per_seed_rows = []
    tail_rows = []
    score_geom_rows = []

    score_out_root = out_root / "scores"

    for seed in seeds:
        for machine in machines:
            exp_paths = expected_test_paths(data_root, machine)

            p0 = load_bundle(pooling_score_path(pooling_root, "P0_mean", machine))
            p1 = load_bundle(pooling_score_path(pooling_root, "P1_mean_std", machine))
            a0 = load_bundle(a0_score_path(stage14a_root, machine))
            a1 = load_bundle(a1_score_path(stage14a_root, seed, machine))
            v63 = load_bundle(v63_score_path(v63_root, seed, machine))

            ref = p1

            bundles = {
                "P0_mean": p0,
                "P1_mean_std": p1,
                "A0_raw_BEATs": a0,
                "A1": a1,
                "V63": v63,
            }

            for cand, bundle in bundles.items():
                row = alignment_check(machine, seed, cand, ref, bundle, exp_paths)
                alignment_rows.append(row)
                if not row["alignment_ok"]:
                    raise RuntimeError(f"Alignment failed: seed={seed} machine={machine} candidate={cand}")

            y = p1["y"]
            d = p1["d"]
            paths = p1["paths"] if p1["paths"] is not None else exp_paths

            p1_r = rank_norm(p1["scores"])
            a1_r = rank_norm(a1["scores"])
            v63_r = rank_norm(v63["scores"])
            a0_r = rank_norm(a0["scores"])

            candidates = {
                "P0_mean": p0["scores"],
                "P1_mean_std": p1["scores"],
                "A0_raw_BEATs": a0["scores"],
                "A1": a1["scores"],
                "V63": v63["scores"],
                "F_P1_mean_std_A1_050": 0.5 * p1_r + 0.5 * a1_r,
                "F_P1_mean_std_A1_065": 0.65 * p1_r + 0.35 * a1_r,
                "F_P1_mean_std_V63_050": 0.5 * p1_r + 0.5 * v63_r,
                "F_P1_mean_std_TRI": (p1_r + v63_r + a0_r) / 3.0,
            }

            for cand, scores in candidates.items():
                ev = evaluate(scores, y, d)

                per_seed_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "candidate": cand,
                    "hmean": ev["hmean"],
                    "pAUC": ev["pAUC"],
                    "KS": ev["KS"],
                    "TPR@95n": ev["TPR@95n"],
                    "TPR@99n": ev["TPR@99n"],
                })

                tail_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "candidate": cand,
                    "q95": ev["q95"],
                    "q99": ev["q99"],
                    "anomaly_median": ev["anomaly_median"],
                    "anomaly_q75": ev["anomaly_q75"],
                    "TPR@95n": ev["TPR@95n"],
                    "TPR@99n": ev["TPR@99n"],
                })

                score_geom_rows.append({
                    "seed": seed,
                    "machine": machine,
                    "candidate": cand,
                    "KS": ev["KS"],
                })

                score_dir = score_out_root / cand / f"seed_{seed}"
                score_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    score_dir / f"{machine}_scores.npz",
                    test_scores=scores.astype(np.float64),
                    y_test=y.astype(np.int64),
                    d_test=d.astype("U16"),
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

    # Confirmation verdicts.
    a1_sum = get_summary(summary, "A1")
    p0_sum = get_summary(summary, "P0_mean")
    p1_sum = get_summary(summary, "P1_mean_std")
    f065_sum = get_summary(summary, "F_P1_mean_std_A1_065")

    critical_machines = ["bearing", "fan", "toycar"]
    collapse_rows = []
    severe_collapse = False

    for machine in critical_machines:
        a1_pm = get_pm(per_machine, machine, "A1")
        f_pm = get_pm(per_machine, machine, "F_P1_mean_std_A1_065")
        gap = float(f_pm["mean_hmean"] - a1_pm["mean_hmean"])
        collapse = gap < -0.030
        severe_collapse = severe_collapse or collapse
        collapse_rows.append({
            "machine": machine,
            "A1_mean_hmean": a1_pm["mean_hmean"],
            "F_P1_mean_std_A1_065_mean_hmean": f_pm["mean_hmean"],
            "hmean_gap_vs_A1": gap,
            "severe_collapse_threshold": -0.030,
            "severe_collapse": collapse,
        })

    write_csv(out_root / "critical_machine_collapse_check.csv", collapse_rows)

    f065_seed_rows = [r for r in seed_summary if r["candidate"] == "F_P1_mean_std_A1_065"]
    f065_seeds_ge_06 = int(np.sum([r["hmean"] >= 0.600 for r in f065_seed_rows]))

    f065_confirmed = bool(
        f065_sum["mean_hmean"] >= 0.600
        and f065_seeds_ge_06 >= 2
        and f065_sum["mean_pAUC"] >= a1_sum["mean_pAUC"] - 0.005
        and f065_sum["mean_TPR@99n"] >= a1_sum["mean_TPR@99n"] - 0.010
        and not severe_collapse
    )

    p1_confirmed = bool(
        p1_sum["mean_hmean"] >= 0.600
        and p1_sum["mean_pAUC"] >= p0_sum["mean_pAUC"]
        and p1_sum["mean_TPR@99n"] >= p0_sum["mean_TPR@99n"]
    )

    verdict_rows = [
        {
            "candidate": "P1_mean_std",
            "mean_hmean": p1_sum["mean_hmean"],
            "mean_pAUC": p1_sum["mean_pAUC"],
            "mean_TPR@99n": p1_sum["mean_TPR@99n"],
            "gate_hmean_ge_0p600": p1_sum["mean_hmean"] >= 0.600,
            "gate_pAUC_ge_P0": p1_sum["mean_pAUC"] >= p0_sum["mean_pAUC"],
            "gate_TPR99_ge_P0": p1_sum["mean_TPR@99n"] >= p0_sum["mean_TPR@99n"],
            "confirmed_positive": p1_confirmed,
            "verdict": "CONFIRMED_POSITIVE" if p1_confirmed else "NOT_CONFIRMED",
        },
        {
            "candidate": "F_P1_mean_std_A1_065",
            "mean_hmean": f065_sum["mean_hmean"],
            "std_hmean": f065_sum["std_hmean"],
            "mean_pAUC": f065_sum["mean_pAUC"],
            "mean_TPR@99n": f065_sum["mean_TPR@99n"],
            "A1_mean_hmean": a1_sum["mean_hmean"],
            "A1_mean_pAUC": a1_sum["mean_pAUC"],
            "A1_mean_TPR@99n": a1_sum["mean_TPR@99n"],
            "gate_mean_hmean_ge_0p600": f065_sum["mean_hmean"] >= 0.600,
            "gate_2of3_seeds_ge_0p600": f065_seeds_ge_06 >= 2,
            "num_seeds_ge_0p600": f065_seeds_ge_06,
            "gate_pAUC_ge_A1_minus_0p005": f065_sum["mean_pAUC"] >= a1_sum["mean_pAUC"] - 0.005,
            "gate_TPR99_ge_A1_minus_0p010": f065_sum["mean_TPR@99n"] >= a1_sum["mean_TPR@99n"] - 0.010,
            "gate_no_severe_bearing_fan_toycar_collapse": not severe_collapse,
            "confirmed_positive": f065_confirmed,
            "verdict": "CONFIRMED_POSITIVE" if f065_confirmed else "NOT_CONFIRMED",
        },
    ]

    write_csv(out_root / "confirmation_verdict.csv", verdict_rows)

    best = max(summary, key=lambda r: r["mean_hmean"])

    doc_lines = [
        "# CTAF Stage16B Clean EAT P1 Confirmation Audit",
        "",
        "## 1. Purpose",
        "",
        "This is a score-level confirmation audit for the clean EAT P1_mean_std route.",
        "",
        "No EAT model loading, no model forward, no embedding extraction, no training, and no alpha search were performed.",
        "",
        "## 2. Inputs",
        "",
        f"- pooling_root: `{pooling_root}`",
        f"- v63_root: `{v63_root}`",
        f"- stage14a_root: `{stage14a_root}`",
        f"- machines: `{','.join(machines)}`",
        f"- seeds: `{','.join(map(str, seeds))}`",
        "",
        "## 3. Stage16B-5A Recap",
        "",
        "- P1_mean_std previous single-run hmean: `0.604190`",
        "- F_P1_mean_std_A1_065 previous single-run hmean: `0.608999`",
        "",
        "## 4. Confirmation Summary",
        "",
        f"- best mean-hmean candidate: `{best['candidate']}`",
        f"- best mean hmean: `{best['mean_hmean']:.6f}`",
        f"- best mean pAUC: `{best['mean_pAUC']:.6f}`",
        f"- best mean TPR@99n: `{best['mean_TPR@99n']:.6f}`",
        "",
        "## 5. Key Candidate Verdicts",
        "",
        f"- P1_mean_std confirmed positive: `{p1_confirmed}`",
        f"- F_P1_mean_std_A1_065 confirmed positive: `{f065_confirmed}`",
        "",
        "## 6. Gate Notes",
        "",
        "- F_P1_mean_std_A1_065 positive gate:",
        "  - mean hmean >= 0.600",
        "  - at least 2/3 seeds hmean >= 0.600",
        "  - mean pAUC >= A1 - 0.005",
        "  - mean TPR@99n >= A1 - 0.010",
        "  - no severe bearing/fan/toycar collapse relative to A1",
        "",
        "## 7. Confirmation",
        "",
        "- no training",
        "- no fine-tuning",
        "- no LoRA",
        "- no adapter",
        "- no model forward",
        "- no embedding extraction",
        "- no new alpha sweep",
        "- no per-machine selection",
        "- no oracle route",
        "- no Stage16C opened automatically",
    ]

    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(doc_lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "confirm_fusion":
        run_confirm_fusion(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()