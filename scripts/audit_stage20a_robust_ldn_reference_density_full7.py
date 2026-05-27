#!/usr/bin/env python
"""Stage 20A: robust LDN reference-density audit.

Frozen full-7 backend audit over fixed reference-density candidates.
No model load, no embedding extraction, no training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - check mode should report import status
    np = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_stage19f_helpers():
    stage19f_path = ROOT / "scripts" / "audit_stage19f_fan_anchor_lora_eval.py"
    spec = importlib.util.spec_from_file_location("stage19f_helpers", stage19f_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Stage19F helpers from {stage19f_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


stage19f = load_stage19f_helpers()


MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]

DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
STAGE18G_ROOT_DEFAULT = "results/stage18g_ldn_diff_k1_explainability"
STAGE18E_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
STAGE16D_ROOT_DEFAULT = "results/stage16d_minclamp_mean_full7"
OUTPUT_ROOT_DEFAULT = "results/stage20a_robust_ldn_reference_density_full7"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE20A_ROBUST_LDN_REFERENCE_DENSITY_FULL7_AUDIT.md"

STAGE18F_BEST_HMEAN = 0.6159467288425517
STAGE18F_BEST_PAUC = 0.5603007518796993
STAGE18F_BEST_TPR99 = 0.10571428571428572

STAGE18F_VERDICT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze/stage18f_route_freeze_verdict.json"
STAGE18G_VERDICT_DEFAULT = "results/stage18g_ldn_diff_k1_explainability/stage18g_explainability_verdict.json"
STAGE18E_SUMMARY_DEFAULT = "results/stage18e_local_density_protocol_full7/summary.csv"
STAGE18E_PER_MACHINE_DEFAULT = "results/stage18e_local_density_protocol_full7/per_machine.csv"
STAGE18E_RANK_DEFAULT = "results/stage18e_local_density_protocol_full7/candidate_rank.csv"
STAGE18F_PER_MACHINE_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze/per_machine.csv"
STAGE16D_PER_MACHINE_DEFAULT = "results/stage16d_minclamp_mean_full7/per_machine.csv"
STAGE16D_EMBEDDINGS_DEFAULT = "results/stage16d_minclamp_mean_full7/embeddings"

EXPECTED_KEYS = ["train_embeddings", "test_embeddings", "y_test", "d_test", "test_paths"]
FIXED_CANDIDATES = [
    "baseline_knn_k5",
    "ldn_diff_K1_scorek5",
    "ldn_diff_K5_medianrho_scorek5",
    "ldn_diff_K10_trim20rho_scorek5",
    "ldn_diff_K10_winsor20rho_scorek5",
    "ldn_diff_core80_K5_medianrho_scorek5",
]

STRONG_MACHINES = {"bearing", "toytrain", "valve"}


write_json = stage19f.write_json
write_csv = stage19f.write_csv
write_markdown = stage19f.write_markdown
ensure_writable = stage19f.ensure_writable
import_status = stage19f.import_status
baseline_knn_scores = stage19f.baseline_knn_scores
local_density_knn = stage19f.local_density_knn
chord_distance_matrix = stage19f.chord_distance_matrix
evaluate = stage19f.evaluate
l2_normalize_np = stage19f.l2_normalize_np


def parse_args():
    p = argparse.ArgumentParser(description="Stage20A robust LDN reference-density full7 audit")
    p.add_argument("--mode", required=True, choices=["check", "score_robust_ldn"])
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--stage18g_root", default=STAGE18G_ROOT_DEFAULT)
    p.add_argument("--stage18e_root", default=STAGE18E_ROOT_DEFAULT)
    p.add_argument("--stage16d_root", default=STAGE16D_ROOT_DEFAULT)
    p.add_argument("--stage18f_verdict", default=STAGE18F_VERDICT_DEFAULT)
    p.add_argument("--stage18g_verdict", default=STAGE18G_VERDICT_DEFAULT)
    p.add_argument("--stage18e_summary", default=STAGE18E_SUMMARY_DEFAULT)
    p.add_argument("--stage18e_per_machine", default=STAGE18E_PER_MACHINE_DEFAULT)
    p.add_argument("--stage18e_candidate_rank", default=STAGE18E_RANK_DEFAULT)
    p.add_argument("--stage18f_per_machine", default=STAGE18F_PER_MACHINE_DEFAULT)
    p.add_argument("--stage16d_per_machine", default=STAGE16D_PER_MACHINE_DEFAULT)
    p.add_argument("--stage16d_embeddings_root", default=STAGE16D_EMBEDDINGS_DEFAULT)
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def npz_required_keys_ok(path: Path) -> tuple[bool, list[str], dict]:
    if not path.exists():
        return False, EXPECTED_KEYS[:], {}
    arr = np.load(path, allow_pickle=True)
    keys = list(arr.keys())
    missing = [k for k in EXPECTED_KEYS if k not in keys]
    shapes = {}
    for k in EXPECTED_KEYS:
        if k in arr:
            shapes[k] = getattr(arr[k], "shape", None)
    return len(missing) == 0, missing, shapes


def load_embedding_pack(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    required_ok, missing, _ = npz_required_keys_ok(path)
    if not required_ok:
        raise RuntimeError(f"Missing NPZ keys in {path}: {missing}")
    return {
        "train_embeddings": arr["train_embeddings"],
        "test_embeddings": arr["test_embeddings"],
        "y_test": arr["y_test"],
        "d_test": arr["d_test"],
        "test_paths": arr["test_paths"],
    }


def emb_path_for_machine(stage16d_embeddings_root: Path, machine: str) -> Path:
    return stage16d_embeddings_root / f"{machine}_minclamp_mean_embeddings.npz"


def load_reference_rows(paths: list[Path], candidate_name: str) -> dict[str, float]:
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            machine = row.get("machine")
            candidate = row.get("candidate")
            if machine and candidate == candidate_name:
                return {
                    "source_path": str(path),
                    "reference_hmean": safe_float(row.get("hmean")),
                }
    return {}


def load_machine_reference_map(args) -> dict[str, dict]:
    paths = [Path(args.stage18f_per_machine), Path(args.stage18e_per_machine), Path(args.stage16d_per_machine)]
    ref_map = {}
    for machine in MACHINES:
        row = None
        source_path = None
        for path in paths:
            if not path.exists():
                continue
            rows = read_csv_rows(path)
            for candidate_row in rows:
                if candidate_row.get("machine") == machine and candidate_row.get("candidate") == "ldn_diff_K1_scorek5":
                    row = candidate_row
                    source_path = str(path)
                    break
            if row is not None:
                break
        if row is None:
            ref_map[machine] = {
                "source_path": None,
                "reference_hmean": None,
                "candidate": "ldn_diff_K1_scorek5",
            }
        else:
            ref_map[machine] = {
                "source_path": source_path,
                "reference_hmean": safe_float(row.get("hmean")),
                "candidate": row.get("candidate"),
            }
    return ref_map


def check_embedding_files(stage16d_embeddings_root: Path) -> tuple[list[dict], bool]:
    rows = []
    all_ok = True
    for machine in MACHINES:
        path = emb_path_for_machine(stage16d_embeddings_root, machine)
        ok, missing, shapes = npz_required_keys_ok(path)
        exists = path.exists()
        row = {
            "machine": machine,
            "embedding_path": str(path),
            "exists": exists,
            "required_keys_ok": ok,
            "missing_keys": "|".join(missing),
            "train_shape": str(shapes.get("train_embeddings")),
            "test_shape": str(shapes.get("test_embeddings")),
            "y_test_shape": str(shapes.get("y_test")),
            "d_test_shape": str(shapes.get("d_test")),
            "test_paths_shape": str(shapes.get("test_paths")),
        }
        rows.append(row)
        all_ok = all_ok and exists and ok
    return rows, all_ok


def split_core80_mask(rho: np.ndarray) -> tuple[np.ndarray, float, float]:
    rho_q10 = float(np.quantile(rho, 0.10))
    rho_q90 = float(np.quantile(rho, 0.90))
    keep = (rho >= rho_q10) & (rho <= rho_q90)
    if not np.any(keep):
        keep = np.ones_like(rho, dtype=bool)
    return keep, rho_q10, rho_q90


def stats_row(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "rho_mean": None,
            "rho_std": None,
            "rho_median": None,
            "rho_q25": None,
            "rho_q75": None,
            "rho_q95": None,
            "rho_q99": None,
        }
    return {
        "rho_mean": float(np.mean(values)),
        "rho_std": float(np.std(values)),
        "rho_median": float(np.median(values)),
        "rho_q25": float(np.quantile(values, 0.25)),
        "rho_q75": float(np.quantile(values, 0.75)),
        "rho_q95": float(np.quantile(values, 0.95)),
        "rho_q99": float(np.quantile(values, 0.99)),
    }


def nearest_other_distances(ref_ref_dist: np.ndarray, k: int) -> np.ndarray:
    if ref_ref_dist.shape[0] <= 1:
        raise RuntimeError("Need at least two reference embeddings for rho estimation.")
    d = ref_ref_dist.copy()
    np.fill_diagonal(d, np.inf)
    k_eff = min(k, d.shape[1] - 1)
    if k_eff <= 0:
        raise RuntimeError("Insufficient reference count for rho estimation.")
    part = np.partition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.sort(part, axis=1)


def rho_vector_for_candidate(candidate: str, ref_ref_dist: np.ndarray, train_emb: np.ndarray) -> tuple[np.ndarray | None, dict, dict | None]:
    n_ref = ref_ref_dist.shape[0]
    if candidate == "baseline_knn_k5":
        return None, {"rho_mode": "none", "reference_count": int(n_ref), "core_reference_count": ""}, None

    if candidate == "ldn_diff_K1_scorek5":
        nearest1 = nearest_other_distances(ref_ref_dist, 1)[:, 0]
        diag = {"rho_mode": "K1", "reference_count": int(n_ref), "core_reference_count": ""}
        return nearest1, diag, None

    if candidate == "ldn_diff_K5_medianrho_scorek5":
        nearest5 = nearest_other_distances(ref_ref_dist, 5)
        rho = np.median(nearest5, axis=1)
        diag = {"rho_mode": "K5_median", "reference_count": int(n_ref), "core_reference_count": ""}
        return rho, diag, None

    if candidate == "ldn_diff_K10_trim20rho_scorek5":
        nearest10 = nearest_other_distances(ref_ref_dist, 10)
        sorted10 = np.sort(nearest10, axis=1)
        core = sorted10[:, 2:8]
        rho = np.mean(core, axis=1)
        diag = {"rho_mode": "K10_trim20", "reference_count": int(n_ref), "core_reference_count": ""}
        return rho, diag, None

    if candidate == "ldn_diff_K10_winsor20rho_scorek5":
        nearest10 = nearest_other_distances(ref_ref_dist, 10)
        q20 = np.quantile(nearest10, 0.20, axis=1)
        q80 = np.quantile(nearest10, 0.80, axis=1)
        clipped = np.clip(nearest10, q20[:, None], q80[:, None])
        rho = np.mean(clipped, axis=1)
        diag = {"rho_mode": "K10_winsor20", "reference_count": int(n_ref), "core_reference_count": ""}
        return rho, diag, None

    if candidate == "ldn_diff_core80_K5_medianrho_scorek5":
        nearest5 = nearest_other_distances(ref_ref_dist, 5)
        rho_all = np.median(nearest5, axis=1)
        keep_mask, rho_q10, rho_q90 = split_core80_mask(rho_all)
        rho_core = rho_all[keep_mask]
        core_diag = {
            "original_reference_count": int(n_ref),
            "kept_reference_count": int(rho_core.shape[0]),
            "kept_ratio": float(rho_core.shape[0] / max(n_ref, 1)),
            "rho_q10": float(rho_q10),
            "rho_q90": float(rho_q90),
            "notes": "kept references with rho between q10 and q90",
        }
        diag = {"rho_mode": "core80_K5_median", "reference_count": int(n_ref), "core_reference_count": int(rho_core.shape[0])}
        return rho_core, diag, core_diag | {"keep_mask": keep_mask}

    raise RuntimeError(f"Unsupported candidate: {candidate}")


def compute_candidate_scores(machine: str, pack: dict) -> tuple[dict, dict, dict, dict, dict]:
    train_emb = l2_normalize_np(pack["train_embeddings"].astype(np.float64), axis=1)
    test_emb = l2_normalize_np(pack["test_embeddings"].astype(np.float64), axis=1)
    y_test = pack["y_test"].astype(np.int64)
    d_test = pack["d_test"].astype(str)
    test_paths = pack["test_paths"].astype(str)

    base_dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref_dist = chord_distance_matrix(train_emb, train_emb)

    score_rows = {}
    per_machine_rows = {}
    tail_rows = {}
    geom_rows = {}
    rho_rows = {}
    rho_dist_rows = []
    subset_rows = []

    for candidate in FIXED_CANDIDATES:
        rho_vec, rho_diag, subset_diag = rho_vector_for_candidate(candidate, ref_ref_dist, train_emb)
        if candidate == "baseline_knn_k5":
            scores = baseline_knn_scores(base_dist, score_k=5)
            rho_values = np.array([], dtype=np.float64)
        elif candidate == "ldn_diff_core80_K5_medianrho_scorek5":
            assert subset_diag is not None
            keep_mask = subset_diag.pop("keep_mask")
            core_dist = chord_distance_matrix(test_emb, train_emb[keep_mask])
            scores = baseline_knn_scores(core_dist - rho_vec[None, :], score_k=5)
            rho_values = rho_vec
            subset_rows.append(
                {
                    "machine": machine,
                    "candidate": candidate,
                    "original_reference_count": int(ref_ref_dist.shape[0]),
                    "kept_reference_count": int(rho_vec.shape[0]),
                    "kept_ratio": float(rho_vec.shape[0] / max(ref_ref_dist.shape[0], 1)),
                    "rho_q10": float(subset_diag["rho_q10"]),
                    "rho_q90": float(subset_diag["rho_q90"]),
                    "notes": subset_diag["notes"],
                }
            )
        else:
            adjusted = base_dist - rho_vec[None, :]
            scores = baseline_knn_scores(adjusted, score_k=5)
            rho_values = rho_vec

        summary, tail, geom = evaluate(scores, y_test, d_test)
        summary = {k: float(v) for k, v in summary.items()}
        tail = {k: float(v) for k, v in tail.items()}
        geom_rows[candidate] = {"machine": machine, **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in geom.items()}}
        per_machine_rows[candidate] = {
            "machine": machine,
            "candidate": candidate,
            **summary,
        }
        tail_rows[candidate] = {
            "machine": machine,
            "candidate": candidate,
            "q95": tail["q95"],
            "q99": tail["q99"],
            "anomaly_median": tail["anomaly_median"],
            "anomaly_q75": tail["anomaly_q75"],
            "TPR@95n": tail["TPR@95n"],
            "TPR@99n": tail["TPR@99n"],
        }
        rho_stats = stats_row(rho_values) if rho_values.size else {
            "rho_mean": None,
            "rho_std": None,
            "rho_median": None,
            "rho_q25": None,
            "rho_q75": None,
            "rho_q95": None,
            "rho_q99": None,
        }
        rho_rows.setdefault(candidate, []).append(
            {
                "machine": machine,
                "candidate": candidate,
                **rho_diag,
                **rho_stats,
            }
        )
        rho_dist_rows.append(
            {
                "machine": machine,
                "candidate": candidate,
                "rho_mode": rho_diag["rho_mode"],
                **rho_stats,
            }
        )
        score_rows[candidate] = {
            "machine": machine,
            "scores": scores.astype(np.float64),
            "y_test": y_test,
            "d_test": d_test,
            "test_paths": test_paths,
            "train_count": int(train_emb.shape[0]),
            "test_count": int(test_emb.shape[0]),
            "ref_ref_dist": ref_ref_dist,
            "rho_vec": rho_vec,
            "rho_diag": rho_diag,
            "subset_diag": subset_diag,
        }

    return score_rows, per_machine_rows, tail_rows, geom_rows, rho_rows, rho_dist_rows, subset_rows


def load_reference_hmean_map(args) -> dict[str, float | None]:
    ref_rows = load_reference_rows(
        [
            Path(args.stage18f_per_machine),
            Path(args.stage18e_per_machine),
            Path(args.stage16d_per_machine),
        ],
        "ldn_diff_K1_scorek5",
    )
    return {}


def build_reference_map(args) -> dict[str, dict]:
    paths = [Path(args.stage18f_per_machine), Path(args.stage18e_per_machine), Path(args.stage16d_per_machine)]
    ref_map = {}
    for machine in MACHINES:
        ref_map[machine] = {
            "source_path": None,
            "reference_hmean": None,
            "candidate": "ldn_diff_K1_scorek5",
        }
        for path in paths:
            if not path.exists():
                continue
            for row in read_csv_rows(path):
                if row.get("machine") == machine and row.get("candidate") == "ldn_diff_K1_scorek5":
                    ref_map[machine] = {
                        "source_path": str(path),
                        "reference_hmean": safe_float(row.get("hmean")),
                        "candidate": row.get("candidate"),
                    }
                    break
            if ref_map[machine]["reference_hmean"] is not None:
                break
    return ref_map


def aggregate_summary(per_machine_rows: list[dict], collapse_rows: list[dict]) -> list[dict]:
    by_candidate = {}
    for row in per_machine_rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    collapse_by_candidate = {}
    for row in collapse_rows:
        collapse_by_candidate.setdefault(row["candidate"], []).append(row)

    summary_rows = []
    for candidate, rows in by_candidate.items():
        summary_rows.append(
            {
                "candidate": candidate,
                "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
                "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "mean_KS": float(np.mean([r["KS"] for r in rows])),
                "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
                "improved_vs_stage18f_best": float(np.mean([r["hmean"] for r in rows])) > STAGE18F_BEST_HMEAN,
                "passes_0p620": float(np.mean([r["hmean"] for r in rows])) >= 0.620,
                "severe_collapse_count": int(sum(1 for r in collapse_by_candidate.get(candidate, []) if r["severe_collapse_gt_0p02"])),
            }
        )
    return summary_rows


def add_promotion_fields(candidate_row: dict, collapse_rows: list[dict], ref_map: dict[str, dict]) -> tuple[bool, str]:
    candidate = candidate_row["candidate"]
    candidate_rows = [r for r in collapse_rows if r["candidate"] == candidate]
    strong_collapse = any(r["strong_machine"] and r["severe_collapse_gt_0p02"] for r in candidate_rows)
    comparable_rows = [r for r in candidate_rows if r["delta"] is not None]
    if not comparable_rows:
        return False, "missing machine reference hmean for guardrail comparison"
    worse_than_baseline_count = sum(1 for r in comparable_rows if r["delta"] < 0.0)
    improved_machine_count = sum(1 for r in comparable_rows if r["delta"] > 0.0)
    mean_hmean = candidate_row["mean_hmean"]

    if strong_collapse:
        return False, "strong-machine severe collapse"
    if mean_hmean <= STAGE18F_BEST_HMEAN:
        return False, "below current Stage18F best"
    if improved_machine_count < 2:
        return False, "improvement concentrated in one machine only"
    if worse_than_baseline_count > 3:
        return False, "worse than baseline on more than 3 machines"
    if mean_hmean >= 0.620:
        return True, "meets threshold and guardrails; above preferred 0.620"
    return True, "meets minimum and guardrails; below preferred 0.620"


def build_candidate_rank(summary_rows: list[dict], collapse_rows: list[dict], ref_map: dict[str, dict]) -> list[dict]:
    ordered = sorted(summary_rows, key=lambda r: (-r["mean_hmean"], -r["mean_pAUC"], r["candidate"]))
    rows = []
    for idx, row in enumerate(ordered, start=1):
        selected, reason = add_promotion_fields(row, collapse_rows, ref_map)
        rows.append(
            {
                "rank": idx,
                "candidate": row["candidate"],
                "mean_hmean": row["mean_hmean"],
                "mean_pAUC": row["mean_pAUC"],
                "mean_TPR@99n": row["mean_TPR@99n"],
                "delta_hmean_vs_stage18f_best": row["mean_hmean"] - STAGE18F_BEST_HMEAN,
                "delta_pAUC_vs_stage18f_best": row["mean_pAUC"] - STAGE18F_BEST_PAUC,
                "selected_for_promotion": selected,
                "reason": reason,
            }
        )
    return rows


def build_comparison_vs_stage18f(summary_rows: list[dict]) -> list[dict]:
    rows = [
        {
            "candidate": "stage18f_best_reference",
            "mean_hmean": STAGE18F_BEST_HMEAN,
            "mean_pAUC": STAGE18F_BEST_PAUC,
            "mean_TPR@99n": STAGE18F_BEST_TPR99,
            "delta_hmean_vs_stage18f_best": 0.0,
            "delta_pAUC_vs_stage18f_best": 0.0,
            "delta_TPR99_vs_stage18f_best": 0.0,
            "note": "reference",
        }
    ]
    for row in sorted(summary_rows, key=lambda r: r["candidate"]):
        rows.append(
            {
                "candidate": row["candidate"],
                "mean_hmean": row["mean_hmean"],
                "mean_pAUC": row["mean_pAUC"],
                "mean_TPR@99n": row["mean_TPR@99n"],
                "delta_hmean_vs_stage18f_best": row["mean_hmean"] - STAGE18F_BEST_HMEAN,
                "delta_pAUC_vs_stage18f_best": row["mean_pAUC"] - STAGE18F_BEST_PAUC,
                "delta_TPR99_vs_stage18f_best": row["mean_TPR@99n"] - STAGE18F_BEST_TPR99,
                "note": "",
            }
        )
    return rows


def aggregate_tail_rows(tail_rows: list[dict]) -> list[dict]:
    by_candidate = {}
    for row in tail_rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    out = []
    for candidate, rows in by_candidate.items():
        out.append(
            {
                "candidate": candidate,
                "mean_q95": float(np.mean([r["q95"] for r in rows])),
                "mean_q99": float(np.mean([r["q99"] for r in rows])),
                "mean_anomaly_median": float(np.mean([r["anomaly_median"] for r in rows])),
                "mean_anomaly_q75": float(np.mean([r["anomaly_q75"] for r in rows])),
                "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            }
        )
    return out


def aggregate_score_geometry_rows(geom_rows: list[dict]) -> list[dict]:
    by_candidate = {}
    for row in geom_rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    out = []
    for candidate, rows in by_candidate.items():
        ks_vals = [safe_float(r.get("source_target_ks")) for r in rows if safe_float(r.get("source_target_ks")) is not None]
        out.append(
            {
                "candidate": candidate,
                "mean_KS": float(np.mean(ks_vals)) if ks_vals else None,
                "min_KS": float(np.min(ks_vals)) if ks_vals else None,
                "max_KS": float(np.max(ks_vals)) if ks_vals else None,
            }
        )
    return out


def build_rho_distribution_by_machine(rho_dist_rows: list[dict]) -> list[dict]:
    return rho_dist_rows


def build_rho_diagnostics(rho_rows: dict[str, list[dict]]) -> list[dict]:
    out = []
    for candidate, rows in rho_rows.items():
        out.extend(rows)
    return out


def build_reference_subset_diagnostics(subset_rows: list[dict]) -> list[dict]:
    return subset_rows


def build_check_markdown(status: dict) -> list[str]:
    failed = [row["machine"] for row in status["embedding_file_check"] if not (row["exists"] and row["required_keys_ok"])]
    return [
        "# CTAF Stage20A Robust LDN Reference-Density Full7 Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no training",
        "- no embedding extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- output root writable: `{status['output_root_writable']}`",
        f"- all embedding files present and keyed correctly: `{status['all_embedding_files_ok']}`",
        f"- failed embedding machines: `{', '.join(failed) if failed else 'none'}`",
        f"- stage18f verdict exists: `{status['stage18f_verdict_exists']}`",
        f"- stage18g verdict exists: `{status['stage18g_verdict_exists']}`",
        f"- stage18e summary exists: `{status['stage18e_summary_exists']}`",
        f"- stage18e per-machine exists: `{status['stage18e_per_machine_exists']}`",
        f"- stage18e rank exists: `{status['stage18e_candidate_rank_exists']}`",
        "",
        "## Candidates",
        "",
        *[f"- `{cand}`" for cand in FIXED_CANDIDATES],
    ]


def build_audit_markdown(verdict: dict, candidate_rank_rows: list[dict], reference_map: dict[str, dict]) -> list[str]:
    best = next(row for row in candidate_rank_rows if row["candidate"] == verdict["best_candidate"])
    return [
        "# CTAF Stage20A Robust LDN Reference-Density Full7 Audit",
        "",
        "## Fixed Candidate Set",
        "",
        *[f"- `{cand}`" for cand in FIXED_CANDIDATES],
        "",
        "## Protocol",
        "",
        "- no model load",
        "- no forward",
        "- no training",
        "- no embedding extraction",
        "- no fusion",
        "- no selector",
        "- no oracle",
        "- no candidate tuning after results",
        "",
        "## Best Candidate",
        "",
        f"- candidate: `{verdict['best_candidate']}`",
        f"- mean hmean: `{verdict['best_mean_hmean']:.6f}`",
        f"- mean pAUC: `{verdict['best_mean_pAUC']:.6f}`",
        f"- recommendation: `{verdict['promotion_recommended']}`",
        f"- next step: `{verdict['recommended_next_step']}`",
        "",
        "## Stage18F Comparison",
        "",
        f"- Stage18F best hmean: `{STAGE18F_BEST_HMEAN:.6f}`",
        f"- delta vs Stage18F best: `{verdict['delta_vs_stage18f_best']:.6f}`",
        "",
        "## Fan Behavior",
        "",
        f"- improved machines vs original LDN: `{verdict['improved_machine_count_vs_original_ldn']}`",
        f"- strong-machine guardrail pass: `{verdict['strong_machine_guardrail_pass']}`",
        "",
        "## Promotion",
        "",
        f"- selected row reason: `{best['reason']}`",
        f"- selected_for_promotion: `{best['selected_for_promotion']}`",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    emb_root = Path(args.stage16d_embeddings_root)

    embedding_file_check, all_embedding_files_ok = check_embedding_files(emb_root)
    stage18f_verdict_exists = Path(args.stage18f_verdict).exists()
    stage18g_verdict_exists = Path(args.stage18g_verdict).exists()
    stage18e_summary_exists = Path(args.stage18e_summary).exists()
    stage18e_per_machine_exists = Path(args.stage18e_per_machine).exists()
    stage18e_candidate_rank_exists = Path(args.stage18e_candidate_rank).exists()
    stage18f_per_machine_exists = Path(args.stage18f_per_machine).exists()
    stage16d_per_machine_exists = Path(args.stage16d_per_machine).exists()

    status = {
        "mode": "check",
        "data_root_exists": data_root.exists(),
        "stage18f_verdict_exists": stage18f_verdict_exists,
        "stage18g_verdict_exists": stage18g_verdict_exists,
        "stage18e_summary_exists": stage18e_summary_exists,
        "stage18e_per_machine_exists": stage18e_per_machine_exists,
        "stage18e_candidate_rank_exists": stage18e_candidate_rank_exists,
        "stage18f_per_machine_exists": stage18f_per_machine_exists,
        "stage16d_per_machine_exists": stage16d_per_machine_exists,
        "output_root_writable": ensure_writable(out_root),
        "all_embedding_files_ok": all_embedding_files_ok,
        "embedding_file_check": embedding_file_check,
        "imports": import_status(),
        "notes": "No model load, no forward, no scoring, no extraction.",
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "embedding_file_check.csv", embedding_file_check)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_score_robust_ldn(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = Path(args.stage16d_embeddings_root)
    ref_map = build_reference_map(args)
    embedding_file_check, all_embedding_files_ok = check_embedding_files(emb_root)
    check_status = {
        "mode": "score_robust_ldn",
        "data_root_exists": Path(args.data_root).exists(),
        "stage18f_verdict_exists": Path(args.stage18f_verdict).exists(),
        "stage18g_verdict_exists": Path(args.stage18g_verdict).exists(),
        "stage18e_summary_exists": Path(args.stage18e_summary).exists(),
        "stage18e_per_machine_exists": Path(args.stage18e_per_machine).exists(),
        "stage18e_candidate_rank_exists": Path(args.stage18e_candidate_rank).exists(),
        "stage18f_per_machine_exists": Path(args.stage18f_per_machine).exists(),
        "stage16d_per_machine_exists": Path(args.stage16d_per_machine).exists(),
        "output_root_writable": ensure_writable(out_root),
        "all_embedding_files_ok": all_embedding_files_ok,
        "embedding_file_check": embedding_file_check,
        "imports": import_status(),
        "notes": "No model load, no forward, no training, no embedding extraction.",
    }
    write_json(out_root / "check_status.json", check_status)
    write_csv(out_root / "embedding_file_check.csv", embedding_file_check)
    if not all_embedding_files_ok:
        raise RuntimeError("One or more Stage16D embedding files are missing or missing required keys.")

    score_rows = {}
    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    rho_rows = {}
    rho_dist_rows = []
    subset_rows = []
    reference_hmean_lookup = {}

    for machine in MACHINES:
        pack = load_embedding_pack(emb_path_for_machine(emb_root, machine))
        score_pack, per_machine, tails, geoms, rho_diag_rows, rho_dist_rows_machine, subset_rows_machine = compute_machine_candidate_tables(machine, pack)
        score_rows[machine] = score_pack
        for candidate, row in per_machine.items():
            per_machine_rows.append(row)
        for candidate, row in tails.items():
            tail_rows.append(row)
        for candidate, row in geoms.items():
            geom_rows.append(row)
        for candidate, rows in rho_diag_rows.items():
            rho_rows.setdefault(candidate, []).extend(rows)
        rho_dist_rows.extend(rho_dist_rows_machine)
        subset_rows.extend(subset_rows_machine)
        reference_hmean_lookup[machine] = ref_map[machine]["reference_hmean"]

    collapse_rows = collapse_rows_from_per_machine(per_machine_rows, ref_map)
    summary_rows = aggregate_summary(per_machine_rows, collapse_rows)
    candidate_rank_rows = build_candidate_rank(summary_rows, collapse_rows, ref_map)
    comparison_rows = build_comparison_vs_stage18f(summary_rows)
    tail_summary_rows = aggregate_tail_rows(tail_rows)
    geometry_summary_rows = aggregate_score_geometry_rows(geom_rows)

    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "candidate_rank.csv", candidate_rank_rows)
    write_csv(out_root / "tail_table.csv", tail_summary_rows)
    write_csv(out_root / "score_geometry.csv", geometry_summary_rows)
    write_csv(out_root / "rho_diagnostics.csv", build_rho_diagnostics(rho_rows))
    write_csv(out_root / "rho_distribution_by_machine.csv", build_rho_distribution_by_machine(rho_dist_rows))
    write_csv(out_root / "reference_subset_diagnostics.csv", build_reference_subset_diagnostics(subset_rows))
    write_csv(out_root / "comparison_vs_stage18f.csv", comparison_rows)
    write_csv(out_root / "collapse_check.csv", collapse_rows)

    best = candidate_rank_rows[0]
    strong_collapse = any(row["strong_machine"] and row["severe_collapse_gt_0p02"] for row in collapse_rows if row["candidate"] == best["candidate"])
    improved_machine_count = sum(1 for row in collapse_rows if row["candidate"] == best["candidate"] and row["delta"] is not None and row["delta"] > 0.0)
    passes_minimum_improvement = best["mean_hmean"] > STAGE18F_BEST_HMEAN
    passes_0p620 = best["mean_hmean"] >= 0.620
    strong_machine_guardrail_pass = not strong_collapse
    if strong_collapse:
        promotion_recommended = "false"
        recommended_next_step = "stop robust-rho branch or move to light pooling audit"
    elif best["mean_hmean"] <= STAGE18F_BEST_HMEAN:
        promotion_recommended = "false"
        recommended_next_step = "stop robust-rho branch or move to light pooling audit"
    elif best["mean_hmean"] >= 0.620 and strong_machine_guardrail_pass:
        promotion_recommended = "true"
        recommended_next_step = "Stage20B route freeze / paper packaging"
    else:
        promotion_recommended = "maybe"
        recommended_next_step = "paper ablation only unless very stable"

    verdict = {
        "stage": "Stage20A",
        "best_candidate": best["candidate"],
        "best_mean_hmean": best["mean_hmean"],
        "best_mean_pAUC": best["mean_pAUC"],
        "best_mean_TPR99": best["mean_TPR@99n"],
        "current_stage18f_best_hmean": STAGE18F_BEST_HMEAN,
        "delta_vs_stage18f_best": best["mean_hmean"] - STAGE18F_BEST_HMEAN,
        "passes_minimum_improvement": passes_minimum_improvement,
        "passes_0p620": passes_0p620,
        "strong_machine_guardrail_pass": strong_machine_guardrail_pass,
        "improved_machine_count_vs_original_ldn": int(improved_machine_count),
        "negative": promotion_recommended != "true",
        "promotion_recommended": promotion_recommended,
        "recommended_next_step": recommended_next_step,
    }
    write_json(out_root / "stage20a_robust_ldn_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_markdown(verdict, candidate_rank_rows, ref_map))


def collapse_rows_from_per_machine(per_machine_rows: list[dict], ref_map: dict[str, dict]) -> list[dict]:
    rows = []
    for row in per_machine_rows:
        machine = row["machine"]
        ref_hmean = ref_map[machine]["reference_hmean"]
        candidate_hmean = row["hmean"]
        delta = None if ref_hmean is None else candidate_hmean - ref_hmean
        severe = False if delta is None else delta < -0.02
        rows.append(
            {
                "candidate": row["candidate"],
                "machine": machine,
                "reference_hmean": ref_hmean,
                "candidate_hmean": candidate_hmean,
                "delta": delta,
                "strong_machine": machine in STRONG_MACHINES,
                "severe_collapse_gt_0p02": severe,
                "pass_guardrail": not severe,
            }
        )
    return rows


def compute_machine_candidate_tables(machine: str, pack: dict):
    train_emb = l2_normalize_np(pack["train_embeddings"].astype(np.float64), axis=1)
    test_emb = l2_normalize_np(pack["test_embeddings"].astype(np.float64), axis=1)
    y_test = pack["y_test"].astype(np.int64)
    d_test = pack["d_test"].astype(str)
    test_paths = pack["test_paths"].astype(str)

    base_dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref_dist = chord_distance_matrix(train_emb, train_emb)

    per_machine_rows = {}
    tail_rows = {}
    geom_rows = {}
    rho_diag_rows = {}
    rho_dist_rows = []
    subset_rows = []
    score_rows = {}

    for candidate in FIXED_CANDIDATES:
        rho_vec, rho_diag, subset_diag = rho_vector_for_candidate(candidate, ref_ref_dist, train_emb)
        if candidate == "baseline_knn_k5":
            scores = baseline_knn_scores(base_dist, score_k=5)
            rho_values = np.array([], dtype=np.float64)
        elif candidate == "ldn_diff_core80_K5_medianrho_scorek5":
            assert subset_diag is not None
            keep_mask = subset_diag.pop("keep_mask")
            core_dist = chord_distance_matrix(test_emb, train_emb[keep_mask])
            scores = baseline_knn_scores(core_dist - rho_vec[None, :], score_k=5)
            rho_values = rho_vec
            subset_rows.append(
                {
                    "machine": machine,
                    "original_reference_count": int(ref_ref_dist.shape[0]),
                    "kept_reference_count": int(rho_vec.shape[0]),
                    "kept_ratio": float(rho_vec.shape[0] / max(ref_ref_dist.shape[0], 1)),
                    "rho_q10": float(subset_diag["rho_q10"]),
                    "rho_q90": float(subset_diag["rho_q90"]),
                    "notes": subset_diag["notes"],
                }
            )
        else:
            scores = baseline_knn_scores(base_dist - rho_vec[None, :], score_k=5)
            rho_values = rho_vec

        summary, tail, geom = evaluate(scores, y_test, d_test)
        per_machine_rows[candidate] = {"machine": machine, "candidate": candidate, **summary}
        tail_rows[candidate] = {
            "machine": machine,
            "candidate": candidate,
            "q95": float(tail["q95"]),
            "q99": float(tail["q99"]),
            "anomaly_median": float(tail["anomaly_median"]),
            "anomaly_q75": float(tail["anomaly_q75"]),
            "TPR@95n": float(tail["TPR@95n"]),
            "TPR@99n": float(tail["TPR@99n"]),
        }
        geom_rows[candidate] = {
            "machine": machine,
            "candidate": candidate,
            **{k: safe_float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in geom.items()},
        }
        rho_stats = stats_row(rho_values) if rho_values.size else {
            "rho_mean": None,
            "rho_std": None,
            "rho_median": None,
            "rho_q25": None,
            "rho_q75": None,
            "rho_q95": None,
            "rho_q99": None,
        }
        rho_diag_rows.setdefault(candidate, []).append(
            {
                "machine": machine,
                "candidate": candidate,
                **rho_diag,
                **rho_stats,
            }
        )
        rho_dist_rows.append(
            {
                "machine": machine,
                "candidate": candidate,
                "rho_mode": rho_diag["rho_mode"],
                **rho_stats,
            }
        )
        score_rows[candidate] = scores.astype(np.float64)

    return score_rows, per_machine_rows, tail_rows, geom_rows, rho_diag_rows, rho_dist_rows, subset_rows


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "score_robust_ldn":
        run_score_robust_ldn(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
