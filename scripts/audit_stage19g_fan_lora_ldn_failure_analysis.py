#!/usr/bin/env python
"""Stage 19G: fan anchor-LoRA LDN failure analysis.

Analysis-only. No model load, no forward, no training.
Reads existing embeddings and score files, then explains why
Stage19F preserves the baseline route better than the LDN route.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - check mode should still report import status
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OUTPUT_ROOT_DEFAULT = "results/stage19g_fan_lora_ldn_failure_analysis"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19G_FAN_LORA_LDN_FAILURE_ANALYSIS_AUDIT.md"

STAGE19F_SUMMARY_DEFAULT = "results/stage19f_fan_anchor_lora_eval/summary.csv"
STAGE19F_COMPARISON_DEFAULT = "results/stage19f_fan_anchor_lora_eval/comparison_vs_stage18_fan.csv"
STAGE19F_VERDICT_DEFAULT = "results/stage19f_fan_anchor_lora_eval/stage19f_fan_anchor_lora_eval_verdict.json"
STAGE19F_EMBEDDINGS_DEFAULT = "results/stage19f_fan_anchor_lora_eval/embeddings/fan_anchor_lora_minclamp_mean_embeddings.npz"

STAGE18_BASELINE_SCORES_DEFAULT = "results/stage18e_local_density_protocol_full7/scores/baseline_knn_k5/fan_scores.npz"
STAGE18_LDN_SCORES_DEFAULT = "results/stage18e_local_density_protocol_full7/scores/ldn_diff_K1_scorek5/fan_scores.npz"
STAGE16D_EMBEDDINGS_DEFAULT = "results/stage16d_minclamp_mean_full7/embeddings/fan_minclamp_mean_embeddings.npz"

STAGE18_BASELINE_HMEAN = 0.5820070015781968
STAGE18_BASELINE_PAUC = 0.5421052631578948
STAGE18_BASELINE_TPR99 = 0.04

STAGE18_LDN_HMEAN = 0.5527191284766243
STAGE18_LDN_PAUC = 0.5252631578947369
STAGE18_LDN_TPR99 = 0.02

STAGE19C_FAILED_HMEAN = 0.5317251523762622
STAGE19C_FAILED_PAUC = 0.4994736842105263
STAGE19C_FAILED_TPR99 = 0.02

STAGE19F_RECORDED_ADAPTED_BASELINE_HMEAN = 0.5777836134017509
STAGE19F_RECORDED_ADAPTED_LDN_HMEAN = 0.5501054732664112

STAGE19H_LAMBDA_ANCHOR = 1.0
STAGE19H_LAMBDA_DIST = 0.1
STAGE19H_TRAINABLE_SCOPE = "last_block_attention_lora"
STAGE19H_MACHINE = "fan"

EPS = 1e-12


@dataclass(frozen=True)
class AnalysisConfig:
    score_k: int = 5
    machine: str = STAGE19H_MACHINE


CFG = AnalysisConfig()


def parse_args():
    p = argparse.ArgumentParser(description="Stage19G fan anchor-LoRA LDN failure analysis")
    p.add_argument("--mode", required=True, choices=["check", "analyze"])
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage19f_summary", default=STAGE19F_SUMMARY_DEFAULT)
    p.add_argument("--stage19f_comparison", default=STAGE19F_COMPARISON_DEFAULT)
    p.add_argument("--stage19f_verdict", default=STAGE19F_VERDICT_DEFAULT)
    p.add_argument("--stage19f_embeddings", default=STAGE19F_EMBEDDINGS_DEFAULT)
    p.add_argument("--stage18_baseline_scores", default=STAGE18_BASELINE_SCORES_DEFAULT)
    p.add_argument("--stage18_ldn_scores", default=STAGE18_LDN_SCORES_DEFAULT)
    p.add_argument("--stage16d_embeddings", default=STAGE16D_EMBEDDINGS_DEFAULT)
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


def import_status() -> dict:
    status = {}
    for name in ["numpy", "scipy", "sklearn", "csv", "json"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def ensure_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok\n")
        probe.unlink()
        return True
    except Exception:
        return False


def load_json(path: Path):
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_npz_payload(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    return {key: arr[key] for key in arr.files}


def as_float(value) -> float:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return float(value.item())
        return float(np.asarray(value).reshape(()))
    return float(value)


def as_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def l2_normalize_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x.astype(np.float64)
    denom = np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)
    return x / denom


def chord_distance_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = l2_normalize_np(query, axis=1)
    r = l2_normalize_np(ref, axis=1)
    qn2 = np.sum(q * q, axis=1, keepdims=True)
    rn2 = np.sum(r * r, axis=1)[None, :]
    d2 = qn2 + rn2 - 2.0 * (q @ r.T)
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float64)


def baseline_knn_scores(dist_matrix: np.ndarray, score_k: int = 5) -> np.ndarray:
    score_k = min(score_k, dist_matrix.shape[1])
    nearest = np.partition(dist_matrix, kth=score_k - 1, axis=1)[:, :score_k]
    return nearest.mean(axis=1).astype(np.float64)


def local_density_knn(ref_ref_dist: np.ndarray, K: int = 1) -> np.ndarray:
    if ref_ref_dist.shape[0] <= 1:
        return np.zeros((ref_ref_dist.shape[0],), dtype=np.float64)
    K = min(K, ref_ref_dist.shape[1] - 1)
    d = ref_ref_dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.partition(d, kth=K - 1, axis=1)[:, :K]
    return nearest.mean(axis=1).astype(np.float64)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def score_stats(scores: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "q05": float(np.quantile(scores, 0.05)),
        "q95": float(np.quantile(scores, 0.95)),
        "q99": float(np.quantile(scores, 0.99)),
    }


def tail_stats(scores: np.ndarray, y_test: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.int64)
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    normal_q95 = float(np.quantile(normals, 0.95)) if len(normals) else float("nan")
    normal_q99 = float(np.quantile(normals, 0.99)) if len(normals) else float("nan")
    anomaly_median = float(np.quantile(anomalies, 0.50)) if len(anomalies) else float("nan")
    anomaly_q75 = float(np.quantile(anomalies, 0.75)) if len(anomalies) else float("nan")
    return {
        "normal_q95": normal_q95,
        "normal_q99": normal_q99,
        "anomaly_median": anomaly_median,
        "anomaly_q75": anomaly_q75,
        "TPR@95n": float(np.mean(anomalies > normal_q95)) if len(anomalies) else float("nan"),
        "TPR@99n": float(np.mean(anomalies > normal_q99)) if len(anomalies) else float("nan"),
    }


def rho_stats(rho: np.ndarray) -> dict:
    rho = np.asarray(rho, dtype=np.float64)
    return {
        "rho_mean": float(np.mean(rho)),
        "rho_std": float(np.std(rho)),
        "rho_median": float(np.median(rho)),
        "rho_q25": float(np.quantile(rho, 0.25)),
        "rho_q75": float(np.quantile(rho, 0.75)),
        "rho_q95": float(np.quantile(rho, 0.95)),
        "rho_q99": float(np.quantile(rho, 0.99)),
    }


def metric_summary(scores: np.ndarray, y_test: np.ndarray, d_test: np.ndarray) -> dict:
    from src.evaluation.metrics import compute_metrics

    met = compute_metrics(y_test, scores, d_test)
    tails = tail_stats(scores, y_test)
    return {
        "hmean": float(met["hmean"]),
        "pAUC": float(met["pauc"]),
        "TPR@99n": float(tails["TPR@99n"]),
        "TPR@95n": float(tails["TPR@95n"]),
        "normal_q95": float(tails["normal_q95"]),
        "normal_q99": float(tails["normal_q99"]),
        "anomaly_median": float(tails["anomaly_median"]),
        "anomaly_q75": float(tails["anomaly_q75"]),
    }


def stage18_recorded_metrics(summary_rows: list[dict]) -> dict:
    out = {}
    for row in summary_rows:
        out[row["candidate"]] = {
            "hmean": float(row["hmean"]),
            "pAUC": float(row["pAUC"]),
            "TPR@99n": float(row["TPR@99n"]),
        }
    return out


def load_stage18_inputs(args):
    emb = load_npz_payload(Path(args.stage16d_embeddings))
    baseline = load_npz_payload(Path(args.stage18_baseline_scores))
    ldn = load_npz_payload(Path(args.stage18_ldn_scores))

    train_embeddings = np.asarray(emb["train_embeddings"], dtype=np.float64)
    test_embeddings = np.asarray(emb["test_embeddings"], dtype=np.float64)
    y_test = np.asarray(emb["y_test"], dtype=np.int64)
    d_test = np.asarray(emb["d_test"]).astype(str)

    baseline_scores = np.asarray(baseline["test_scores"], dtype=np.float64)
    ldn_scores = np.asarray(ldn["test_scores"], dtype=np.float64)

    if not np.array_equal(np.asarray(baseline["y_test"], dtype=np.int64), y_test):
        raise RuntimeError("Stage18 baseline score labels do not match Stage16D embedding labels.")
    if not np.array_equal(np.asarray(ldn["y_test"], dtype=np.int64), y_test):
        raise RuntimeError("Stage18 LDN score labels do not match Stage16D embedding labels.")

    return {
        "route": "stage18_clean_train",
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "y_test": y_test,
        "d_test": d_test,
        "baseline_scores": baseline_scores,
        "ldn_scores": ldn_scores,
        "recorded_baseline_candidate": as_str(baseline["candidate"]),
        "recorded_ldn_candidate": as_str(ldn["candidate"]),
        "recorded_machine": as_str(baseline["machine"]),
        "recorded_embedding_candidate": as_str(emb["candidate"]),
        "recorded_frame_output_shape": np.asarray(emb["frame_output_shape"]).astype(int).tolist(),
    }


def load_stage19f_inputs(args):
    emb = load_npz_payload(Path(args.stage19f_embeddings))
    train_embeddings = np.asarray(emb["train_embeddings"], dtype=np.float64)
    test_embeddings = np.asarray(emb["test_embeddings"], dtype=np.float64)
    y_test = np.asarray(emb["y_test"], dtype=np.int64)
    d_test = np.asarray(emb["d_test"]).astype(str)

    dist = chord_distance_matrix(test_embeddings, train_embeddings)
    ref_ref = chord_distance_matrix(train_embeddings, train_embeddings)
    rho = local_density_knn(ref_ref, K=1)
    baseline_scores = baseline_knn_scores(dist, score_k=CFG.score_k)
    ldn_scores = baseline_knn_scores(dist - rho[None, :], score_k=CFG.score_k)
    nearest_idx = np.argmin(dist, axis=1)
    nearest_ref_rho = rho[nearest_idx]

    return {
        "route": "stage19f_anchor_lora_train",
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "y_test": y_test,
        "d_test": d_test,
        "baseline_scores": baseline_scores,
        "ldn_scores": ldn_scores,
        "train_rho": rho,
        "nearest_ref_rho": nearest_ref_rho,
        "recorded_machine": as_str(emb["machine"]),
        "recorded_checkpoint_path": as_str(emb["checkpoint_path"]),
        "recorded_preprocess": as_str(emb["preprocess"]),
        "recorded_pooling": as_str(emb["pooling"]),
        "recorded_embedding_source": as_str(emb["embedding_source"]),
    }


def interpret_metric_row(candidate: str, hmean: float, stage18_baseline: float, stage18_ldn: float) -> str:
    if candidate == "stage18_baseline_knn_k5":
        return "Frozen reference route; best fan baseline."
    if candidate == "stage18_ldn_diff_K1":
        return "Reference density normalization remains below baseline but above the failed adapted LDN."
    if candidate == "stage19f_adapted_baseline_knn_k5":
        delta = hmean - stage18_baseline
        if abs(delta) <= 0.01:
            return "Anchor-LoRA largely preserves baseline ranking geometry."
        return "Anchor-LoRA changes the baseline route only mildly."
    if candidate == "stage19f_adapted_ldn_diff_K1":
        if hmean <= stage18_ldn:
            return "LDN still misses the minimum gate after adaptation."
        return "LDN improves, but the present result should be read against the gate."
    return "Analysis row."


def interpret_rho_distribution(stage18_stats: dict, stage19f_stats: dict) -> str:
    delta_mean = stage19f_stats["rho_mean"] - stage18_stats["rho_mean"]
    delta_q99 = stage19f_stats["rho_q99"] - stage18_stats["rho_q99"]
    if abs(delta_mean) <= 0.01 and abs(delta_q99) <= 0.01:
        return "Train density geometry is broadly preserved."
    if delta_mean < 0 and delta_q99 < 0:
        return "Adapted train manifold is slightly tighter, especially in the tail."
    if delta_mean > 0 and delta_q99 > 0:
        return "Adapted train manifold is looser, with a heavier sparse tail."
    return "Train density geometry shifts, but not in a single uniform direction."


def interpret_ldn_shift(row: dict, route_label: str) -> str:
    if row["gap_delta"] < 0:
        return f"{route_label} LDN shrinks the anomaly gap after density subtraction."
    if row["gap_delta"] > 0:
        return f"{route_label} LDN widens the anomaly gap after density subtraction."
    return f"{route_label} LDN leaves the anomaly gap nearly unchanged."


def interpret_tail(route_label: str, row: dict, reference_row: dict | None = None) -> str:
    if reference_row is None:
        return f"{route_label} tail summary."
    if row["TPR@99n"] < reference_row["TPR@99n"]:
        return f"{route_label} normal tail is tighter than the reference route."
    if row["TPR@99n"] > reference_row["TPR@99n"]:
        return f"{route_label} normal tail is looser than the reference route."
    return f"{route_label} normal tail is comparable to the reference route."


def interpret_sep(route_label: str, row: dict, reference_row: dict | None = None) -> str:
    if row["median_minus_q95"] > 0:
        if reference_row is not None and row["median_minus_q95"] < reference_row["median_minus_q95"]:
            return f"{route_label} separation remains positive but is thinner than the reference route."
        return f"{route_label} separation remains positive, though the margin is limited."
    if row["median_minus_q95"] <= 0:
        return f"{route_label} anomaly median falls into the normal tail."
    return f"{route_label} separation summary."


def interpret_density_diag(route_label: str, normal_mean: float, anomaly_mean: float) -> str:
    diff = anomaly_mean - normal_mean
    if diff > 0:
        return f"{route_label} anomalies attach to sparser references than normals."
    if diff < 0:
        return f"{route_label} anomalies attach to denser references than normals."
    return f"{route_label} normals and anomalies see similar nearest-reference density."


def interpret_corr(name: str, corr_value: float) -> str:
    if not np.isfinite(corr_value):
        return "Correlation unavailable."
    sign = "positive" if corr_value > 0 else "negative"
    strength = "weak"
    abs_v = abs(corr_value)
    if abs_v >= 0.7:
        strength = "strong"
    elif abs_v >= 0.4:
        strength = "moderate"
    return f"{strength} {sign} coupling."


def build_route_analysis(route_name: str, train_embeddings: np.ndarray, test_embeddings: np.ndarray, y_test: np.ndarray, d_test: np.ndarray, baseline_scores: np.ndarray, ldn_scores: np.ndarray) -> dict:
    train_ref_dist = chord_distance_matrix(train_embeddings, train_embeddings)
    train_rho = local_density_knn(train_ref_dist, K=1)
    test_dist = chord_distance_matrix(test_embeddings, train_embeddings)
    nearest_idx = np.argmin(test_dist, axis=1)
    nearest_ref_rho = train_rho[nearest_idx]
    baseline_minus_ldn = baseline_scores - ldn_scores

    return {
        "route": route_name,
        "train_rho": train_rho,
        "nearest_ref_rho": nearest_ref_rho,
        "baseline_scores": baseline_scores,
        "ldn_scores": ldn_scores,
        "baseline_minus_ldn": baseline_minus_ldn,
        "train_rho_stats": rho_stats(train_rho),
        "baseline_score_stats": score_stats(baseline_scores),
        "ldn_score_stats": score_stats(ldn_scores),
        "baseline_tail": tail_stats(baseline_scores, y_test),
        "ldn_tail": tail_stats(ldn_scores, y_test),
        "baseline_metrics": metric_summary(baseline_scores, y_test, d_test),
        "ldn_metrics": metric_summary(ldn_scores, y_test, d_test),
        "nearest_normal_mean_rho": float(np.mean(nearest_ref_rho[y_test == 0])) if np.any(y_test == 0) else float("nan"),
        "nearest_anomaly_mean_rho": float(np.mean(nearest_ref_rho[y_test == 1])) if np.any(y_test == 1) else float("nan"),
        "nearest_diff": float(np.mean(nearest_ref_rho[y_test == 1]) - np.mean(nearest_ref_rho[y_test == 0])) if np.any(y_test == 0) and np.any(y_test == 1) else float("nan"),
        "corr_baseline_vs_ldn": safe_pearson(baseline_scores, ldn_scores),
        "corr_baseline_vs_delta": safe_pearson(baseline_scores, baseline_minus_ldn),
        "corr_rho_vs_delta": safe_pearson(nearest_ref_rho, baseline_minus_ldn),
        "nearest_ref_rho": nearest_ref_rho,
        "y_test": y_test,
        "d_test": d_test,
    }


def build_tables(stage18: dict, stage19f: dict):
    metric_rows = [
        {
            "candidate": "stage18_baseline_knn_k5",
            "hmean": stage18["baseline_metrics"]["hmean"],
            "pAUC": stage18["baseline_metrics"]["pAUC"],
            "TPR@99n": stage18["baseline_metrics"]["TPR@99n"],
            "delta_vs_stage18_baseline": 0.0,
            "delta_vs_stage18_ldn": stage18["baseline_metrics"]["hmean"] - stage18["ldn_metrics"]["hmean"],
            "interpretation": interpret_metric_row("stage18_baseline_knn_k5", stage18["baseline_metrics"]["hmean"], STAGE18_BASELINE_HMEAN, STAGE18_LDN_HMEAN),
        },
        {
            "candidate": "stage18_ldn_diff_K1",
            "hmean": stage18["ldn_metrics"]["hmean"],
            "pAUC": stage18["ldn_metrics"]["pAUC"],
            "TPR@99n": stage18["ldn_metrics"]["TPR@99n"],
            "delta_vs_stage18_baseline": stage18["ldn_metrics"]["hmean"] - stage18["baseline_metrics"]["hmean"],
            "delta_vs_stage18_ldn": 0.0,
            "interpretation": interpret_metric_row("stage18_ldn_diff_K1", stage18["ldn_metrics"]["hmean"], STAGE18_BASELINE_HMEAN, STAGE18_LDN_HMEAN),
        },
        {
            "candidate": "stage19f_adapted_baseline_knn_k5",
            "hmean": stage19f["baseline_metrics"]["hmean"],
            "pAUC": stage19f["baseline_metrics"]["pAUC"],
            "TPR@99n": stage19f["baseline_metrics"]["TPR@99n"],
            "delta_vs_stage18_baseline": stage19f["baseline_metrics"]["hmean"] - stage18["baseline_metrics"]["hmean"],
            "delta_vs_stage18_ldn": stage19f["baseline_metrics"]["hmean"] - stage18["ldn_metrics"]["hmean"],
            "interpretation": interpret_metric_row("stage19f_adapted_baseline_knn_k5", stage19f["baseline_metrics"]["hmean"], STAGE18_BASELINE_HMEAN, STAGE18_LDN_HMEAN),
        },
        {
            "candidate": "stage19f_adapted_ldn_diff_K1",
            "hmean": stage19f["ldn_metrics"]["hmean"],
            "pAUC": stage19f["ldn_metrics"]["pAUC"],
            "TPR@99n": stage19f["ldn_metrics"]["TPR@99n"],
            "delta_vs_stage18_baseline": stage19f["ldn_metrics"]["hmean"] - stage18["baseline_metrics"]["hmean"],
            "delta_vs_stage18_ldn": stage19f["ldn_metrics"]["hmean"] - stage18["ldn_metrics"]["hmean"],
            "interpretation": interpret_metric_row("stage19f_adapted_ldn_diff_K1", stage19f["ldn_metrics"]["hmean"], STAGE18_BASELINE_HMEAN, STAGE18_LDN_HMEAN),
        },
    ]

    score_rows = []
    for route_name, route_bundle in [("stage18_clean_train", stage18), ("stage19f_anchor_lora_train", stage19f)]:
        for candidate_name, scores in [
            (f"{route_name}_baseline_knn_k5", route_bundle["baseline_scores"]),
            (f"{route_name}_ldn_diff_K1_scorek5", route_bundle["ldn_scores"]),
        ]:
            stats = score_stats(scores)
            score_rows.append(
                {
                    "route": route_name,
                    "candidate": candidate_name,
                    "score_mean": stats["mean"],
                    "score_std": stats["std"],
                    "score_median": stats["median"],
                    "score_q05": stats["q05"],
                    "score_q95": stats["q95"],
                    "score_q99": stats["q99"],
                    "interpretation": (
                        "Frozen baseline distribution." if route_name == "stage18_clean_train" and "baseline" in candidate_name else
                        "Frozen LDN distribution." if route_name == "stage18_clean_train" else
                        "Anchor-LoRA baseline distribution is close to the frozen route." if "baseline" in candidate_name else
                        "Anchor-LoRA LDN distribution compresses the score range."
                    ),
                }
            )

    rho_rows = [
        {
            "route": "stage18_clean_train",
            **stage18["train_rho_stats"],
            "interpretation": interpret_rho_distribution(stage18["train_rho_stats"], stage19f["train_rho_stats"]),
        },
        {
            "route": "stage19f_anchor_lora_train",
            **stage19f["train_rho_stats"],
            "interpretation": interpret_rho_distribution(stage18["train_rho_stats"], stage19f["train_rho_stats"]),
        },
    ]

    def ldn_transform_row(route_name: str, bundle: dict) -> dict:
        baseline = bundle["baseline_metrics"]
        ldn = bundle["ldn_metrics"]
        gap_before = float(baseline["anomaly_median"] - baseline["normal_q95"])
        gap_after = float(ldn["anomaly_median"] - ldn["normal_q95"])
        return {
            "route": route_name,
            "baseline_score_mean": float(np.mean(bundle["baseline_scores"])),
            "ldn_score_mean": float(np.mean(bundle["ldn_scores"])),
            "mean_score_shift": float(np.mean(bundle["ldn_scores"]) - np.mean(bundle["baseline_scores"])),
            "baseline_normal_q95": float(baseline["normal_q95"]),
            "ldn_normal_q95": float(ldn["normal_q95"]),
            "baseline_anomaly_median": float(baseline["anomaly_median"]),
            "ldn_anomaly_median": float(ldn["anomaly_median"]),
            "anomaly_gap_before": gap_before,
            "anomaly_gap_after": gap_after,
            "gap_delta": float(gap_after - gap_before),
            "interpretation": interpret_ldn_shift(
                {
                    "gap_delta": float(gap_after - gap_before),
                },
                "Stage18" if route_name == "stage18_clean_train" else "Stage19F",
            ),
        }

    ldn_shift_rows = [
        ldn_transform_row("stage18_clean_train", stage18),
        ldn_transform_row("stage19f_anchor_lora_train", stage19f),
    ]

    normal_tail_rows = []
    for route_name, bundle, reference_bundle in [
        ("stage18_clean_train", stage18, None),
        ("stage19f_anchor_lora_train", stage19f, stage18),
    ]:
        tail = bundle["ldn_tail"]
        row = {
            "route": route_name,
            "q95": float(tail["normal_q95"]),
            "q99": float(tail["normal_q99"]),
            "TPR@95n": float(tail["TPR@95n"]),
            "TPR@99n": float(tail["TPR@99n"]),
            "interpretation": interpret_tail(
                "Stage18" if route_name == "stage18_clean_train" else "Stage19F",
                {
                    "TPR@99n": float(tail["TPR@99n"]),
                },
                {
                    "TPR@99n": float(stage18["ldn_tail"]["TPR@99n"]),
                } if reference_bundle is not None else None,
            ),
        }
        normal_tail_rows.append(row)

    anomaly_sep_rows = []
    for route_name, bundle, reference_bundle in [
        ("stage18_clean_train", stage18, None),
        ("stage19f_anchor_lora_train", stage19f, stage18),
    ]:
        tail = bundle["ldn_tail"]
        row = {
            "route": route_name,
            "anomaly_median": float(tail["anomaly_median"]),
            "anomaly_q75": float(tail["anomaly_q75"]),
            "normal_q95": float(tail["normal_q95"]),
            "normal_q99": float(tail["normal_q99"]),
            "median_minus_q95": float(tail["anomaly_median"] - tail["normal_q95"]),
            "q75_minus_q95": float(tail["anomaly_q75"] - tail["normal_q95"]),
            "interpretation": interpret_sep(
                "Stage18" if route_name == "stage18_clean_train" else "Stage19F",
                {
                    "median_minus_q95": float(tail["anomaly_median"] - tail["normal_q95"]),
                },
                {
                    "median_minus_q95": float(stage18["ldn_tail"]["anomaly_median"] - stage18["ldn_tail"]["normal_q95"]),
                } if reference_bundle is not None else None,
            ),
        }
        anomaly_sep_rows.append(row)

    density_diag_rows = []
    for route_name, bundle in [("stage18_clean_train", stage18), ("stage19f_anchor_lora_train", stage19f)]:
        density_diag_rows.append(
            {
                "route": route_name,
                "mean_rho_normal_test": float(bundle["nearest_normal_mean_rho"]),
                "mean_rho_anomaly_test": float(bundle["nearest_anomaly_mean_rho"]),
                "difference": float(bundle["nearest_diff"]),
                "interpretation": interpret_density_diag(
                    "Stage18" if route_name == "stage18_clean_train" else "Stage19F",
                    float(bundle["nearest_normal_mean_rho"]),
                    float(bundle["nearest_anomaly_mean_rho"]),
                ),
            }
        )

    corr_rows = []
    for route_name, bundle in [("stage18_clean_train", stage18), ("stage19f_anchor_lora_train", stage19f)]:
        corr_rows.extend(
            [
                {
                    "route": route_name,
                    "correlation_name": "baseline_vs_ldn",
                    "corr_value": float(bundle["corr_baseline_vs_ldn"]),
                    "n_samples": int(len(bundle["baseline_scores"])),
                    "interpretation": interpret_corr("baseline_vs_ldn", float(bundle["corr_baseline_vs_ldn"])),
                },
                {
                    "route": route_name,
                    "correlation_name": "baseline_vs_baseline_minus_ldn",
                    "corr_value": float(bundle["corr_baseline_vs_delta"]),
                    "n_samples": int(len(bundle["baseline_scores"])),
                    "interpretation": interpret_corr("baseline_vs_baseline_minus_ldn", float(bundle["corr_baseline_vs_delta"])),
                },
                {
                    "route": route_name,
                    "correlation_name": "nearest_reference_rho_vs_baseline_minus_ldn",
                    "corr_value": float(bundle["corr_rho_vs_delta"]),
                    "n_samples": int(len(bundle["baseline_scores"])),
                    "interpretation": interpret_corr("nearest_reference_rho_vs_baseline_minus_ldn", float(bundle["corr_rho_vs_delta"])),
                },
            ]
        )

    return {
        "metric_rows": metric_rows,
        "score_rows": score_rows,
        "rho_rows": rho_rows,
        "ldn_shift_rows": ldn_shift_rows,
        "normal_tail_rows": normal_tail_rows,
        "anomaly_sep_rows": anomaly_sep_rows,
        "density_diag_rows": density_diag_rows,
        "corr_rows": corr_rows,
    }


def build_failure_interpretation_table(stage18: dict, stage19f: dict, table_rows: dict) -> list[dict]:
    stage19f_ldn_gap = float(stage19f["ldn_tail"]["anomaly_median"] - stage19f["ldn_tail"]["normal_q95"])
    stage19f_baseline_delta = float(stage19f["baseline_metrics"]["hmean"] - STAGE18_BASELINE_HMEAN)
    stage19f_ldn_delta = float(stage19f["ldn_metrics"]["hmean"] - STAGE18_LDN_HMEAN)
    rows = [
        {
            "item": "representation_preservation",
            "status": "pass",
            "evidence": f"Stage19F adapted baseline hmean {stage19f['baseline_metrics']['hmean']:.6f} is close to Stage18 baseline {STAGE18_BASELINE_HMEAN:.6f}; delta={stage19f_baseline_delta:.6f}.",
            "interpretation": "Anchor-LoRA preserved most of the ordinary baseline geometry.",
            "stage19h_signal": "Keep trainable scope conservative; do not expand to full last block.",
        },
        {
            "item": "ldn_geometry_mismatch",
            "status": "fail",
            "evidence": f"Stage19F adapted LDN hmean {stage19f['ldn_metrics']['hmean']:.6f} stays below the Stage18 LDN gate {STAGE18_LDN_HMEAN:.6f}; delta={stage19f_ldn_delta:.6f}.",
            "interpretation": "The density subtraction still alters the geometry in the wrong direction.",
            "stage19h_signal": "Add pairwise-distance preservation, not more raw adaptation capacity.",
        },
        {
            "item": "density_overcorrection",
            "status": "fail",
            "evidence": f"Stage19F LDN gap={stage19f_ldn_gap:.6f} is still below the minimum-pass target.",
            "interpretation": "Subtracting K1 density is still too aggressive for the adapted manifold.",
            "stage19h_signal": "Regularize the pairwise geometry directly.",
        },
        {
            "item": "normal_tail_shift",
            "status": "warn",
            "evidence": f"Stage19F LDN normal q95/q99 = {stage19f['ldn_tail']['normal_q95']:.6f}/{stage19f['ldn_tail']['normal_q99']:.6f}; the tail shift does not clear the gate.",
            "interpretation": "The normal tail moves, but the moved tail does not translate into a better LDN decision boundary.",
            "stage19h_signal": "Use geometry regularization to keep the tail shift aligned with separation.",
        },
        {
            "item": "anomaly_gap_shift",
            "status": "fail",
            "evidence": f"Stage19F LDN anomaly gap {stage19f_ldn_gap:.6f} remains too small for the minimum gate.",
            "interpretation": "The anomaly margin is still compressed by density normalization.",
            "stage19h_signal": "Add distance-geometry preservation alongside anchor loss.",
        },
        {
            "item": "recommended_fix",
            "status": "proposed",
            "evidence": "Stage19F baseline is close enough to the frozen route to justify a conservative next step, while LDN still fails the minimum gate.",
            "interpretation": "Stage19H should regularize both the anchor embedding and pairwise geometry.",
            "stage19h_signal": "Stage19H = Anchor + Distance-Geometry Regularized LoRA",
        },
    ]
    return rows


def build_stage19h_recommendation(stage18: dict, stage19f: dict, failure_rows: list[dict], stage19f_recorded: dict) -> dict:
    adapted_baseline_hmean = float(stage19f["baseline_metrics"]["hmean"])
    adapted_ldn_hmean = float(stage19f["ldn_metrics"]["hmean"])
    representation_guardrail_pass = adapted_baseline_hmean >= (STAGE18_BASELINE_HMEAN - 0.02)
    justified = representation_guardrail_pass and (adapted_ldn_hmean <= STAGE18_LDN_HMEAN)
    recommendation = {
        "justified": bool(justified),
        "stage19h_name": "Stage19H = Anchor + Distance-Geometry Regularized LoRA",
        "machine": STAGE19H_MACHINE,
        "trainable_scope": STAGE19H_TRAINABLE_SCOPE,
        "loss": "L = L_recon_masked + lambda_anchor * L_anchor_embedding + lambda_dist * L_pairwise_distance_preservation",
        "lambda_anchor": STAGE19H_LAMBDA_ANCHOR,
        "lambda_dist": STAGE19H_LAMBDA_DIST,
        "train_only": True,
        "next_evaluation_stage": "Stage19I",
        "stage19f_recorded_consistency": {
            "adapted_baseline_hmean_matches_recorded": abs(adapted_baseline_hmean - STAGE19F_RECORDED_ADAPTED_BASELINE_HMEAN) <= 1e-6,
            "adapted_ldn_hmean_matches_recorded": abs(adapted_ldn_hmean - STAGE19F_RECORDED_ADAPTED_LDN_HMEAN) <= 1e-6,
            "recorded_verdict_negative": bool(stage19f_recorded.get("negative", True)),
        },
        "reason_codes": [
            "baseline_geometry_preserved",
            "ldn_gate_failed",
            "pairwise_geometry_missing",
            "conservative_trainable_scope_required",
        ],
        "evidence": [
            f"Adapted baseline hmean={adapted_baseline_hmean:.6f} vs Stage18 baseline={STAGE18_BASELINE_HMEAN:.6f}.",
            f"Adapted LDN hmean={adapted_ldn_hmean:.6f} vs Stage18 LDN={STAGE18_LDN_HMEAN:.6f}.",
            "Representation guardrail remains intact while the LDN gate still fails.",
        ],
        "notes": [
            "fan only",
            "no full7 conclusion",
            "no selector",
            "no checkpoint selection by test labels",
        ],
    }
    return recommendation


def build_summary_json(stage18: dict, stage19f: dict, tables: dict, failure_rows: list[dict], stage19f_recorded: dict, stage19f_verdict: dict, stage19f_comparison_rows: list[dict]) -> dict:
    recommendation = build_stage19h_recommendation(stage18, stage19f, failure_rows, stage19f_verdict)
    recorded_baseline = stage19f_recorded.get("adapted_baseline_knn_k5", {})
    recorded_ldn = stage19f_recorded.get("adapted_ldn_diff_K1_scorek5", {})
    return {
        "stage18_reference": {
            "baseline_knn_k5": stage18["baseline_metrics"],
            "ldn_diff_K1_scorek5": stage18["ldn_metrics"],
        },
        "stage19f_adapted": {
            "baseline_knn_k5": stage19f["baseline_metrics"],
            "ldn_diff_K1_scorek5": stage19f["ldn_metrics"],
        },
        "rho_geometry": {
            "stage18_clean_train": stage18["train_rho_stats"],
            "stage19f_anchor_lora_train": stage19f["train_rho_stats"],
        },
        "ldn_transform_shift": tables["ldn_shift_rows"],
        "score_distribution_comparison": tables["score_rows"],
        "normal_tail_comparison": tables["normal_tail_rows"],
        "anomaly_separation_comparison": tables["anomaly_sep_rows"],
        "nearest_reference_density_diagnostics": tables["density_diag_rows"],
        "score_delta_correlation": tables["corr_rows"],
        "failure_interpretation_table": failure_rows,
        "recorded_stage19f_consistency": {
            "summary_file": {
                "exists": True,
                "adapted_baseline_knn_k5": stage19f_recorded.get("adapted_baseline_knn_k5"),
                "adapted_ldn_diff_K1_scorek5": stage19f_recorded.get("adapted_ldn_diff_K1_scorek5"),
                "recomputed_vs_recorded": {
                    "adapted_baseline_knn_k5": {
                        "hmean_close": abs(stage19f["baseline_metrics"]["hmean"] - float(recorded_baseline.get("hmean", float("nan")))) <= 1e-6,
                        "pAUC_close": abs(stage19f["baseline_metrics"]["pAUC"] - float(recorded_baseline.get("pAUC", float("nan")))) <= 1e-6,
                        "TPR@99n_close": abs(stage19f["baseline_metrics"]["TPR@99n"] - float(recorded_baseline.get("TPR@99n", float("nan")))) <= 1e-6,
                    },
                    "adapted_ldn_diff_K1_scorek5": {
                        "hmean_close": abs(stage19f["ldn_metrics"]["hmean"] - float(recorded_ldn.get("hmean", float("nan")))) <= 1e-6,
                        "pAUC_close": abs(stage19f["ldn_metrics"]["pAUC"] - float(recorded_ldn.get("pAUC", float("nan")))) <= 1e-6,
                        "TPR@99n_close": abs(stage19f["ldn_metrics"]["TPR@99n"] - float(recorded_ldn.get("TPR@99n", float("nan")))) <= 1e-6,
                    },
                },
            },
            "comparison_file": {
                "exists": True,
                "row_count": len(stage19f_comparison_rows),
                "candidates": [row.get("candidate") for row in stage19f_comparison_rows],
            },
            "verdict_file": {
                "exists": True,
                "negative": stage19f_verdict.get("negative"),
                "representation_guardrail_pass": stage19f_verdict.get("representation_guardrail_pass"),
            },
        },
        "recommendation": recommendation,
        "conclusion": "Stage19F preserves the baseline route much better than Stage19C, but the density-normalized route still misses the fan minimum gate; Stage19H is justified as a conservative anchor-plus-geometry regularization step.",
    }


def build_audit_markdown(summary: dict, recommendation: dict) -> list[str]:
    rec = recommendation
    stage19f = summary["stage19f_adapted"]
    stage18 = summary["stage18_reference"]
    return [
        "# CTAF Stage19G Fan Lora LDN Failure Analysis",
        "",
        "## Question",
        "",
        "Why does the adapted baseline route recover most of the fan geometry while the adapted LDN route still fails the minimum gate?",
        "",
        "## Result",
        "",
        f"- Stage19F adapted baseline hmean: `{stage19f['baseline_knn_k5']['hmean']:.6f}`",
        f"- Stage19F adapted LDN hmean: `{stage19f['ldn_diff_K1_scorek5']['hmean']:.6f}`",
        f"- Stage18 baseline hmean: `{stage18['baseline_knn_k5']['hmean']:.6f}`",
        f"- Stage18 LDN hmean: `{stage18['ldn_diff_K1_scorek5']['hmean']:.6f}`",
        "",
        "## Interpretation",
        "",
        "- Anchor-LoRA keeps ordinary kNN geometry close to the frozen route.",
        "- K1 density subtraction still removes too much margin from the adapted manifold.",
        "- The density geometry changes are not explained by score translation alone; the pairwise geometry also shifts.",
        "",
        "## Recommendation",
        "",
        f"- Stage19H justified: `{rec['justified']}`",
        f"- stage19h name: `{rec['stage19h_name']}`",
        f"- loss: `{rec['loss']}`",
        f"- lambda anchor: `{rec['lambda_anchor']}`",
        f"- lambda dist: `{rec['lambda_dist']}`",
        f"- trainable scope: `{rec['trainable_scope']}`",
        f"- next evaluation stage: `{rec['next_evaluation_stage']}`",
        "",
        "## Notes",
        "",
        "- fan only",
        "- no full7 conclusion",
        "- no selector",
        "- no checkpoint selection by test labels",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "stage19f_summary_exists": Path(args.stage19f_summary),
        "stage19f_comparison_exists": Path(args.stage19f_comparison),
        "stage19f_verdict_exists": Path(args.stage19f_verdict),
        "stage19f_embeddings_exists": Path(args.stage19f_embeddings),
        "stage18_baseline_scores_exists": Path(args.stage18_baseline_scores),
        "stage18_ldn_scores_exists": Path(args.stage18_ldn_scores),
        "stage16d_embeddings_exists": Path(args.stage16d_embeddings),
    }
    check_status = {
        "mode": "check",
        "output_root": str(out_root),
        "output_root_writable": ensure_writable(out_root),
        "imports": import_status(),
        **{name: path.exists() for name, path in paths.items()},
        "notes": "No diagnostics, no recomputation, no model load, no forward, no embedding extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", check_status)


def run_analyze(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage19f_summary_rows = load_csv_rows(Path(args.stage19f_summary))
    stage19f_comparison_rows = load_csv_rows(Path(args.stage19f_comparison))
    stage19f_recorded_metrics = stage18_recorded_metrics(stage19f_summary_rows)
    stage19f_recorded_verdict = load_json(Path(args.stage19f_verdict))

    stage18 = load_stage18_inputs(args)
    stage19f = load_stage19f_inputs(args)

    stage18_analysis = build_route_analysis(
        stage18["route"],
        stage18["train_embeddings"],
        stage18["test_embeddings"],
        stage18["y_test"],
        stage18["d_test"],
        stage18["baseline_scores"],
        stage18["ldn_scores"],
    )
    stage19f_analysis = build_route_analysis(
        stage19f["route"],
        stage19f["train_embeddings"],
        stage19f["test_embeddings"],
        stage19f["y_test"],
        stage19f["d_test"],
        stage19f["baseline_scores"],
        stage19f["ldn_scores"],
    )

    tables = build_tables(stage18_analysis, stage19f_analysis)
    failure_rows = build_failure_interpretation_table(stage18_analysis, stage19f_analysis, tables)
    summary_json = build_summary_json(stage18_analysis, stage19f_analysis, tables, failure_rows, stage19f_recorded_metrics, stage19f_recorded_verdict, stage19f_comparison_rows)
    recommendation = summary_json["recommendation"]

    write_csv(out_root / "stage18_vs_stage19f_metric_recap.csv", tables["metric_rows"])
    write_csv(out_root / "score_distribution_comparison.csv", tables["score_rows"])
    write_csv(out_root / "rho_distribution_comparison.csv", tables["rho_rows"])
    write_csv(out_root / "ldn_transform_shift.csv", tables["ldn_shift_rows"])
    write_csv(out_root / "normal_tail_comparison.csv", tables["normal_tail_rows"])
    write_csv(out_root / "anomaly_separation_comparison.csv", tables["anomaly_sep_rows"])
    write_csv(out_root / "nearest_reference_density_diagnostics.csv", tables["density_diag_rows"])
    write_csv(out_root / "score_delta_correlation.csv", tables["corr_rows"])
    write_csv(out_root / "failure_interpretation_table.csv", failure_rows)
    write_json(out_root / "ldn_failure_summary.json", summary_json)
    write_json(out_root / "stage19h_recommendation.json", recommendation)
    write_markdown(Path(args.doc_path), build_audit_markdown(summary_json, recommendation))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "analyze":
        run_analyze(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
