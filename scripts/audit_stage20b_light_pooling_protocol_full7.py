#!/usr/bin/env python
"""Stage 20B: light pooling protocol audit over frozen clean EAT.

Frozen full-7 audit over a fixed pooling set.
No training, no adaptation, no selector, no fusion.
Model forward is allowed only in extract mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - check mode should report import status
    np = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
POOLINGS = [
    "min_clamp__mean",
    "min_clamp__mean_std",
    "min_clamp__q90",
    "min_clamp__mean_q90",
    "min_clamp__top10norm_mean",
]
BACKENDS = ["baseline_knn_k5", "ldn_diff_K1_scorek5"]
FIXED_CANDIDATES = [f"{pooling}__{backend}" for pooling in POOLINGS for backend in BACKENDS]
STRONG_MACHINES = {"bearing", "toytrain", "valve"}

STAGE18F_BEST_HMEAN = 0.6159467288425517
STAGE18F_BEST_PAUC = 0.5603007518796993
STAGE18F_BEST_TPR99 = 0.10571428571428572

DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
EAT_MODEL_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
OUTPUT_ROOT_DEFAULT = "results/stage20b_light_pooling_protocol_full7"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE20B_LIGHT_POOLING_PROTOCOL_FULL7_AUDIT.md"

STAGE18F_VERDICT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze/stage18f_route_freeze_verdict.json"
STAGE18F_PER_MACHINE_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze/per_machine.csv"
STAGE20A_VERDICT_DEFAULT = "results/stage20a_robust_ldn_reference_density_full7/stage20a_robust_ldn_verdict.json"
STAGE20A_SUMMARY_DEFAULT = "results/stage20a_robust_ldn_reference_density_full7/summary.csv"
STAGE20A_RANK_DEFAULT = "results/stage20a_robust_ldn_reference_density_full7/candidate_rank.csv"
STAGE16D_PER_MACHINE_DEFAULT = "results/stage16d_minclamp_mean_full7/per_machine.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Stage20B light pooling protocol full7 audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_light_pooling", "score_light_pooling"])
    p.add_argument("--eat_model_path", default=EAT_MODEL_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage18f_verdict", default=STAGE18F_VERDICT_DEFAULT)
    p.add_argument("--stage18f_per_machine", default=STAGE18F_PER_MACHINE_DEFAULT)
    p.add_argument("--stage20a_verdict", default=STAGE20A_VERDICT_DEFAULT)
    p.add_argument("--stage20a_summary", default=STAGE20A_SUMMARY_DEFAULT)
    p.add_argument("--stage20a_candidate_rank", default=STAGE20A_RANK_DEFAULT)
    p.add_argument("--stage16d_per_machine", default=STAGE16D_PER_MACHINE_DEFAULT)
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
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def parse_machines(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ensure_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".codex_write_probe"
        probe.write_text("ok\n")
        probe.unlink()
        return True
    except Exception:
        return False


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


def load_stage18f_reference_map(path: Path) -> dict[str, dict]:
    ref_map = {}
    if not path.exists():
        return ref_map
    for row in read_csv_rows(path):
        if row.get("candidate") == "ldn_diff_K1_scorek5" and row.get("machine"):
            ref_map[row["machine"]] = {
                "source_path": str(path),
                "reference_hmean": safe_float(row.get("hmean")),
                "candidate": row.get("candidate"),
            }
    return ref_map


def load_stage16d_reference_map(path: Path) -> dict[str, dict]:
    ref_map = {}
    if not path.exists():
        return ref_map
    for row in read_csv_rows(path):
        if row.get("candidate") == "min_clamp__mean" and row.get("machine"):
            ref_map[row["machine"]] = {
                "source_path": str(path),
                "reference_hmean": safe_float(row.get("hmean")),
                "reference_pAUC": safe_float(row.get("pAUC")),
                "reference_TPR99": safe_float(row.get("TPR@99n")),
                "candidate": row.get("candidate"),
            }
    return ref_map


def load_dataset_splits(data_root: Path, machine: str):
    from src.data.dataset import filter_items, scan_dataset, split_train_test

    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, machine)
    train_items, test_items = split_train_test(machine_items, use_target_train=False)
    return machine_items, train_items, test_items


def load_clean_eat(model_path: Path, device: str):
    import torch

    package_parent = model_path.parent.resolve()
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from checkpoints.configuration_eat import EATConfig
    from checkpoints.modeling_eat import EATModel
    from safetensors.torch import load_file

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    try:
        model = EATModel.from_pretrained(str(model_path), trust_remote_code=False)
    except Exception:
        cfg = EATConfig.from_pretrained(str(model_path))
        model = EATModel(cfg)
        state = load_file(str(model_path / "model.safetensors"))
        model.load_state_dict(state, strict=True)

    model.eval()
    model.to(requested)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, requested


def eat_frames(model, waveform: np.ndarray, orig_sr: int, device):
    import torch
    import torchaudio

    target_sr = 16000
    wav = torch.tensor(waveform, dtype=torch.float32)
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)

    with torch.no_grad():
        mel = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=target_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        target_length = 1024
        if mel.shape[0] < target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
        else:
            mel = mel[:target_length, :]
        norm_mean = -4.268
        norm_std = 4.569
        mel = (mel - norm_mean) / (norm_std * 2)
        x = mel.unsqueeze(0).unsqueeze(0).to(device)
        output = model(x)
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            feat = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            feat = output
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            feat = output[0]
        else:
            raise RuntimeError(f"Unsupported EAT output type: {type(output)}")
        if feat.ndim != 3:
            raise RuntimeError(f"Unexpected EAT output shape: {tuple(feat.shape)}")
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32), target_sr


def min_clamp(frames: np.ndarray) -> np.ndarray:
    return np.where(np.abs(frames) < 0.1, 0.0, frames).astype(np.float32)


def pool_mean(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0, dtype=np.float64).astype(np.float64)


def pool_std(frames: np.ndarray) -> np.ndarray:
    return frames.std(axis=0, dtype=np.float64).astype(np.float64)


def pool_q90(frames: np.ndarray) -> np.ndarray:
    return np.quantile(frames, 0.90, axis=0).astype(np.float64)


def pool_mean_q90(frames: np.ndarray) -> np.ndarray:
    return np.concatenate([pool_mean(frames), pool_q90(frames)], axis=0).astype(np.float64)


def pool_top10norm_mean(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] <= 1:
        return pool_mean(frames)
    norms = np.linalg.norm(frames, axis=1)
    count = max(1, int(np.ceil(0.10 * frames.shape[0])))
    idx = np.argsort(norms)[-count:]
    selected = frames[idx]
    return selected.mean(axis=0, dtype=np.float64).astype(np.float64)


def pool_embedding(frames: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "min_clamp__mean":
        return pool_mean(frames)
    if pooling == "min_clamp__mean_std":
        return np.concatenate([pool_mean(frames), pool_std(frames)], axis=0).astype(np.float64)
    if pooling == "min_clamp__q90":
        return pool_q90(frames)
    if pooling == "min_clamp__mean_q90":
        return pool_mean_q90(frames)
    if pooling == "min_clamp__top10norm_mean":
        return pool_top10norm_mean(frames)
    raise RuntimeError(f"Unsupported pooling candidate: {pooling}")


def pooling_dim(pooling: str) -> int:
    if pooling in {"min_clamp__mean", "min_clamp__q90", "min_clamp__top10norm_mean"}:
        return 768
    if pooling in {"min_clamp__mean_std", "min_clamp__mean_q90"}:
        return 1536
    raise RuntimeError(f"Unsupported pooling: {pooling}")


def l2_normalize_np(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def chord_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d2 = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * (x @ y.T)
    )
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2 + 1e-12)


def baseline_knn_scores(dist: np.ndarray, score_k: int = 5) -> np.ndarray:
    k = max(1, min(score_k, dist.shape[1]))
    nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1).astype(np.float64)


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


def ldn_diff_k1_scores(dist: np.ndarray, ref_ref_dist: np.ndarray, score_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rho = nearest_other_distances(ref_ref_dist, 1)[:, 0]
    adjusted = dist - rho[None, :]
    scores = baseline_knn_scores(adjusted, score_k=score_k)
    return scores, rho


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
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.score_geometry import compute_score_geometry

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


def build_machine_check_rows(data_root: Path, machines: list[str]) -> list[dict]:
    rows = []
    for machine in machines:
        machine_items, train_items, test_items = load_dataset_splits(data_root, machine)
        rows.append(
            {
                "machine": machine,
                "exists_in_data_root": bool(machine_items),
                "source_train_normal_count": len(train_items),
                "test_count": len(test_items),
            }
        )
    return rows


def build_check_markdown(status: dict) -> list[str]:
    failed = [row["machine"] for row in status["machine_check"] if not row["exists_in_data_root"]]
    return [
        "# CTAF Stage20B Light Pooling Protocol Full7 Audit",
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
        f"- clean EAT checkpoint exists: `{status['eat_model_path_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- missing machines: `{', '.join(failed) if failed else 'none'}`",
        f"- stage18f verdict exists: `{status['stage18f_verdict_exists']}`",
        f"- stage20a verdict exists: `{status['stage20a_verdict_exists']}`",
        f"- stage20a summary exists: `{status['stage20a_summary_exists']}`",
        f"- stage20a rank exists: `{status['stage20a_candidate_rank_exists']}`",
        f"- stage18f per-machine exists: `{status['stage18f_per_machine_exists']}`",
        f"- stage16d per-machine exists: `{status['stage16d_per_machine_exists']}`",
        "",
        "## Pooling Candidates",
        "",
        *[f"- `{cand}`" for cand in FIXED_CANDIDATES],
    ]


def build_extract_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "# CTAF Stage20B Light Pooling Protocol Full7 Audit",
        "",
        "## Mode",
        "",
        "- `extract_light_pooling`",
        "- frozen clean EAT only",
        "- no scoring",
        "",
        "## Embedding Summary",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['machine']}` / `{row['pooling']}` dim=`{row['embedding_dim']}` train=`{row['train_count']}` test=`{row['test_count']}`"
        )
    return lines


def build_audit_markdown(verdict: dict, candidate_rank_rows: list[dict], stage16d_compare: dict) -> list[str]:
    best = next(row for row in candidate_rank_rows if row["candidate"] == verdict["best_candidate"])
    lines = [
        "# CTAF Stage20B Light Pooling Protocol Full7 Audit",
        "",
        "## Fixed Candidate Set",
        "",
    ]
    lines.extend(f"- `{cand}`" for cand in FIXED_CANDIDATES)
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- full7 audit",
            "- no selector",
            "- no fusion",
            "- no training",
            "- no test-label candidate selection",
            "",
            "## Best Candidate",
            "",
            f"- candidate: `{verdict['best_candidate']}`",
            f"- pooling: `{verdict['best_pooling']}`",
            f"- backend: `{verdict['best_backend']}`",
            f"- mean hmean: `{verdict['best_mean_hmean']:.6f}`",
            f"- mean pAUC: `{verdict['best_mean_pAUC']:.6f}`",
            f"- mean TPR@99n: `{verdict['best_mean_TPR99']:.6f}`",
            f"- promotion: `{verdict['promotion_recommended']}`",
            f"- next step: `{verdict['recommended_next_step']}`",
            "",
            "## Stage18F Comparison",
            "",
            f"- Stage18F best hmean: `{STAGE18F_BEST_HMEAN:.6f}`",
            f"- delta vs Stage18F best: `{verdict['delta_vs_stage18f_best']:.6f}`",
            f"- improved machines vs Stage18F route: `{verdict['improved_machine_count_vs_stage18f']}`",
            "",
            "## Fan Behavior",
            "",
        ]
    )
    for machine in ["fan", "gearbox", "slider", "toycar"]:
        lines.append(f"- `{machine}` delta guardrail focus recorded in `collapse_check.csv`")
    lines.extend(
        [
            "",
            "## Strong-Machine Guardrail",
            "",
            f"- pass: `{verdict['strong_machine_guardrail_pass']}`",
            f"- negative: `{verdict['negative']}`",
            "",
            "## Stage16D Reproduction",
            "",
        ]
    )
    if stage16d_compare.get("available"):
        lines.extend(
            [
                f"- available: `true`",
                f"- max abs hmean delta: `{stage16d_compare['max_abs_hmean_delta']:.6f}`",
                f"- max abs pAUC delta: `{stage16d_compare['max_abs_pauc_delta']:.6f}`",
                f"- max abs TPR@99 delta: `{stage16d_compare['max_abs_tpr99_delta']:.6f}`",
                f"- exact reproduction: `{stage16d_compare['exact_reproduction']}`",
            ]
        )
    else:
        lines.append("- available: `false`")
    lines.extend(
        [
            "",
            "## Promotion Reason",
            "",
            f"- selected row reason: `{best['reason']}`",
            f"- selected_for_promotion: `{best['selected_for_promotion']}`",
        ]
    )
    return lines


def extract_and_pool(args):
    import soundfile as sf

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    model, device = load_clean_eat(Path(args.eat_model_path), args.device)
    machines = parse_machines(args.machines)

    summary_rows = []
    diag_rows = []

    for machine in machines:
        machine_items, train_items, test_items = load_dataset_splits(Path(args.data_root), machine)
        if not train_items or not test_items:
            raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")

        train_by_pooling = {pooling: [] for pooling in POOLINGS}
        test_by_pooling = {pooling: [] for pooling in POOLINGS}
        y_test = []
        d_test = []
        test_paths = []
        sample_frame_shape = None
        sample_sr = None

        def process_item(item, target_lists: dict[str, list], is_test: bool):
            nonlocal sample_frame_shape, sample_sr
            wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
            wav = wav.mean(axis=1)
            frames, final_sr = eat_frames(model, wav, int(sr), device)
            frames = min_clamp(frames)
            if sample_frame_shape is None:
                sample_frame_shape = list(frames.shape)
            sample_sr = final_sr
            for pooling in POOLINGS:
                target_lists[pooling].append(pool_embedding(frames, pooling))
            if is_test:
                y_test.append(int(item.label))
                d_test.append(str(item.domain))
                test_paths.append(str(item.path))

        for item in train_items:
            process_item(item, train_by_pooling, False)
        for item in test_items:
            process_item(item, test_by_pooling, True)

        y_test_arr = np.asarray(y_test, dtype=np.int64)
        d_test_arr = np.asarray(d_test, dtype="U16")
        test_paths_arr = np.asarray(test_paths, dtype=object)

        for pooling in POOLINGS:
            train_emb = np.stack(train_by_pooling[pooling], axis=0).astype(np.float64)
            test_emb = np.stack(test_by_pooling[pooling], axis=0).astype(np.float64)
            pooling_dir = emb_root / pooling
            pooling_dir.mkdir(parents=True, exist_ok=True)
            emb_path = pooling_dir / f"{machine}_embeddings.npz"
            np.savez_compressed(
                emb_path,
                train_embeddings=train_emb,
                test_embeddings=test_emb,
                y_test=y_test_arr,
                d_test=d_test_arr,
                test_paths=test_paths_arr,
                machine=machine,
                pooling=pooling,
                preprocess="min_clamp",
                embedding_source="clean_eat_frozen",
                checkpoint_path=str(args.eat_model_path),
                sample_rate=sample_sr,
                frame_output_shape=sample_frame_shape,
            )
            summary_rows.append(
                {
                    "machine": machine,
                    "pooling": pooling,
                    "train_count": len(train_items),
                    "test_count": len(test_items),
                    "embedding_dim": int(train_emb.shape[1]),
                    "sample_rate": sample_sr,
                    "frame_output_shape": sample_frame_shape,
                    "embedding_path": str(emb_path),
                }
            )
            diag_rows.append(
                {
                    "machine": machine,
                    "pooling": pooling,
                    "preprocess": "min_clamp",
                    "embedding_source": "clean_eat_frozen",
                    "checkpoint_path": str(args.eat_model_path),
                    "frame_output_shape": sample_frame_shape,
                    "embedding_dim": int(train_emb.shape[1]),
                    "expected_dim": pooling_dim(pooling),
                    "min_clamp_applied": True,
                    "train_count": len(train_items),
                    "test_count": len(test_items),
                    "note": "Stage16D min_clamp__mean reproduction will be compared during scoring if reference rows are available." if pooling == "min_clamp__mean" else "",
                }
            )

    write_csv(out_root / "embedding_summary.csv", summary_rows)
    write_csv(out_root / "pooling_diagnostics.csv", diag_rows)
    write_markdown(Path(args.doc_path), build_extract_markdown(summary_rows))


def score_tail_summary(tail_rows: list[dict]) -> list[dict]:
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


def score_geometry_summary(geom_rows: list[dict]) -> list[dict]:
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


def build_reference_hmean_map(stage18f_map: dict[str, dict], candidate_rows: list[dict]) -> dict[str, float | None]:
    fallback = {}
    for row in candidate_rows:
        if row["candidate"] == "min_clamp__mean__ldn_diff_K1_scorek5":
            fallback[row["machine"]] = row["hmean"]
    ref_map = {}
    for machine in MACHINES:
        if machine in stage18f_map and stage18f_map[machine].get("reference_hmean") is not None:
            ref_map[machine] = stage18f_map[machine]["reference_hmean"]
        else:
            ref_map[machine] = fallback.get(machine)
    return ref_map


def collapse_rows_from_per_machine(per_machine_rows: list[dict], ref_map: dict[str, float | None]) -> list[dict]:
    rows = []
    for row in per_machine_rows:
        machine = row["machine"]
        ref_hmean = ref_map.get(machine)
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


def aggregate_summary(per_machine_rows: list[dict], collapse_rows: list[dict]) -> list[dict]:
    by_candidate = {}
    for row in per_machine_rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    collapse_by_candidate = {}
    for row in collapse_rows:
        collapse_by_candidate.setdefault(row["candidate"], []).append(row)

    summary_rows = []
    for candidate, rows in by_candidate.items():
        mean_hmean = float(np.mean([r["hmean"] for r in rows]))
        summary_rows.append(
            {
                "candidate": candidate,
                "pooling": rows[0]["pooling"],
                "backend": rows[0]["backend"],
                "mean_hmean": mean_hmean,
                "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
                "mean_KS": float(np.mean([r["KS"] for r in rows])),
                "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
                "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
                "delta_hmean_vs_stage18f_best": mean_hmean - STAGE18F_BEST_HMEAN,
                "improved_vs_stage18f_best": mean_hmean > STAGE18F_BEST_HMEAN,
                "passes_0p620": mean_hmean >= 0.620,
                "severe_collapse_count": int(sum(1 for r in collapse_by_candidate.get(candidate, []) if r["severe_collapse_gt_0p02"])),
            }
        )
    return summary_rows


def add_promotion_fields(candidate_row: dict, collapse_rows: list[dict]) -> tuple[bool, str]:
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


def build_candidate_rank(summary_rows: list[dict], collapse_rows: list[dict]) -> list[dict]:
    ordered = sorted(summary_rows, key=lambda r: (-r["mean_hmean"], -r["mean_pAUC"], r["candidate"]))
    rows = []
    for idx, row in enumerate(ordered, start=1):
        selected, reason = add_promotion_fields(row, collapse_rows)
        rows.append(
            {
                "rank": idx,
                "candidate": row["candidate"],
                "pooling": row["pooling"],
                "backend": row["backend"],
                "mean_hmean": row["mean_hmean"],
                "mean_pAUC": row["mean_pAUC"],
                "mean_TPR@99n": row["mean_TPR@99n"],
                "delta_hmean_vs_stage18f_best": row["delta_hmean_vs_stage18f_best"],
                "selected_for_promotion": selected,
                "reason": reason,
            }
        )
    return rows


def build_comparison_vs_stage18f(summary_rows: list[dict], stage16d_compare: dict) -> list[dict]:
    stage16d_summary = stage16d_compare.get("summary") if stage16d_compare.get("available") else None
    rows = [
        {
            "candidate": "stage18f_best_reference",
            "pooling": "",
            "backend": "",
            "mean_hmean": STAGE18F_BEST_HMEAN,
            "mean_pAUC": STAGE18F_BEST_PAUC,
            "mean_TPR@99n": STAGE18F_BEST_TPR99,
            "delta_hmean_vs_stage18f_best": 0.0,
            "delta_pAUC_vs_stage18f_best": 0.0,
            "delta_TPR99_vs_stage18f_best": 0.0,
            "stage16d_compare_available": False,
            "stage16d_reference_hmean": "",
            "stage16d_reference_pAUC": "",
            "stage16d_reference_TPR99": "",
            "delta_hmean_vs_stage16d": "",
            "delta_pAUC_vs_stage16d": "",
            "delta_TPR99_vs_stage16d": "",
            "note": "reference",
        }
    ]
    for row in sorted(summary_rows, key=lambda r: r["candidate"]):
        stage16d_available = row["candidate"] == "min_clamp__mean__baseline_knn_k5" and stage16d_compare.get("available", False)
        rows.append(
            {
                "candidate": row["candidate"],
                "pooling": row["pooling"],
                "backend": row["backend"],
                "mean_hmean": row["mean_hmean"],
                "mean_pAUC": row["mean_pAUC"],
                "mean_TPR@99n": row["mean_TPR@99n"],
                "delta_hmean_vs_stage18f_best": row["delta_hmean_vs_stage18f_best"],
                "delta_pAUC_vs_stage18f_best": row["mean_pAUC"] - STAGE18F_BEST_PAUC,
                "delta_TPR99_vs_stage18f_best": row["mean_TPR@99n"] - STAGE18F_BEST_TPR99,
                "stage16d_compare_available": stage16d_available,
                "stage16d_reference_hmean": stage16d_summary["mean_reference_hmean"] if stage16d_summary and stage16d_available else "",
                "stage16d_reference_pAUC": stage16d_summary["mean_reference_pAUC"] if stage16d_summary and stage16d_available else "",
                "stage16d_reference_TPR99": stage16d_summary["mean_reference_TPR99"] if stage16d_summary and stage16d_available else "",
                "delta_hmean_vs_stage16d": stage16d_summary["mean_delta_hmean"] if stage16d_summary and stage16d_available else "",
                "delta_pAUC_vs_stage16d": stage16d_summary["mean_delta_pAUC"] if stage16d_summary and stage16d_available else "",
                "delta_TPR99_vs_stage16d": stage16d_summary["mean_delta_TPR99"] if stage16d_summary and stage16d_available else "",
                "note": "" if not stage16d_available else "Stage16D reproduction comparison available",
            }
        )
    return rows


def build_stage16d_compare_summary(per_machine_rows: list[dict], stage16d_ref_map: dict[str, dict]) -> dict:
    compare_rows = []
    for row in per_machine_rows:
        if row["candidate"] != "min_clamp__mean__baseline_knn_k5":
            continue
        ref = stage16d_ref_map.get(row["machine"])
        if not ref:
            continue
        compare_rows.append(
            {
                "machine": row["machine"],
                "mean_hmean": row["mean_hmean"],
                "mean_pAUC": row["mean_pAUC"],
                "mean_TPR@99n": row["mean_TPR@99n"],
                "reference_hmean": ref.get("reference_hmean"),
                "reference_pAUC": ref.get("reference_pAUC"),
                "reference_TPR99": ref.get("reference_TPR99"),
                "delta_hmean": row["mean_hmean"] - ref.get("reference_hmean"),
                "delta_pAUC": row["mean_pAUC"] - ref.get("reference_pAUC"),
                "delta_TPR99": row["mean_TPR@99n"] - ref.get("reference_TPR99"),
            }
        )
    if not compare_rows:
        return {"available": False}
    max_abs_hmean = float(np.max(np.abs([r["delta_hmean"] for r in compare_rows])))
    max_abs_pauc = float(np.max(np.abs([r["delta_pAUC"] for r in compare_rows])))
    max_abs_tpr99 = float(np.max(np.abs([r["delta_TPR99"] for r in compare_rows])))
    mean_reference_hmean = float(np.mean([r["reference_hmean"] for r in compare_rows]))
    mean_reference_pauc = float(np.mean([r["reference_pAUC"] for r in compare_rows]))
    mean_reference_tpr99 = float(np.mean([r["reference_TPR99"] for r in compare_rows]))
    mean_candidate_hmean = float(np.mean([r["mean_hmean"] for r in compare_rows]))
    mean_candidate_pauc = float(np.mean([r["mean_pAUC"] for r in compare_rows]))
    mean_candidate_tpr99 = float(np.mean([r["mean_TPR@99n"] for r in compare_rows]))
    exact_reproduction = bool(max_abs_hmean == 0.0 and max_abs_pauc == 0.0 and max_abs_tpr99 == 0.0)
    return {
        "available": True,
        "rows": {row["machine"]: row for row in compare_rows},
        "summary": {
            "mean_candidate_hmean": mean_candidate_hmean,
            "mean_candidate_pAUC": mean_candidate_pauc,
            "mean_candidate_TPR99": mean_candidate_tpr99,
            "mean_reference_hmean": mean_reference_hmean,
            "mean_reference_pAUC": mean_reference_pauc,
            "mean_reference_TPR99": mean_reference_tpr99,
            "mean_delta_hmean": mean_candidate_hmean - mean_reference_hmean,
            "mean_delta_pAUC": mean_candidate_pauc - mean_reference_pauc,
            "mean_delta_TPR99": mean_candidate_tpr99 - mean_reference_tpr99,
        },
        "max_abs_hmean_delta": max_abs_hmean,
        "max_abs_pauc_delta": max_abs_pauc,
        "max_abs_tpr99_delta": max_abs_tpr99,
        "exact_reproduction": exact_reproduction,
    }


def score_light_pooling(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    machines = parse_machines(args.machines)

    stage18f_map = load_stage18f_reference_map(Path(args.stage18f_per_machine))
    stage16d_ref_map = load_stage16d_reference_map(Path(args.stage16d_per_machine))

    machine_check_rows = build_machine_check_rows(Path(args.data_root), machines)
    write_csv(out_root / "machine_check.csv", machine_check_rows)

    status = {
        "mode": "score_light_pooling",
        "data_root_exists": Path(args.data_root).exists(),
        "eat_model_path_exists": Path(args.eat_model_path).exists(),
        "stage18f_verdict_exists": Path(args.stage18f_verdict).exists(),
        "stage20a_verdict_exists": Path(args.stage20a_verdict).exists(),
        "stage20a_summary_exists": Path(args.stage20a_summary).exists(),
        "stage20a_candidate_rank_exists": Path(args.stage20a_candidate_rank).exists(),
        "stage18f_per_machine_exists": Path(args.stage18f_per_machine).exists(),
        "stage16d_per_machine_exists": Path(args.stage16d_per_machine).exists(),
        "output_root_writable": ensure_writable(out_root),
        "imports": import_status(),
        "notes": "No model load, no forward, no training, no embedding extraction.",
    }
    write_json(out_root / "check_status.json", status)

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    score_rows = []
    stage16d_stage_rows = []
    stage16d_compare_details = []
    machine_ref_rows = []

    for machine in machines:
        reference_row = None
        if machine in stage18f_map and stage18f_map[machine].get("reference_hmean") is not None:
            reference_row = stage18f_map[machine]
        elif machine in stage16d_ref_map and stage16d_ref_map[machine].get("reference_hmean") is not None:
            reference_row = {
                "source_path": stage16d_ref_map[machine]["source_path"],
                "reference_hmean": stage16d_ref_map[machine]["reference_hmean"],
                "candidate": stage16d_ref_map[machine]["candidate"],
            }
        machine_ref_rows.append(
            {
                "machine": machine,
                "reference_source_path": reference_row["source_path"] if reference_row else "",
                "reference_hmean": reference_row["reference_hmean"] if reference_row else "",
                "reference_candidate": reference_row["candidate"] if reference_row else "",
            }
        )

        for pooling in POOLINGS:
            emb_path = emb_root / pooling / f"{machine}_embeddings.npz"
            if not emb_path.exists():
                raise FileNotFoundError(f"Missing embedding file: {emb_path}")
            arr = np.load(emb_path, allow_pickle=True)
            train_emb = arr["train_embeddings"].astype(np.float64)
            test_emb = arr["test_embeddings"].astype(np.float64)
            y_test = arr["y_test"].astype(np.int64)
            d_test = arr["d_test"].astype(str)
            test_paths = arr["test_paths"].astype(str)

            train_n = l2_normalize_np(train_emb, axis=1)
            test_n = l2_normalize_np(test_emb, axis=1)
            base_dist = chord_distance_matrix(test_n, train_n)
            ref_ref_dist = chord_distance_matrix(train_n, train_n)

            for backend in BACKENDS:
                candidate = f"{pooling}__{backend}"
                candidate_score_dir = scores_root / candidate
                candidate_score_dir.mkdir(parents=True, exist_ok=True)
                if backend == "baseline_knn_k5":
                    scores = baseline_knn_scores(base_dist, score_k=5)
                    rho_vec = None
                elif backend == "ldn_diff_K1_scorek5":
                    scores, rho_vec = ldn_diff_k1_scores(base_dist, ref_ref_dist, score_k=5)
                else:  # pragma: no cover
                    raise RuntimeError(f"Unsupported backend: {backend}")

                summary, tail, geom = evaluate(scores, y_test, d_test)

                np.savez_compressed(
                    candidate_score_dir / f"{machine}_scores.npz",
                    test_scores=scores.astype(np.float64),
                    y_test=y_test.astype(np.int64),
                    d_test=d_test.astype("U16"),
                    test_paths=np.asarray(test_paths, dtype=object),
                    candidate=candidate,
                    pooling=pooling,
                    backend=backend,
                    machine=machine,
                    score_k=5,
                )

                per_machine_rows.append(
                    {
                        "machine": machine,
                        "candidate": candidate,
                        "pooling": pooling,
                        "backend": backend,
                        **summary,
                    }
                )
                tail_rows.append(
                    {
                        "machine": machine,
                        "candidate": candidate,
                        "pooling": pooling,
                        "backend": backend,
                        **tail,
                    }
                )
                geom_rows.append(
                    {
                        "machine": machine,
                        "candidate": candidate,
                        "pooling": pooling,
                        "backend": backend,
                        **geom,
                    }
                )

                if candidate == "min_clamp__mean__baseline_knn_k5" and machine in stage16d_ref_map:
                    ref = stage16d_ref_map[machine]
                    stage16d_stage_rows.append(
                        {
                            "machine": machine,
                            "candidate": candidate,
                            "mean_hmean": summary["hmean"],
                            "mean_pAUC": summary["pAUC"],
                            "mean_TPR@99n": summary["TPR@99n"],
                            "reference_hmean": ref.get("reference_hmean"),
                            "reference_pAUC": ref.get("reference_pAUC"),
                            "reference_TPR99": ref.get("reference_TPR99"),
                            "delta_hmean": summary["hmean"] - ref.get("reference_hmean"),
                            "delta_pAUC": summary["pAUC"] - ref.get("reference_pAUC"),
                            "delta_TPR99": summary["TPR@99n"] - ref.get("reference_TPR99"),
                        }
                    )

    collapse_rows = collapse_rows_from_per_machine(per_machine_rows, build_reference_hmean_map(stage18f_map, per_machine_rows))
    summary_rows = aggregate_summary(per_machine_rows, collapse_rows)
    candidate_rank_rows = build_candidate_rank(summary_rows, collapse_rows)
    tail_summary_rows = score_tail_summary(tail_rows)
    geom_summary_rows = score_geometry_summary(geom_rows)
    stage16d_compare = build_stage16d_compare_summary(stage16d_stage_rows, stage16d_ref_map)

    write_csv(out_root / "machine_check.csv", machine_check_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "candidate_rank.csv", candidate_rank_rows)
    write_csv(out_root / "tail_table.csv", tail_summary_rows)
    write_csv(out_root / "score_geometry.csv", geom_summary_rows)
    write_csv(out_root / "comparison_vs_stage18f.csv", build_comparison_vs_stage18f(summary_rows, stage16d_compare))
    write_csv(out_root / "collapse_check.csv", collapse_rows)

    best = candidate_rank_rows[0]
    best_candidate = best["candidate"]
    best_rows = [r for r in per_machine_rows if r["candidate"] == best_candidate]
    strong_collapse = any(row["strong_machine"] and row["severe_collapse_gt_0p02"] for row in collapse_rows if row["candidate"] == best_candidate)
    improved_machine_count = sum(1 for row in collapse_rows if row["candidate"] == best_candidate and row["delta"] is not None and row["delta"] > 0.0)
    passes_minimum_improvement = best["mean_hmean"] > STAGE18F_BEST_HMEAN
    passes_0p620 = best["mean_hmean"] >= 0.620
    strong_machine_guardrail_pass = not strong_collapse

    if strong_collapse:
        promotion_recommended = "false"
        recommended_next_step = "stop light-pooling branch or move to carefully preregistered fixed fusion audit"
    elif best["mean_hmean"] <= STAGE18F_BEST_HMEAN:
        promotion_recommended = "false"
        recommended_next_step = "stop light-pooling branch or move to carefully preregistered fixed fusion audit"
    elif best["mean_hmean"] >= 0.620 and strong_machine_guardrail_pass:
        promotion_recommended = "true"
        recommended_next_step = "Stage20C route freeze / paper packaging"
    else:
        promotion_recommended = "maybe"
        recommended_next_step = "freeze only as ablation unless very stable"

    verdict = {
        "stage": "Stage20B",
        "best_candidate": best_candidate,
        "best_pooling": best["pooling"],
        "best_backend": best["backend"],
        "best_mean_hmean": best["mean_hmean"],
        "best_mean_pAUC": best["mean_pAUC"],
        "best_mean_TPR99": best["mean_TPR@99n"],
        "current_stage18f_best_hmean": STAGE18F_BEST_HMEAN,
        "delta_vs_stage18f_best": best["mean_hmean"] - STAGE18F_BEST_HMEAN,
        "passes_minimum_improvement": passes_minimum_improvement,
        "passes_0p620": passes_0p620,
        "strong_machine_guardrail_pass": strong_machine_guardrail_pass,
        "improved_machine_count_vs_stage18f": int(improved_machine_count),
        "negative": promotion_recommended != "true",
        "promotion_recommended": promotion_recommended,
        "recommended_next_step": recommended_next_step,
        "stage16d_compare_available": stage16d_compare.get("available", False),
        "stage16d_max_abs_hmean_delta": stage16d_compare.get("max_abs_hmean_delta", ""),
        "stage16d_max_abs_pauc_delta": stage16d_compare.get("max_abs_pauc_delta", ""),
        "stage16d_max_abs_tpr99_delta": stage16d_compare.get("max_abs_tpr99_delta", ""),
        "stage16d_exact_reproduction": stage16d_compare.get("exact_reproduction", ""),
        "notes": [
            "full7 audit",
            "no selector",
            "no fusion",
            "no training",
            "no test-label candidate selection",
        ],
    }
    write_json(out_root / "stage20b_light_pooling_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_markdown(verdict, candidate_rank_rows, stage16d_compare))


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    eat_model_path = Path(args.eat_model_path)

    machine_check_rows = build_machine_check_rows(data_root, parse_machines(args.machines))
    status = {
        "mode": "check",
        "data_root_exists": data_root.exists(),
        "eat_model_path_exists": eat_model_path.exists(),
        "stage18f_verdict_exists": Path(args.stage18f_verdict).exists(),
        "stage20a_verdict_exists": Path(args.stage20a_verdict).exists(),
        "stage20a_summary_exists": Path(args.stage20a_summary).exists(),
        "stage20a_candidate_rank_exists": Path(args.stage20a_candidate_rank).exists(),
        "stage18f_per_machine_exists": Path(args.stage18f_per_machine).exists(),
        "stage16d_per_machine_exists": Path(args.stage16d_per_machine).exists(),
        "output_root_writable": ensure_writable(out_root),
        "machine_check": machine_check_rows,
        "imports": import_status(),
        "notes": "No model load, no forward, no embedding extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "machine_check.csv", machine_check_rows)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_light_pooling":
        extract_and_pool(args)
        return
    if args.mode == "score_light_pooling":
        score_light_pooling(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
