#!/usr/bin/env python
"""Stage 19C: fan adapted EAT embedding evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ADAPTED_CKPT_DEFAULT = "results/stage19b_fan_dap_ssl_sanity/checkpoints/fan_stage19b_best.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE19B_ROOT_DEFAULT = "results/stage19b_fan_dap_ssl_sanity"
STAGE18E_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
OUTPUT_ROOT_DEFAULT = "results/stage19c_fan_adapted_eval"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19C_FAN_ADAPTED_EVAL_AUDIT.md"
MACHINE_DEFAULT = "fan"
EPS = 1e-12

STAGE18_BASELINE_HMEAN = 0.5820070015781968
STAGE18_BASELINE_PAUC = 0.5421052631578948
STAGE18_BASELINE_TPR99 = 0.04

STAGE18_LDN_HMEAN = 0.5527191284766243
STAGE18_LDN_PAUC = 0.5252631578947369
STAGE18_LDN_TPR99 = 0.02


def parse_args():
    p = argparse.ArgumentParser(description="Stage19C fan adapted embedding evaluation")
    p.add_argument("--mode", required=True, choices=["check", "inspect_checkpoint", "extract_adapted_embeddings", "score_adapted_fan"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--adapted_ckpt_path", default=ADAPTED_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage19b_root", default=STAGE19B_ROOT_DEFAULT)
    p.add_argument("--stage18e_root", default=STAGE18E_ROOT_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
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


def load_json(path: Path):
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


def load_clean_eat_local(model_path: Path, device: str):
    import torch

    package_parent = model_path.parent.resolve()
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from checkpoints.configuration_eat import EATConfig
    from checkpoints.modeling_eat import EATModel

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    try:
        model = EATModel.from_pretrained(str(model_path), trust_remote_code=False)
    except Exception:
        from safetensors.torch import load_file

        cfg = EATConfig.from_pretrained(str(model_path))
        model = EATModel(cfg)
        state = load_file(str(model_path / "model.safetensors"))
        model.load_state_dict(state, strict=True)

    model.to(requested)
    model.eval()
    return model, requested


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad_(False)


def enable_last_n_blocks(model, n_last: int) -> list[str]:
    names = []
    if n_last <= 0:
        return names
    if not hasattr(model, "model") or not hasattr(model.model, "blocks"):
        raise RuntimeError("Clean EAT model does not expose model.blocks.")
    blocks = model.model.blocks
    total = len(blocks)
    start = max(0, total - n_last)
    for idx in range(start, total):
        for param in blocks[idx].parameters():
            param.requires_grad_(True)
        names.append(f"eat.model.blocks.{idx}")
    return names


def build_reconstruction_head(device):
    import torch.nn as nn

    head = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, 128),
    )
    head.to(device)
    return head


def parameter_audit_rows(model, recon_head) -> list[dict]:
    rows = []
    for scope, module in [("eat", model), ("recon_head", recon_head)]:
        for name, param in module.named_parameters():
            rows.append({
                "parameter_name": f"{scope}.{name}",
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
            })
    return rows


def load_stage19b_adapted(model_path: Path, adapted_ckpt_path: Path, device: str):
    import torch

    model, requested = load_clean_eat_local(model_path, device)
    freeze_all_parameters(model)

    blob = torch.load(str(adapted_ckpt_path), map_location=requested)
    if not isinstance(blob, dict):
        raise RuntimeError(f"Unexpected Stage19B checkpoint type: {type(blob)}")
    if "student_state_dict" not in blob:
        raise RuntimeError("Stage19B checkpoint missing student_state_dict.")

    train_cfg = blob.get("train_config", {})
    trainable_last_blocks = int(train_cfg.get("trainable_last_blocks", 1))
    allowed_blocks = enable_last_n_blocks(model, trainable_last_blocks)
    recon_head = build_reconstruction_head(requested)
    if "recon_head_state_dict" in blob:
        recon_head.load_state_dict(blob["recon_head_state_dict"], strict=True)
    model.load_state_dict(blob["student_state_dict"], strict=True)
    model.eval()
    return model, recon_head, requested, blob, allowed_blocks


def load_waveform_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != 16000:
        tensor = torchaudio.functional.resample(tensor, int(sr), 16000)
    return tensor.detach().cpu().numpy().astype(np.float32)


def waveform_to_eat_tensor(wav_16k: np.ndarray):
    import torch
    import torchaudio

    wav = torch.tensor(wav_16k, dtype=torch.float32)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=16000,
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
    mel = (mel - (-4.268)) / (4.569 * 2)
    return mel.to(dtype=torch.float32)


def extract_frame_output(model, mel_batch, device):
    import torch

    x = mel_batch.unsqueeze(1).to(device)
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
    return feat


def min_clamp(frames: np.ndarray) -> np.ndarray:
    return np.where(np.abs(frames) < 0.1, 0.0, frames).astype(np.float32)


def mean_pool(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0, dtype=np.float64).astype(np.float64)


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def chord_distance_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = l2_normalize(query.astype(np.float64), axis=1)
    r = l2_normalize(ref.astype(np.float64), axis=1)
    qn2 = np.sum(q * q, axis=1, keepdims=True)
    rn2 = np.sum(r * r, axis=1)[None, :]
    d2 = qn2 + rn2 - 2.0 * (q @ r.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float64)


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
        "# CTAF Stage19C Fan Adapted Eval Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- clean EAT checkpoint exists: `{status['eat_ckpt_exists']}`",
        f"- adapted checkpoint exists: `{status['adapted_ckpt_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- Stage19B root exists: `{status['stage19b_root_exists']}`",
        f"- Stage18E root exists: `{status['stage18e_root_exists']}`",
        f"- Stage18F root exists: `{status['stage18f_root_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    eat_ckpt = Path(args.eat_model_path)
    adapted_ckpt = Path(args.adapted_ckpt_path)
    data_root = Path(args.data_root)
    stage19b_root = Path(args.stage19b_root)
    stage18e_root = Path(args.stage18e_root)
    stage18f_root = Path(args.stage18f_root)

    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, args.machine) if items else []
    train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])

    status = {
        "mode": "check",
        "machine": args.machine,
        "eat_ckpt_exists": eat_ckpt.exists(),
        "adapted_ckpt_exists": adapted_ckpt.exists(),
        "data_root_exists": data_root.exists(),
        "stage19b_root_exists": stage19b_root.exists(),
        "stage18e_root_exists": stage18e_root.exists(),
        "stage18f_root_exists": stage18f_root.exists(),
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "imports": import_status(),
        "notes": "No model load, no forward, no extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", status)


def run_inspect_checkpoint(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stage19b_root = Path(args.stage19b_root)
    adapted_ckpt = Path(args.adapted_ckpt_path)
    if not adapted_ckpt.exists():
        raise FileNotFoundError(f"Missing adapted checkpoint: {adapted_ckpt}")
    if not (stage19b_root / "stage19b_train_config.json").exists():
        raise FileNotFoundError(f"Missing Stage19B train config in {stage19b_root}")

    model, recon_head, device, blob, allowed_blocks = load_stage19b_adapted(Path(args.eat_model_path), adapted_ckpt, args.device)
    audit_rows = parameter_audit_rows(model, recon_head)
    stage19b_summary = load_json(stage19b_root / "trainable_summary.json")
    checkpoint_status = {
        "mode": "inspect_checkpoint",
        "adapted_ckpt_path": str(adapted_ckpt),
        "device": str(device),
        "checkpoint_top_keys": list(blob.keys())[:20],
        "contains_student_state_dict": "student_state_dict" in blob,
        "contains_recon_head_state_dict": "recon_head_state_dict" in blob,
        "contains_train_config": "train_config" in blob,
    }
    recap = {
        "stage19b_trainable_last_blocks": blob.get("train_config", {}).get("trainable_last_blocks"),
        "stage19b_best_epoch": blob.get("best_epoch"),
        "stage19b_best_val_loss": blob.get("best_val_loss"),
        "allowed_block_names": allowed_blocks,
        "stage19b_trainable_boundary_clean": stage19b_summary.get("no_unexpected_earlier_layer_trainable_params"),
    }
    write_json(out_root / "checkpoint_status.json", checkpoint_status)
    write_csv(out_root / "adapted_parameter_audit.csv", audit_rows)
    write_json(out_root / "trainable_boundary_recap.json", recap)


def run_extract_adapted_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    adapted_ckpt = Path(args.adapted_ckpt_path)
    if not adapted_ckpt.exists():
        raise FileNotFoundError(f"Missing adapted checkpoint: {adapted_ckpt}")

    model, recon_head, device, blob, allowed_blocks = load_stage19b_adapted(Path(args.eat_model_path), adapted_ckpt, args.device)
    items = filter_items(scan_dataset(Path(args.data_root)), args.machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)}, test={len(test_items)}")

    train_embs = []
    test_embs = []
    sample_frame_shape = None
    y_test = []
    d_test = []
    test_paths = []
    for item in train_items:
        wav = load_waveform_16k(Path(item.path))
        mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            frames = extract_frame_output(model, mel, device)
        frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
        frame_np = min_clamp(frame_np)
        train_embs.append(mean_pool(frame_np))
        sample_frame_shape = list(frame_np.shape)
    for item in test_items:
        wav = load_waveform_16k(Path(item.path))
        mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            frames = extract_frame_output(model, mel, device)
        frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
        frame_np = min_clamp(frame_np)
        test_embs.append(mean_pool(frame_np))
        y_test.append(int(item.label))
        d_test.append(item.domain)
        test_paths.append(str(item.path))
        sample_frame_shape = list(frame_np.shape)

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)
    d_test_arr = np.asarray(d_test, dtype="U16")
    test_paths_arr = np.asarray(test_paths, dtype=object)

    np.savez_compressed(
        emb_root / "fan_adapted_minclamp_mean_embeddings.npz",
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test_arr,
        d_test=d_test_arr,
        test_paths=test_paths_arr,
        machine=args.machine,
        checkpoint_path=str(adapted_ckpt),
        preprocess="min_clamp",
        pooling="mean",
        embedding_source="stage19b_adapted_eat",
    )
    write_csv(out_root / "embedding_summary.csv", [{
        "machine": args.machine,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "embedding_dim": int(train_emb.shape[1]),
        "frame_shape": sample_frame_shape,
        "checkpoint_path": str(adapted_ckpt),
        "embedding_path": str(emb_root / "fan_adapted_minclamp_mean_embeddings.npz"),
    }])


def run_score_adapted_fan(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_path = out_root / "embeddings" / "fan_adapted_minclamp_mean_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing adapted embedding file: {emb_path}")

    arr = np.load(emb_path, allow_pickle=True)
    train_emb = arr["train_embeddings"].astype(np.float64)
    test_emb = arr["test_embeddings"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    test_paths = arr["test_paths"].astype(str)

    dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref = chord_distance_matrix(train_emb, train_emb)
    rho_k1 = local_density_knn(ref_ref, K=1)

    score_baseline = baseline_knn_scores(dist, score_k=args.latent_knn_k)
    score_ldn = baseline_knn_scores(dist - rho_k1[None, :], score_k=args.latent_knn_k)

    rows = []
    tail_rows = []
    geom_rows = []
    scores_root = out_root / "scores"
    (scores_root / "adapted_baseline_knn_k5").mkdir(parents=True, exist_ok=True)
    (scores_root / "adapted_ldn_diff_K1_scorek5").mkdir(parents=True, exist_ok=True)

    for candidate, scores in {
        "adapted_baseline_knn_k5": score_baseline,
        "adapted_ldn_diff_K1_scorek5": score_ldn,
    }.items():
        summary, tail, geom = evaluate(scores, y_test, d_test)
        rows.append({"candidate": candidate, **summary})
        tail_rows.append({"candidate": candidate, **tail})
        geom_rows.append({"candidate": candidate, **geom})
        np.savez(
            scores_root / candidate / "fan_scores.npz",
            test_scores=scores.astype(np.float64),
            y_test=y_test,
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            candidate=candidate,
            machine=args.machine,
        )

    write_csv(out_root / "summary.csv", rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    ldn_row = next(r for r in rows if r["candidate"] == "adapted_ldn_diff_K1_scorek5")
    baseline_row = next(r for r in rows if r["candidate"] == "adapted_baseline_knn_k5")
    comparison_rows = [
        {
            "candidate": "stage18_baseline_knn_k5",
            "hmean": STAGE18_BASELINE_HMEAN,
            "pAUC": STAGE18_BASELINE_PAUC,
            "TPR@99n": STAGE18_BASELINE_TPR99,
            "delta_hmean_vs_stage18_ldn": STAGE18_BASELINE_HMEAN - STAGE18_LDN_HMEAN,
            "delta_hmean_vs_stage18_baseline": 0.0,
            "pass_minimum_gate": True,
            "pass_strong_gate": True,
        },
        {
            "candidate": "stage18_ldn_diff_K1_scorek5",
            "hmean": STAGE18_LDN_HMEAN,
            "pAUC": STAGE18_LDN_PAUC,
            "TPR@99n": STAGE18_LDN_TPR99,
            "delta_hmean_vs_stage18_ldn": 0.0,
            "delta_hmean_vs_stage18_baseline": STAGE18_LDN_HMEAN - STAGE18_BASELINE_HMEAN,
            "pass_minimum_gate": False,
            "pass_strong_gate": False,
        },
        {
            "candidate": "adapted_baseline_knn_k5",
            "hmean": baseline_row["hmean"],
            "pAUC": baseline_row["pAUC"],
            "TPR@99n": baseline_row["TPR@99n"],
            "delta_hmean_vs_stage18_ldn": baseline_row["hmean"] - STAGE18_LDN_HMEAN,
            "delta_hmean_vs_stage18_baseline": baseline_row["hmean"] - STAGE18_BASELINE_HMEAN,
            "pass_minimum_gate": baseline_row["hmean"] > STAGE18_LDN_HMEAN,
            "pass_strong_gate": baseline_row["hmean"] >= STAGE18_BASELINE_HMEAN,
        },
        {
            "candidate": "adapted_ldn_diff_K1_scorek5",
            "hmean": ldn_row["hmean"],
            "pAUC": ldn_row["pAUC"],
            "TPR@99n": ldn_row["TPR@99n"],
            "delta_hmean_vs_stage18_ldn": ldn_row["hmean"] - STAGE18_LDN_HMEAN,
            "delta_hmean_vs_stage18_baseline": ldn_row["hmean"] - STAGE18_BASELINE_HMEAN,
            "pass_minimum_gate": ldn_row["hmean"] > STAGE18_LDN_HMEAN,
            "pass_strong_gate": ldn_row["hmean"] >= STAGE18_BASELINE_HMEAN,
        },
    ]
    write_csv(out_root / "comparison_vs_stage18_fan.csv", comparison_rows)

    verdict = {
        "adapted_ldn_hmean": float(ldn_row["hmean"]),
        "adapted_ldn_pAUC": float(ldn_row["pAUC"]),
        "adapted_ldn_TPR99": float(ldn_row["TPR@99n"]),
        "stage18_ldn_hmean": STAGE18_LDN_HMEAN,
        "stage18_baseline_hmean": STAGE18_BASELINE_HMEAN,
        "delta_vs_stage18_ldn": float(ldn_row["hmean"]) - STAGE18_LDN_HMEAN,
        "delta_vs_stage18_baseline": float(ldn_row["hmean"]) - STAGE18_BASELINE_HMEAN,
        "minimum_pass": float(ldn_row["hmean"]) > STAGE18_LDN_HMEAN,
        "strong_pass": float(ldn_row["hmean"]) >= STAGE18_BASELINE_HMEAN,
        "negative": float(ldn_row["hmean"]) <= STAGE18_LDN_HMEAN,
        "notes": [
            "fan only",
            "no full7 conclusion",
            "no selector",
            "no checkpoint selection by test labels",
        ],
    }
    write_json(out_root / "stage19c_fan_adapted_eval_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage19C Fan Adapted Eval Audit",
        "",
        f"- adapted ldn hmean: `{ldn_row['hmean']:.6f}`",
        f"- adapted ldn pAUC: `{ldn_row['pAUC']:.6f}`",
        f"- adapted ldn TPR@99n: `{ldn_row['TPR@99n']:.6f}`",
        f"- minimum pass: `{verdict['minimum_pass']}`",
        f"- strong pass: `{verdict['strong_pass']}`",
        "",
        "No selector, oracle, or full7 conclusion in this stage.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_checkpoint":
        run_inspect_checkpoint(args)
        return
    if args.mode == "extract_adapted_embeddings":
        run_extract_adapted_embeddings(args)
        return
    if args.mode == "score_adapted_fan":
        run_score_adapted_fan(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
