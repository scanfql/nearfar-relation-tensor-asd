#!/usr/bin/env python
"""Stage 19F: fan anchor-LoRA adapted evaluation.

Fan-only evaluation of the Stage19E anchor-LoRA checkpoint.
No selector, no fusion, no full7, no training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - check mode should report import status
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ADAPTED_CKPT_DEFAULT = "results/stage19e_fan_anchor_lora_sanity/checkpoints/fan_stage19e_anchor_lora_best.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE19E_ROOT_DEFAULT = "results/stage19e_fan_anchor_lora_sanity"
STAGE18E_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
OUTPUT_ROOT_DEFAULT = "results/stage19f_fan_anchor_lora_eval"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19F_FAN_ANCHOR_LORA_EVAL_AUDIT.md"
EMBEDDING_NPZ_DEFAULT = "results/stage19f_fan_anchor_lora_eval/embeddings/fan_anchor_lora_minclamp_mean_embeddings.npz"
MACHINE_DEFAULT = "fan"

STAGE18_BASELINE_HMEAN = 0.5820070015781968
STAGE18_BASELINE_PAUC = 0.5421052631578948
STAGE18_BASELINE_TPR99 = 0.04

STAGE18_LDN_HMEAN = 0.5527191284766243
STAGE18_LDN_PAUC = 0.5252631578947369
STAGE18_LDN_TPR99 = 0.02

STAGE19C_FAILED_HMEAN = 0.5317251523762622
STAGE19C_FAILED_PAUC = 0.4994736842105263
STAGE19C_FAILED_TPR99 = 0.02

EXPECTED_LORA_TARGET_NAMES = [
    "model.blocks.11.attn.qkv",
    "model.blocks.11.attn.proj",
]

REPRESENTATION_GUARDRAIL_MAX_DROP = 0.02


@dataclass(frozen=True)
class Stage19FProtocol:
    trainable_scope: str = "last_block_attention_lora"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05
    batch_size: int = 8
    seed: int = 13711
    sample_rate: int = 16000
    num_mel_bins: int = 128
    frame_shift_ms: int = 10
    target_length: int = 1024
    norm_mean: float = -4.268
    norm_std: float = 4.569
    hidden_size: int = 768
    recon_dim: int = 128
    score_k: int = 5


CFG = Stage19FProtocol()


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="Stage19F fan anchor-LoRA adapted evaluation")
    p.add_argument("--mode", required=True, choices=["check", "inspect_checkpoint", "extract_adapted_embeddings", "score_adapted_fan"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--adapted_ckpt_path", default=ADAPTED_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage19e_root", default=STAGE19E_ROOT_DEFAULT)
    p.add_argument("--stage18e_root", default=STAGE18E_ROOT_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--embedding_npz_path", default=EMBEDDING_NPZ_DEFAULT)
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
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_json(path: Path):
    return json.loads(path.read_text())


def ensure_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok\n")
        probe.unlink()
        return True
    except Exception:
        return False


def expected_items(data_root: Path, machine: str):
    from src.data.dataset import filter_items, scan_dataset, split_train_test

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


def get_block_container(model):
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks
    if hasattr(model, "blocks"):
        return model.blocks
    raise RuntimeError("Clean EAT model does not expose blocks for LoRA targeting.")


def get_parent_module(root_module, module_name: str):
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def is_attention_qkv_or_proj_linear(module_name: str, module, last_block_token: str) -> tuple[bool, str]:
    import torch.nn as nn

    if not isinstance(module, nn.Linear):
        return False, "non_linear"

    name_lower = module_name.lower()
    leaf = module_name.split(".")[-1].lower()
    has_attention_token = ".attn." in name_lower or "attention" in name_lower or "attn" in name_lower
    if last_block_token not in module_name:
        return False, "skip_non_last_block"
    if not has_attention_token:
        return False, "skip_non_attention_linear"
    if leaf == "qkv":
        return True, "selected_last_block_attention_qkv"
    if leaf == "proj":
        return True, "selected_last_block_attention_proj"
    return False, "skip_attention_non_qkv_proj_linear"


def estimate_lora_params(in_features: int, out_features: int, rank: int) -> int:
    return int(rank * (in_features + out_features))


def inspect_lora_targets(model, lora_cfg: LoRAConfig) -> tuple[list[dict], list[str]]:
    import torch.nn as nn

    blocks = get_block_container(model)
    last_idx = len(blocks) - 1
    last_block_token = f"blocks.{last_idx}."

    rows = []
    selected_names = []
    for module_name, module in model.named_modules():
        if not module_name or not isinstance(module, nn.Linear):
            continue
        selected, reason = is_attention_qkv_or_proj_linear(module_name, module, last_block_token)
        if selected:
            selected_names.append(module_name)
        rows.append(
            {
                "module_name": module_name,
                "module_type": module.__class__.__name__,
                "selected": selected,
                "selection_reason": reason,
                "block_prefix": last_block_token,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "has_bias": module.bias is not None,
                "estimated_lora_param_numel": estimate_lora_params(int(module.in_features), int(module.out_features), lora_cfg.rank),
                "lora_rank": int(lora_cfg.rank),
                "lora_alpha": float(lora_cfg.alpha),
                "lora_dropout": float(lora_cfg.dropout),
            }
        )
    return rows, selected_names


def build_lora_linear(base_linear, cfg: LoRAConfig):
    import torch.nn as nn

    class _LoRALinear(nn.Module):
        def __init__(self):
            super().__init__()
            if cfg.rank <= 0:
                raise ValueError(f"LoRA rank must be positive; got {cfg.rank}")
            self.base = base_linear
            self.rank = int(cfg.rank)
            self.alpha = float(cfg.alpha)
            self.scaling = self.alpha / max(self.rank, 1)
            self.dropout = nn.Dropout(float(cfg.dropout)) if cfg.dropout > 0 else nn.Identity()
            self.lora_down = nn.Linear(base_linear.in_features, self.rank, bias=False)
            self.lora_up = nn.Linear(self.rank, base_linear.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            for param in self.base.parameters():
                param.requires_grad_(False)
            self._is_lora_wrapper = True

        def forward(self, x):
            return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling

    return _LoRALinear()


def is_lora_parameter_name(name: str) -> bool:
    return name.endswith("lora_down.weight") or name.endswith("lora_up.weight")


def inject_lora_modules(model, selected_names: list[str], lora_cfg: LoRAConfig):
    freeze_all_parameters(model)
    for module_name in selected_names:
        parent, leaf = get_parent_module(model, module_name)
        base_linear = getattr(parent, leaf)
        wrapped = build_lora_linear(base_linear, lora_cfg)
        wrapped = wrapped.to(device=base_linear.weight.device, dtype=base_linear.weight.dtype)
        setattr(parent, leaf, wrapped)
    return model


def build_reconstruction_head(device, hidden_size: int = 768, recon_dim: int = 128):
    import torch.nn as nn

    head = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, recon_dim),
    )
    head.to(device)
    return head


def parameter_audit_rows(model, recon_head=None) -> list[dict]:
    rows = []
    modules = [("eat", model)]
    if recon_head is not None:
        modules.append(("recon_head", recon_head))
    for scope, module in modules:
        for name, param in module.named_parameters():
            full_name = f"{scope}.{name}"
            if scope == "recon_head":
                category = "recon_head_trainable" if param.requires_grad else "frozen_recon_head"
            elif is_lora_parameter_name(name):
                category = "lora_trainable" if param.requires_grad else "unexpected_frozen_lora"
            elif param.requires_grad:
                category = "unexpected_trainable_base"
            else:
                category = "frozen_base"
            rows.append(
                {
                    "parameter_name": full_name,
                    "numel": int(param.numel()),
                    "requires_grad": bool(param.requires_grad),
                    "category": category,
                }
            )
    return rows


def summarize_trainable(rows: list[dict], selected_names: list[str], lora_cfg: LoRAConfig) -> dict:
    total = sum(int(r["numel"]) for r in rows)
    frozen = sum(int(r["numel"]) for r in rows if not r["requires_grad"])
    trainable = sum(int(r["numel"]) for r in rows if r["requires_grad"])
    lora_trainable = sum(int(r["numel"]) for r in rows if r["category"] == "lora_trainable")
    recon_head_trainable = sum(int(r["numel"]) for r in rows if r["category"] == "recon_head_trainable")
    unexpected = [r["parameter_name"] for r in rows if r["category"] == "unexpected_trainable_base"]
    return {
        "total_parameter_count": int(total),
        "frozen_parameter_count": int(frozen),
        "trainable_parameter_count": int(trainable),
        "trainable_ratio": float(trainable / total) if total > 0 else 0.0,
        "lora_trainable_parameter_count": int(lora_trainable),
        "recon_head_trainable_parameter_count": int(recon_head_trainable),
        "selected_lora_target_names": selected_names,
        "lora_config": asdict(lora_cfg),
        "no_unexpected_trainable_base_params": len(unexpected) == 0,
        "unexpected_trainable_base_parameter_names": unexpected,
    }


def load_stage19e_checkpoint(model_path: Path, adapted_ckpt_path: Path, device: str):
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not adapted_ckpt_path.exists():
        raise FileNotFoundError(f"Missing Stage19E adapted checkpoint: {adapted_ckpt_path}")

    model, requested = load_clean_eat_local(model_path, device)
    freeze_all_parameters(model)
    lora_cfg = LoRAConfig()
    target_rows, selected_names = inspect_lora_targets(model, lora_cfg)
    if selected_names != EXPECTED_LORA_TARGET_NAMES:
        raise RuntimeError(
            f"Stage19E LoRA targets do not match expected targets: {selected_names} vs {EXPECTED_LORA_TARGET_NAMES}"
        )
    model = inject_lora_modules(model, selected_names, lora_cfg)

    blob = torch.load(str(adapted_ckpt_path), map_location=requested)
    if not isinstance(blob, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(blob)}")
    if "student_state_dict" not in blob:
        raise RuntimeError("Stage19E checkpoint missing student_state_dict.")
    if "recon_head_state_dict" not in blob:
        raise RuntimeError("Stage19E checkpoint missing recon_head_state_dict.")

    checkpoint_targets = blob.get("lora_target_names", [])
    if checkpoint_targets and list(checkpoint_targets) != EXPECTED_LORA_TARGET_NAMES:
        raise RuntimeError(
            f"Checkpoint LoRA targets do not match expected targets: {checkpoint_targets} vs {EXPECTED_LORA_TARGET_NAMES}"
        )

    checkpoint_lora_cfg = blob.get("lora_config", {})
    if checkpoint_lora_cfg:
        expected_lora_cfg = asdict(lora_cfg)
        for key in ["rank", "alpha", "dropout"]:
            if float(checkpoint_lora_cfg.get(key, expected_lora_cfg[key])) != float(expected_lora_cfg[key]):
                raise RuntimeError(
                    f"Checkpoint LoRA config mismatch for {key}: {checkpoint_lora_cfg.get(key)} vs {expected_lora_cfg[key]}"
                )

    train_cfg = blob.get("train_config", {})
    if train_cfg.get("trainable_scope") and train_cfg.get("trainable_scope") != CFG.trainable_scope:
        raise RuntimeError(f"Unexpected trainable_scope in checkpoint: {train_cfg.get('trainable_scope')}")

    model.load_state_dict(blob["student_state_dict"], strict=True)
    recon_head = build_reconstruction_head(requested, hidden_size=CFG.hidden_size, recon_dim=CFG.recon_dim)
    recon_head.load_state_dict(blob["recon_head_state_dict"], strict=True)
    model.eval()
    recon_head.eval()
    return {
        "model": model,
        "recon_head": recon_head,
        "device": requested,
        "blob": blob,
        "selected_names": selected_names,
        "target_rows": target_rows,
        "lora_cfg": lora_cfg,
    }


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
        sample_frequency=CFG.sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=CFG.num_mel_bins,
        dither=0.0,
        frame_shift=CFG.frame_shift_ms,
    )
    if mel.shape[0] < CFG.target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, CFG.target_length - mel.shape[0]))
    else:
        mel = mel[:CFG.target_length, :]
    mel = (mel - CFG.norm_mean) / (CFG.norm_std * 2)
    return mel.to(dtype=torch.float32)


def build_batch(paths: list[Path], device):
    import torch

    feats = [waveform_to_eat_tensor(load_waveform_16k(path)) for path in paths]
    mel = torch.stack(feats, dim=0)
    return mel.to(device)


def extract_frame_output(model, mel_batch, device):
    import torch

    # Keep the helper interface aligned with the EAT model expectation:
    # [B, T, F] -> unsqueeze channel to [B, 1, T, F].
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


def min_clamp_frames(frames):
    import torch

    return torch.where(torch.abs(frames) < 0.1, torch.zeros_like(frames), frames)


def min_clamp_np(frames: np.ndarray) -> np.ndarray:
    return np.where(np.abs(frames) < 0.1, 0.0, frames).astype(np.float32)


def mean_pool_frames(frames):
    return frames.mean(dim=1)


def mean_pool_np(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0, dtype=np.float64).astype(np.float64)


def l2_normalize(x):
    import torch.nn.functional as F

    return F.normalize(x, dim=1)


def l2_normalize_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    eps = 1e-12
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def chord_distance_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = l2_normalize_np(query.astype(np.float64), axis=1)
    r = l2_normalize_np(ref.astype(np.float64), axis=1)
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


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    eat_ckpt = Path(args.eat_model_path)
    adapted_ckpt = Path(args.adapted_ckpt_path)
    stage19e_root = Path(args.stage19e_root)
    stage18e_root = Path(args.stage18e_root)
    stage18f_root = Path(args.stage18f_root)

    items, train_items, test_items = expected_items(data_root, args.machine) if data_root.exists() else ([], [], [])
    check_status = {
        "mode": "check",
        "machine": args.machine,
        "eat_model_path_exists": eat_ckpt.exists(),
        "adapted_ckpt_path_exists": adapted_ckpt.exists(),
        "data_root_exists": data_root.exists(),
        "stage19e_root_exists": stage19e_root.exists(),
        "stage18e_root_exists": stage18e_root.exists(),
        "stage18f_root_exists": stage18f_root.exists(),
        "output_root_writable": ensure_writable(out_root),
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "inventory_count": len(items),
        "imports": import_status(),
        "notes": "No model load, no forward, no extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", check_status)


def run_inspect_checkpoint(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        artifact = load_stage19e_checkpoint(Path(args.eat_model_path), Path(args.adapted_ckpt_path), args.device)
        model = artifact["model"]
        recon_head = artifact["recon_head"]
        blob = artifact["blob"]
        selected_names = artifact["selected_names"]
        lora_cfg = artifact["lora_cfg"]
        audit_rows = parameter_audit_rows(model, recon_head)
        summary = summarize_trainable(audit_rows, selected_names, lora_cfg)

        checkpoint_status = {
            "mode": "inspect_checkpoint",
            "checkpoint_path": str(Path(args.adapted_ckpt_path)),
            "checkpoint_exists": True,
            "checkpoint_size_bytes": int(Path(args.adapted_ckpt_path).stat().st_size),
            "checkpoint_key_count": len(blob),
            "checkpoint_top_keys": list(blob.keys())[:20],
            "contains_student_state_dict": "student_state_dict" in blob,
            "contains_recon_head_state_dict": "recon_head_state_dict" in blob,
            "contains_train_config": "train_config" in blob,
            "contains_lora_config": "lora_config" in blob,
            "contains_lora_target_names": "lora_target_names" in blob,
            "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
            "checkpoint_lora_target_names": list(blob.get("lora_target_names", [])),
            "lora_target_match": list(blob.get("lora_target_names", [])) == EXPECTED_LORA_TARGET_NAMES,
            "lora_config_match": blob.get("lora_config", {}) == asdict(lora_cfg),
        }
        lora_load_status = {
            "mode": "inspect_checkpoint",
            "checkpoint_path": str(Path(args.adapted_ckpt_path)),
            "loaded_successfully": True,
            "student_state_dict_loaded": True,
            "recon_head_state_dict_loaded": True,
            "strict_load": True,
            "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
            "selected_lora_target_names": selected_names,
            "checkpoint_top_keys": list(blob.keys())[:20],
            "checkpoint_key_count": len(blob),
            "lora_config": asdict(lora_cfg),
            "trainable_scope": CFG.trainable_scope,
        }
        recap = {
            "checkpoint_path": str(Path(args.adapted_ckpt_path)),
            "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
            "checkpoint_lora_target_names": list(blob.get("lora_target_names", [])),
            "lora_target_match": list(blob.get("lora_target_names", [])) == EXPECTED_LORA_TARGET_NAMES,
            "checkpoint_lora_config": blob.get("lora_config", {}),
            "checkpoint_train_config_preview": {
                "trainable_scope": blob.get("train_config", {}).get("trainable_scope"),
                "mask_ratio": blob.get("train_config", {}).get("mask_ratio"),
                "lambda_anchor": blob.get("train_config", {}).get("lambda_anchor"),
                "epochs": blob.get("train_config", {}).get("epochs"),
                "lr": blob.get("train_config", {}).get("lr"),
                "early_stop_patience": blob.get("train_config", {}).get("early_stop_patience"),
            },
            "checkpoint_trainable_summary_preview": {
                "selected_lora_target_names": blob.get("trainable_summary", {}).get("selected_lora_target_names"),
                "no_unexpected_trainable_base_params": blob.get("trainable_summary", {}).get("no_unexpected_trainable_base_params"),
            },
            "parameter_summary": summary,
            "no_unexpected_trainable_base_params": summary["no_unexpected_trainable_base_params"],
            "unexpected_trainable_base_parameter_names": summary["unexpected_trainable_base_parameter_names"],
        }
        write_json(out_root / "checkpoint_status.json", checkpoint_status)
        write_json(out_root / "lora_load_status.json", lora_load_status)
        write_csv(out_root / "adapted_parameter_audit.csv", audit_rows)
        write_json(out_root / "trainable_boundary_recap.json", recap)
    except Exception as exc:
        write_json(
            out_root / "checkpoint_status.json",
            {
                "mode": "inspect_checkpoint",
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "loaded_successfully": False,
                "error": str(exc),
            },
        )
        write_json(
            out_root / "lora_load_status.json",
            {
                "mode": "inspect_checkpoint",
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "loaded_successfully": False,
                "error": str(exc),
            },
        )
        write_json(
            out_root / "trainable_boundary_recap.json",
            {
                "mode": "inspect_checkpoint",
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "loaded_successfully": False,
                "error": str(exc),
            },
        )
        raise


def run_extract_adapted_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    import torch

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    artifact = load_stage19e_checkpoint(Path(args.eat_model_path), Path(args.adapted_ckpt_path), args.device)
    model = artifact["model"]
    device = artifact["device"]

    from src.data.dataset import filter_items, scan_dataset, split_train_test

    items = filter_items(scan_dataset(data_root), args.machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)}, test={len(test_items)}")

    train_embs = []
    test_embs = []
    y_test = []
    d_test = []
    test_paths = []
    sample_frame_shape = None

    with np.errstate(all="ignore"):
        for split_name, split_items, split_embs in [
            ("train", train_items, train_embs),
            ("test", test_items, test_embs),
        ]:
            for start in range(0, len(split_items), CFG.batch_size):
                batch = split_items[start : start + CFG.batch_size]
                paths = [Path(item.path) for item in batch]
                mel_batch = build_batch(paths, device)
                with torch.no_grad():
                    frames = extract_frame_output(model, mel_batch, device)
                frames_np = frames.detach().cpu().numpy().astype(np.float32)
                for idx, item in enumerate(batch):
                    frame_np = min_clamp_np(frames_np[idx])
                    split_embs.append(mean_pool_np(frame_np))
                    sample_frame_shape = list(frame_np.shape)
                    if split_name == "test":
                        y_test.append(int(item.label))
                        d_test.append(item.domain)
                        test_paths.append(str(item.path))

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)
    d_test_arr = np.asarray(d_test, dtype="U16")
    test_paths_arr = np.asarray(test_paths, dtype=object)
    emb_path = emb_root / "fan_anchor_lora_minclamp_mean_embeddings.npz"
    np.savez_compressed(
        emb_path,
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test_arr,
        d_test=d_test_arr,
        test_paths=test_paths_arr,
        machine=args.machine,
        checkpoint_path=str(Path(args.adapted_ckpt_path)),
        preprocess="min_clamp",
        pooling="mean",
        embedding_source="stage19e_anchor_lora_eat",
    )
    write_csv(
        out_root / "embedding_summary.csv",
        [
            {
                "machine": args.machine,
                "train_count": len(train_items),
                "test_count": len(test_items),
                "embedding_dim": int(train_emb.shape[1]),
                "frame_shape": str(sample_frame_shape),
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "embedding_path": str(emb_path),
                "preprocess": "min_clamp",
                "pooling": "mean",
                "embedding_source": "stage19e_anchor_lora_eat",
            }
        ],
    )


def build_candidate_rows(summary_rows: list[dict]) -> list[dict]:
    by_candidate = {row["candidate"]: row for row in summary_rows}
    stage18_baseline = {
        "candidate": "stage18_baseline_knn_k5",
        "hmean": STAGE18_BASELINE_HMEAN,
        "pAUC": STAGE18_BASELINE_PAUC,
        "TPR@99n": STAGE18_BASELINE_TPR99,
        "delta_hmean_vs_stage18_ldn": STAGE18_BASELINE_HMEAN - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": 0.0,
        "delta_hmean_vs_stage19c_failed": STAGE18_BASELINE_HMEAN - STAGE19C_FAILED_HMEAN,
        "pass_minimum_gate": True,
        "pass_strong_gate": True,
    }
    stage18_ldn = {
        "candidate": "stage18_ldn_diff_K1_scorek5",
        "hmean": STAGE18_LDN_HMEAN,
        "pAUC": STAGE18_LDN_PAUC,
        "TPR@99n": STAGE18_LDN_TPR99,
        "delta_hmean_vs_stage18_ldn": 0.0,
        "delta_hmean_vs_stage18_baseline": STAGE18_LDN_HMEAN - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19c_failed": STAGE18_LDN_HMEAN - STAGE19C_FAILED_HMEAN,
        "pass_minimum_gate": False,
        "pass_strong_gate": False,
    }
    stage19c_failed = {
        "candidate": "stage19c_failed_adapted_ldn_diff_K1",
        "hmean": STAGE19C_FAILED_HMEAN,
        "pAUC": STAGE19C_FAILED_PAUC,
        "TPR@99n": STAGE19C_FAILED_TPR99,
        "delta_hmean_vs_stage18_ldn": STAGE19C_FAILED_HMEAN - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": STAGE19C_FAILED_HMEAN - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19c_failed": 0.0,
        "pass_minimum_gate": False,
        "pass_strong_gate": False,
    }
    adapted_baseline_src = by_candidate["adapted_baseline_knn_k5"]
    adapted_ldn_src = by_candidate["adapted_ldn_diff_K1_scorek5"]
    adapted_baseline = {
        "candidate": "adapted_baseline_knn_k5",
        "hmean": adapted_baseline_src["hmean"],
        "pAUC": adapted_baseline_src["pAUC"],
        "TPR@99n": adapted_baseline_src["TPR@99n"],
        "delta_hmean_vs_stage18_ldn": adapted_baseline_src["hmean"] - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": adapted_baseline_src["hmean"] - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19c_failed": adapted_baseline_src["hmean"] - STAGE19C_FAILED_HMEAN,
        "pass_minimum_gate": adapted_baseline_src["hmean"] > STAGE18_LDN_HMEAN,
        "pass_strong_gate": adapted_baseline_src["hmean"] >= STAGE18_BASELINE_HMEAN,
    }
    adapted_ldn = {
        "candidate": "adapted_ldn_diff_K1_scorek5",
        "hmean": adapted_ldn_src["hmean"],
        "pAUC": adapted_ldn_src["pAUC"],
        "TPR@99n": adapted_ldn_src["TPR@99n"],
        "delta_hmean_vs_stage18_ldn": adapted_ldn_src["hmean"] - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": adapted_ldn_src["hmean"] - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19c_failed": adapted_ldn_src["hmean"] - STAGE19C_FAILED_HMEAN,
        "pass_minimum_gate": adapted_ldn_src["hmean"] > STAGE18_LDN_HMEAN,
        "pass_strong_gate": adapted_ldn_src["hmean"] >= STAGE18_BASELINE_HMEAN,
    }
    return [stage18_baseline, stage18_ldn, stage19c_failed, adapted_baseline, adapted_ldn]


def build_audit_markdown(verdict: dict, comparison_rows: list[dict], summary_rows: list[dict]) -> list[str]:
    adapted_ldn = next(row for row in summary_rows if row["candidate"] == "adapted_ldn_diff_K1_scorek5")
    adapted_baseline = next(row for row in summary_rows if row["candidate"] == "adapted_baseline_knn_k5")
    return [
        "# CTAF Stage19F Fan Anchor-LoRA Adapted Evaluation Audit",
        "",
        "## Protocol",
        "",
        "- machine: `fan`",
        "- adapted checkpoint: `Stage19E anchor-LoRA`",
        "- evaluation data: `fan source train-normal / test split`",
        "- preprocess: `16 kHz`, `128-bin Kaldi fbank`, `frame shift 10 ms`, `target length 1024`",
        "- normalization: `(mel - (-4.268)) / (4.569 * 2)`",
        "- frame output: `[T, 768]`",
        "- min-clamp: `abs(x) < 0.1 -> 0`",
        "- pooling: `mean`",
        "- scoring candidates: `adapted_baseline_knn_k5`, `adapted_ldn_diff_K1_scorek5`",
        "- no selector, no oracle, no fusion, no full7",
        "- no checkpoint selection by test labels",
        "",
        "## Gates",
        "",
        f"- minimum pass: `adapted_ldn fan hmean > {STAGE18_LDN_HMEAN}`",
        f"- strong pass: `adapted_ldn fan hmean >= {STAGE18_BASELINE_HMEAN}`",
        f"- hard fail: `adapted_ldn fan hmean <= {STAGE18_LDN_HMEAN}`",
        f"- representation guardrail: `adapted_baseline fan hmean >= {STAGE18_BASELINE_HMEAN - REPRESENTATION_GUARDRAIL_MAX_DROP}`",
        "",
        "## Verdict",
        "",
        f"- adapted baseline hmean: `{adapted_baseline['hmean']:.6f}`",
        f"- adapted ldn hmean: `{adapted_ldn['hmean']:.6f}`",
        f"- minimum pass: `{verdict['minimum_pass']}`",
        f"- strong pass: `{verdict['strong_pass']}`",
        f"- representation guardrail pass: `{verdict['representation_guardrail_pass']}`",
        "",
        "## Comparison Rows",
        "",
        *[
            f"- {row['candidate']}: hmean={row['hmean']:.6f}, pAUC={row['pAUC']:.6f}, TPR@99n={row['TPR@99n']:.6f}"
            for row in comparison_rows
        ],
    ]


def run_score_adapted_fan(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_path = Path(args.embedding_npz_path)
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing adapted embedding file: {emb_path}")

    arr = np.load(emb_path, allow_pickle=True)
    train_emb = arr["train_embeddings"].astype(np.float64)
    test_emb = arr["test_embeddings"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    test_paths = arr["test_paths"].astype(str)

    train_emb = l2_normalize_np(train_emb, axis=1)
    test_emb = l2_normalize_np(test_emb, axis=1)

    dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref = chord_distance_matrix(train_emb, train_emb)
    rho_k1 = local_density_knn(ref_ref, K=1)

    score_baseline = baseline_knn_scores(dist, score_k=CFG.score_k)
    score_ldn = baseline_knn_scores(dist - rho_k1[None, :], score_k=CFG.score_k)

    scores_root = out_root / "scores"
    (scores_root / "adapted_baseline_knn_k5").mkdir(parents=True, exist_ok=True)
    (scores_root / "adapted_ldn_diff_K1_scorek5").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    tail_rows = []
    geom_rows = []
    candidate_score_map = {
        "adapted_baseline_knn_k5": score_baseline,
        "adapted_ldn_diff_K1_scorek5": score_ldn,
    }
    for candidate, scores in candidate_score_map.items():
        summary, tail, geom = evaluate(scores, y_test, d_test)
        summary_rows.append({"candidate": candidate, **summary})
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

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    comparison_rows = build_candidate_rows(summary_rows)
    write_csv(out_root / "comparison_vs_stage18_fan.csv", comparison_rows)

    adapted_baseline = next(row for row in summary_rows if row["candidate"] == "adapted_baseline_knn_k5")
    adapted_ldn = next(row for row in summary_rows if row["candidate"] == "adapted_ldn_diff_K1_scorek5")
    verdict = {
        "adapted_ldn_hmean": float(adapted_ldn["hmean"]),
        "adapted_ldn_pAUC": float(adapted_ldn["pAUC"]),
        "adapted_ldn_TPR99": float(adapted_ldn["TPR@99n"]),
        "adapted_baseline_hmean": float(adapted_baseline["hmean"]),
        "stage18_ldn_hmean": STAGE18_LDN_HMEAN,
        "stage18_baseline_hmean": STAGE18_BASELINE_HMEAN,
        "stage19c_failed_hmean": STAGE19C_FAILED_HMEAN,
        "delta_vs_stage18_ldn": float(adapted_ldn["hmean"]) - STAGE18_LDN_HMEAN,
        "delta_vs_stage18_baseline": float(adapted_ldn["hmean"]) - STAGE18_BASELINE_HMEAN,
        "delta_vs_stage19c_failed": float(adapted_ldn["hmean"]) - STAGE19C_FAILED_HMEAN,
        "minimum_pass": float(adapted_ldn["hmean"]) > STAGE18_LDN_HMEAN,
        "strong_pass": float(adapted_ldn["hmean"]) >= STAGE18_BASELINE_HMEAN,
        "negative": float(adapted_ldn["hmean"]) <= STAGE18_LDN_HMEAN,
        "representation_guardrail_pass": float(adapted_baseline["hmean"]) >= (STAGE18_BASELINE_HMEAN - REPRESENTATION_GUARDRAIL_MAX_DROP),
        "notes": [
            "fan only",
            "no full7 conclusion",
            "no selector",
            "no checkpoint selection by test labels",
        ],
    }
    write_json(out_root / "stage19f_fan_anchor_lora_eval_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_markdown(verdict, comparison_rows, summary_rows))


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
