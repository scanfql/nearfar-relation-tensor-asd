#!/usr/bin/env python
"""Stage 17B: clean EAT LoRA-light one-machine sanity.

Modes
-----
- check:
  metadata/static inspection only
- inspect_lora_targets:
  local clean EAT load + LoRA target discovery/audit, but no forward
- one_machine_lora_sanity:
  teacher-anchored weak-augmentation LoRA-light training on one machine,
  followed by adapted P1_mean_std extraction and fixed kNN scoring

Protocol boundary
-----------------
Allowed:
- local clean EAT load only
- LoRA-light on attention qkv/proj linear only
- source train-normal only for training and memory
- current-year test split for evaluation
- no fusion

Blocked:
- 7-machine execution
- A1 / V63 fusion
- alpha sweep
- per-machine selection
- full fine-tune
- CE / SupCon
- pseudo anomaly
- Stage17C
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINE_DEFAULT = "gearbox"
CHECKPOINT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage17b_eat_lora_light_one_machine"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE17B_EAT_LORA_LIGHT_ONE_MACHINE_AUDIT.md"

REGISTERED_FROZEN_REFERENCE = {
    "candidate": "FROZEN_P1_mean_std_REGISTERED",
    "machine": MACHINE_DEFAULT,
    "hmean": 0.561801,
    "pAUC": 0.493158,
    "TPR@99n": 0.020000,
}

REGISTERED_PREVIOUS_FUSION_REFERENCE = {
    "candidate": "F_P1_A1_065_REGISTERED",
    "machine": MACHINE_DEFAULT,
    "hmean": 0.572219,
    "pAUC": 0.519474,
    "TPR@99n": 0.030000,
}

CONFIG_FILE_CANDIDATES = [
    "config.json",
    "configuration_eat.py",
    "modeling_eat.py",
    "eat_model.py",
    "model_core.py",
]
MODEL_FILE_CANDIDATES = [
    "model.safetensors",
    "pytorch_model.bin",
]
STATIC_MODEL_FILES = [
    "modeling_eat.py",
    "eat_model.py",
    "model_core.py",
]
STATIC_KEYWORDS = ["qkv", "proj", "attn", "attention", "linear"]
FORBIDDEN_NAME_TOKENS = [
    "mlp",
    "ffn",
    "patch_embed",
    "patch.embed",
    "patchembed",
    "norm",
    "classifier",
    "head",
]


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.05


@dataclass
class Stage17BTrainConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 2
    val_ratio: float = 0.10
    gain_min: float = 0.90
    gain_max: float = 1.10
    noise_snr_db: float = 30.0
    time_shift_min: int = -160
    time_shift_max: int = 160
    anchor0_weight: float = 1.0
    anchor1_weight: float = 1.0
    view_weight: float = 0.2
    rel_weight: float = 0.1
    pooling: str = "P1_mean_std"
    objective_name: str = "teacher_anchor_weak_aug_p1"
    loss_space: str = "normalized_pooled_embedding_1536"


def parse_args():
    p = argparse.ArgumentParser(description="Stage17B clean EAT LoRA-light one-machine sanity")
    p.add_argument(
        "--mode",
        required=True,
        choices=["check", "inspect_lora_targets", "one_machine_lora_sanity"],
    )
    p.add_argument("--eat_model_path", default=CHECKPOINT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
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
    for name in [
        "torch",
        "torchaudio",
        "soundfile",
        "librosa",
        "transformers",
        "timm",
        "safetensors",
        "scipy",
        "sklearn",
    ]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def model_file_presence(model_path: Path) -> dict:
    files = {p.name for p in model_path.iterdir()} if model_path.exists() and model_path.is_dir() else set()
    return {
        "config_files_present": [f for f in CONFIG_FILE_CANDIDATES if f in files],
        "model_files_present": [f for f in MODEL_FILE_CANDIDATES if f in files],
        "static_model_files_present": [f for f in STATIC_MODEL_FILES if f in files],
    }


def static_target_discovery(model_path: Path) -> dict:
    file_reports = []
    qkv_hits = 0
    proj_hits = 0
    attn_hits = 0
    linear_hits = 0

    for name in STATIC_MODEL_FILES:
        file_path = model_path / name
        report = {
            "file": str(file_path),
            "exists": file_path.exists(),
            "matched_lines": [],
        }
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                lower = line.lower()
                if not any(tok in lower for tok in STATIC_KEYWORDS):
                    continue
                compact = line.strip()
                if len(compact) > 220:
                    compact = compact[:217] + "..."
                report["matched_lines"].append({"lineno": lineno, "text": compact})
                if "qkv" in lower:
                    qkv_hits += 1
                if "proj" in lower:
                    proj_hits += 1
                if "attn" in lower or "attention" in lower:
                    attn_hits += 1
                if "linear" in lower:
                    linear_hits += 1
            report["matched_line_count"] = len(report["matched_lines"])
            report["matched_lines"] = report["matched_lines"][:40]
        else:
            report["matched_line_count"] = 0
        file_reports.append(report)

    return {
        "model_path": str(model_path),
        "files_scanned": file_reports,
        "static_qkv_token_hits": qkv_hits,
        "static_proj_token_hits": proj_hits,
        "static_attn_token_hits": attn_hits,
        "static_linear_token_hits": linear_hits,
        "static_discovery_supports_qkv_proj_search": bool(qkv_hits > 0 and proj_hits > 0),
    }


def machine_split_inventory(data_root: Path, machine: str) -> tuple[list[dict], dict]:
    if not data_root.exists():
        return [], {
            "requested_machine": machine,
            "machine_exists": False,
            "source_train_normal_count": 0,
            "test_count": 0,
        }

    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    rows = []
    for item in train_items + test_items:
        rows.append(
            {
                "machine": item.machine,
                "section": item.section,
                "domain": item.domain,
                "split": item.split,
                "label": item.label,
                "path": str(item.path),
            }
        )
    return rows, {
        "requested_machine": machine,
        "machine_exists": len(items) > 0,
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "train_files_exist": len(train_items) > 0,
        "test_files_exist": len(test_items) > 0,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad_(False)


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


def is_attention_qkv_or_proj_linear(module_name: str, module) -> tuple[bool, str]:
    import torch.nn as nn

    if not isinstance(module, nn.Linear):
        return False, "non_linear"

    name_lower = module_name.lower()
    leaf = module_name.split(".")[-1].lower()
    has_attention_token = ".attn." in name_lower or "attention" in name_lower or "attn" in name_lower

    if any(tok in name_lower for tok in FORBIDDEN_NAME_TOKENS):
        return False, "skip_forbidden_block"
    if not has_attention_token:
        return False, "skip_non_attention_linear"
    if leaf == "qkv":
        return True, "selected_attention_qkv"
    if leaf == "proj":
        return True, "selected_attention_proj"
    return False, "skip_attention_non_qkv_proj_linear"


def estimate_lora_params(in_features: int, out_features: int, rank: int) -> int:
    return int(rank * (in_features + out_features))


def inspect_linear_modules(model, lora_cfg: LoRAConfig) -> tuple[list[dict], list[str]]:
    import torch.nn as nn

    rows = []
    selected_names = []
    for module_name, module in model.named_modules():
        if not module_name or not isinstance(module, nn.Linear):
            continue
        selected, reason = is_attention_qkv_or_proj_linear(module_name, module)
        if selected:
            selected_names.append(module_name)
        rows.append(
            {
                "module_name": module_name,
                "module_type": module.__class__.__name__,
                "selected": selected,
                "selection_reason": reason,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "has_bias": module.bias is not None,
                "base_param_numel": int(sum(p.numel() for p in module.parameters())),
                "estimated_lora_param_numel": estimate_lora_params(
                    int(module.in_features),
                    int(module.out_features),
                    lora_cfg.rank,
                ),
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


def is_lora_wrapper(module) -> bool:
    return bool(getattr(module, "_is_lora_wrapper", False))


def is_lora_parameter_name(name: str) -> bool:
    return name.endswith("lora_down.weight") or name.endswith("lora_up.weight")


def get_parent_module(root_module, module_name: str):
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_modules(model, selected_names: list[str], lora_cfg: LoRAConfig):
    freeze_all_parameters(model)
    for module_name in selected_names:
        parent, leaf = get_parent_module(model, module_name)
        base_linear = getattr(parent, leaf)
        wrapped = build_lora_linear(base_linear, lora_cfg)
        wrapped = wrapped.to(device=base_linear.weight.device, dtype=base_linear.weight.dtype)
        setattr(parent, leaf, wrapped)
    return model


def parameter_audit_rows(model) -> list[dict]:
    rows = []
    for name, param in model.named_parameters():
        if is_lora_parameter_name(name):
            category = "lora_trainable" if param.requires_grad else "unexpected_frozen_lora"
        elif param.requires_grad:
            category = "unexpected_trainable_base"
        else:
            category = "frozen_base"
        rows.append(
            {
                "parameter_name": name,
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
                "category": category,
            }
        )
    return rows


def summarize_parameter_audit(rows: list[dict]) -> dict:
    total = sum(int(r["numel"]) for r in rows)
    frozen = sum(int(r["numel"]) for r in rows if not r["requires_grad"])
    trainable = sum(int(r["numel"]) for r in rows if r["requires_grad"])
    lora_trainable = sum(int(r["numel"]) for r in rows if r["category"] == "lora_trainable")
    unexpected = [r["parameter_name"] for r in rows if r["category"] == "unexpected_trainable_base"]
    return {
        "total_parameter_count": int(total),
        "frozen_parameter_count": int(frozen),
        "trainable_parameter_count": int(trainable),
        "lora_trainable_parameter_count": int(lora_trainable),
        "unexpected_trainable_base_parameter_names": unexpected,
        "no_unexpected_trainable_base_parameters": len(unexpected) == 0,
        "trainable_fraction": float(trainable / total) if total > 0 else 0.0,
    }


def gradient_audit_rows(model, epoch: int, batch_index: int) -> list[dict]:
    rows = []
    for name, param in model.named_parameters():
        grad = param.grad
        grad_present = grad is not None
        grad_norm = float(grad.norm().item()) if grad_present else 0.0
        grad_abs_sum = float(grad.abs().sum().item()) if grad_present else 0.0
        if is_lora_parameter_name(name):
            category = "lora_grad_only" if grad_present else "lora_missing_grad"
        else:
            category = "unexpected_base_grad" if grad_present else "frozen_no_grad"
        rows.append(
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "parameter_name": name,
                "requires_grad": bool(param.requires_grad),
                "grad_present": bool(grad_present),
                "grad_norm": grad_norm,
                "grad_abs_sum": grad_abs_sum,
                "category": category,
            }
        )
    return rows


def set_student_mode_for_training(student):
    student.eval()
    for module in student.modules():
        if is_lora_wrapper(module):
            module.train()


def split_train_val_items(source_train_items: list, val_ratio: float, seed: int) -> tuple[list, list]:
    if not source_train_items:
        return [], []
    total = len(source_train_items)
    val_count = max(1, int(total * val_ratio))
    val_count = min(val_count, total - 1) if total > 1 else 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    val_idx = set(int(x) for x in perm[:val_count])
    train_items = [item for idx, item in enumerate(source_train_items) if idx not in val_idx]
    val_items = [item for idx, item in enumerate(source_train_items) if idx in val_idx]
    return train_items, val_items


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


def zero_padded_shift(wav: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return wav.astype(np.float32, copy=True)
    out = np.zeros_like(wav, dtype=np.float32)
    if shift > 0:
        out[shift:] = wav[:-shift]
        return out
    n = -shift
    out[:-n] = wav[n:]
    return out


def weak_augment_waveform(wav: np.ndarray, cfg: Stage17BTrainConfig, rng: np.random.Generator) -> np.ndarray:
    out = wav.astype(np.float32, copy=True)
    gain = float(rng.uniform(cfg.gain_min, cfg.gain_max))
    out *= gain

    signal_rms = float(np.sqrt(np.mean(np.square(out.astype(np.float64))) + 1e-12))
    noise_rms = signal_rms / (10.0 ** (cfg.noise_snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=out.shape).astype(np.float32)
    out = out + noise

    shift = int(rng.integers(cfg.time_shift_min, cfg.time_shift_max + 1))
    out = zero_padded_shift(out, shift)
    return out.astype(np.float32)


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
    norm_mean = -4.268
    norm_std = 4.569
    mel = (mel - norm_mean) / (norm_std * 2)
    return mel.to(dtype=torch.float32)


def build_original_input_batch(paths: list[Path], device):
    import torch

    feats = [waveform_to_eat_tensor(load_waveform_16k(path)) for path in paths]
    return torch.stack(feats, dim=0).unsqueeze(1).to(device)


def build_paired_input_batches(
    paths: list[Path],
    device,
    cfg: Stage17BTrainConfig,
    rng: np.random.Generator,
):
    import torch

    originals = []
    augmented = []
    for path in paths:
        wav = load_waveform_16k(path)
        originals.append(waveform_to_eat_tensor(wav))
        augmented.append(waveform_to_eat_tensor(weak_augment_waveform(wav, cfg, rng)))
    x0 = torch.stack(originals, dim=0).unsqueeze(1).to(device)
    x1 = torch.stack(augmented, dim=0).unsqueeze(1).to(device)
    return x0, x1


def extract_frame_output(model, x):
    import torch

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


def pool_p1_mean_std(frames):
    import torch

    mean = frames.mean(dim=1)
    std = frames.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=1)


def compute_stage17b_loss(t0, s0, s1, cfg: Stage17BTrainConfig):
    import torch
    import torch.nn.functional as F

    l_anchor0 = (1.0 - F.cosine_similarity(s0, t0, dim=1)).mean()
    l_anchor1 = (1.0 - F.cosine_similarity(s1, t0, dim=1)).mean()
    l_view = (1.0 - F.cosine_similarity(s0, s1, dim=1)).mean()

    if t0.shape[0] > 1:
        sim_t = t0 @ t0.T
        sim_s = s0 @ s0.T
        mask = ~torch.eye(t0.shape[0], dtype=torch.bool, device=t0.device)
        l_rel = F.mse_loss(sim_s[mask], sim_t[mask])
    else:
        l_rel = torch.zeros([], device=t0.device)

    total = (
        cfg.anchor0_weight * l_anchor0
        + cfg.anchor1_weight * l_anchor1
        + cfg.view_weight * l_view
        + cfg.rel_weight * l_rel
    )
    return {
        "total": total,
        "anchor0": l_anchor0,
        "anchor1": l_anchor1,
        "view": l_view,
        "rel": l_rel,
    }


def batch_loss_pass(
    teacher,
    student,
    batch_paths: list[Path],
    device,
    cfg: Stage17BTrainConfig,
    rng: np.random.Generator,
):
    import torch
    import torch.nn.functional as F

    x0, x1 = build_paired_input_batches(batch_paths, device, cfg, rng)
    with torch.no_grad():
        t0 = F.normalize(pool_p1_mean_std(extract_frame_output(teacher, x0)), dim=1)
    s0 = F.normalize(pool_p1_mean_std(extract_frame_output(student, x0)), dim=1)
    s1 = F.normalize(pool_p1_mean_std(extract_frame_output(student, x1)), dim=1)
    return compute_stage17b_loss(t0, s0, s1, cfg)


def collect_epoch_rows(log_rows: list[dict], epoch: int, train_stats: dict, val_stats: dict, lr: float):
    log_rows.append(
        {
            "epoch": epoch,
            "train_total_loss": train_stats["total"],
            "train_anchor0_loss": train_stats["anchor0"],
            "train_anchor1_loss": train_stats["anchor1"],
            "train_view_loss": train_stats["view"],
            "train_rel_loss": train_stats["rel"],
            "val_total_loss": val_stats["total"],
            "val_anchor0_loss": val_stats["anchor0"],
            "val_anchor1_loss": val_stats["anchor1"],
            "val_view_loss": val_stats["view"],
            "val_rel_loss": val_stats["rel"],
            "lr": lr,
        }
    )


def mean_stats(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def extract_p1_embeddings(model, items: list, device, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            paths = [Path(item.path) for item in batch_items]
            x = build_original_input_batch(paths, device)
            pooled = pool_p1_mean_std(extract_frame_output(model, x))
            out.append(pooled.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1536), dtype=np.float64)


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5) -> np.ndarray:
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, train_n.shape[0])
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


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
        "score_compression_gap": float(anomaly_median - q99),
        "score_compression_flag": bool(anomaly_median <= q99) if len(anomalies) and len(normals) else False,
    }


def evaluate_candidate(candidate: str, machine: str, scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    summary = {
        "candidate": candidate,
        "machine": machine,
        "hmean": float(met["hmean"]),
        "pAUC": float(met["pauc"]),
        "KS": float(geom["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "anomaly_median": float(tail["anomaly_median"]),
        "anomaly_q75": float(tail["anomaly_q75"]),
        "score_compression_gap": float(tail["score_compression_gap"]),
        "score_compression_flag": bool(tail["score_compression_flag"]),
    }
    tail_row = {"candidate": candidate, "machine": machine, **tail}
    geom_row = {"candidate": candidate, "machine": machine, **geom}
    return summary, tail_row, geom_row


def build_audit_markdown(result: dict) -> list[str]:
    adapted = result["adapted_summary"]
    runtime_frozen = result["runtime_frozen_summary"]
    compare = result["comparison"]
    return [
        "# CTAF Stage17B EAT LoRA-Light One-Machine Audit",
        "",
        "## Protocol",
        "",
        "- machine: `gearbox`",
        "- backbone: clean `EAT` local checkpoint only",
        "- LoRA target modules: `model.blocks.*.attn.qkv` and `model.blocks.*.attn.proj` only",
        "- pooling: `P1_mean_std`",
        "- objective: teacher-anchored weak-augmentation consistency on normalized pooled embeddings",
        "- no fusion",
        "",
        "## Split",
        "",
        f"- source train-normal total: `{result['split']['source_train_normal_total']}`",
        f"- train split: `{result['split']['train_count']}`",
        f"- val split: `{result['split']['val_count']}`",
        f"- test split: `{result['split']['test_count']}`",
        "",
        "## LoRA Audit",
        "",
        f"- selected target module count: `{result['selected_target_module_count']}`",
        f"- trainable parameter count: `{result['parameter_audit']['trainable_parameter_count']}`",
        f"- frozen parameter count: `{result['parameter_audit']['frozen_parameter_count']}`",
        f"- no unexpected trainable base parameters: `{result['parameter_audit']['no_unexpected_trainable_base_parameters']}`",
        "",
        "## Training",
        "",
        f"- best epoch: `{result['training']['best_epoch']}`",
        f"- best validation loss: `{result['training']['best_val_loss']:.6f}`",
        f"- checkpoint: `{result['training']['best_checkpoint']}`",
        "",
        "## Metrics",
        "",
        f"- adapted LoRA P1 hmean: `{adapted['hmean']:.6f}`",
        f"- adapted LoRA P1 pAUC: `{adapted['pAUC']:.6f}`",
        f"- adapted LoRA P1 TPR@99n: `{adapted['TPR@99n']:.6f}`",
        f"- runtime frozen P1 hmean: `{runtime_frozen['hmean']:.6f}`",
        f"- runtime frozen P1 pAUC: `{runtime_frozen['pAUC']:.6f}`",
        f"- runtime frozen P1 TPR@99n: `{runtime_frozen['TPR@99n']:.6f}`",
        f"- registered frozen P1 hmean: `{REGISTERED_FROZEN_REFERENCE['hmean']:.6f}`",
        f"- registered frozen P1 pAUC: `{REGISTERED_FROZEN_REFERENCE['pAUC']:.6f}`",
        f"- registered frozen P1 TPR@99n: `{REGISTERED_FROZEN_REFERENCE['TPR@99n']:.6f}`",
        f"- registered F_P1_A1_065 hmean: `{REGISTERED_PREVIOUS_FUSION_REFERENCE['hmean']:.6f}`",
        f"- registered F_P1_A1_065 pAUC: `{REGISTERED_PREVIOUS_FUSION_REFERENCE['pAUC']:.6f}`",
        f"- registered F_P1_A1_065 TPR@99n: `{REGISTERED_PREVIOUS_FUSION_REFERENCE['TPR@99n']:.6f}`",
        "",
        "## Comparison",
        "",
        f"- adapted vs registered frozen hmean delta: `{compare['delta_hmean_vs_registered_frozen']:.6f}`",
        f"- adapted vs registered frozen pAUC delta: `{compare['delta_pAUC_vs_registered_frozen']:.6f}`",
        f"- adapted vs registered frozen TPR@99n delta: `{compare['delta_TPR99_vs_registered_frozen']:.6f}`",
        f"- adapted score compression flag: `{adapted['score_compression_flag']}`",
        "",
        "## Guardrails",
        "",
        "- no A1/V63 fusion was executed in this mode",
        "- no 7-machine run was executed in this mode",
        "- no per-machine selection was executed in this mode",
    ]


def build_check_markdown(result: dict) -> list[str]:
    imports_failed = [k for k, ok in result["imports"].items() if not ok]
    return [
        "# CTAF Stage17B EAT LoRA-Light One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no training",
        "- no extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- checkpoint path exists: `{result['eat_model_path_exists']}`",
        f"- data root exists: `{result['data_root_exists']}`",
        f"- machine exists: `{result['machine_summary']['machine_exists']}`",
        f"- source train-normal count: `{result['machine_summary']['source_train_normal_count']}`",
        f"- test count: `{result['machine_summary']['test_count']}`",
        f"- static qkv/proj discovery supported: `{result['static_target_discovery']['static_discovery_supports_qkv_proj_search']}`",
        f"- failed imports: `{', '.join(imports_failed) if imports_failed else 'none'}`",
    ]


def build_inspect_markdown(summary: dict) -> list[str]:
    selected_preview = summary["selected_target_module_names"][:20]
    return [
        "# CTAF Stage17B EAT LoRA-Light One-Machine Audit",
        "",
        "## Mode",
        "",
        "- `inspect_lora_targets`",
        "- local clean EAT load only",
        "- no forward",
        "- no training",
        "",
        "## LoRA Target Summary",
        "",
        f"- checkpoint path: `{summary['eat_model_path']}`",
        f"- model class: `{summary['model_class']}`",
        f"- resolved device: `{summary['resolved_device']}`",
        f"- candidate linear module count: `{summary['candidate_linear_module_count']}`",
        f"- selected qkv/proj module count: `{summary['selected_target_module_count']}`",
        f"- selected qkv/proj preview: `{selected_preview}`",
        f"- trainable parameter count after LoRA injection: `{summary['parameter_audit']['trainable_parameter_count']}`",
        f"- frozen parameter count after LoRA injection: `{summary['parameter_audit']['frozen_parameter_count']}`",
        f"- no unexpected trainable base parameters: `{summary['parameter_audit']['no_unexpected_trainable_base_parameters']}`",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    model_files = model_file_presence(model_path)
    static_discovery = static_target_discovery(model_path)
    split_rows, machine_summary = machine_split_inventory(data_root, args.machine)

    result = {
        "mode": "check",
        "eat_model_path": str(model_path),
        "data_root": str(data_root),
        "eat_model_path_exists": model_path.exists(),
        "data_root_exists": data_root.exists(),
        "machine_summary": machine_summary,
        "model_files": model_files,
        "static_target_discovery": static_discovery,
        "imports": import_status(),
        "fixed_stage17b_decisions": {
            "adaptation_family": "LoRA-light",
            "machine": args.machine,
            "target_policy": "attention qkv/proj linear only",
            "pooling": "P1_mean_std",
            "training_data": "current-year source train-normal only",
            "memory": "source train-normal only",
            "test_split": "current-year test split",
            "scoring": f"cosine kNN k={args.latent_knn_k}",
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
        },
    }

    write_json(out_root / "check_status.json", result)
    write_json(out_root / "static_target_discovery.json", static_discovery)
    write_csv(out_root / f"{args.machine}_split_inventory.csv", split_rows)
    write_markdown(Path(args.doc_path), build_check_markdown(result))


def run_inspect_lora_targets(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.eat_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")

    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model, resolved_device = load_clean_eat_local(model_path, args.device)
    inventory_rows, selected_names = inspect_linear_modules(model, lora_cfg)
    if not selected_names:
        raise RuntimeError(
            "No attention qkv/proj Linear modules were selected. "
            "Inspect target_module_inventory.csv and freeze the exact target names before proceeding."
        )

    write_csv(out_root / "target_module_inventory.csv", inventory_rows)

    model = inject_lora_modules(model, selected_names, lora_cfg)
    audit_rows = parameter_audit_rows(model)
    audit_summary = summarize_parameter_audit(audit_rows)
    write_csv(out_root / "parameter_audit.csv", audit_rows)

    selected_base_params = sum(int(row["base_param_numel"]) for row in inventory_rows if row["selected"])
    selected_lora_params_estimate = sum(
        int(row["estimated_lora_param_numel"]) for row in inventory_rows if row["selected"]
    )

    summary = {
        "mode": "inspect_lora_targets",
        "eat_model_path": str(model_path),
        "resolved_device": str(resolved_device),
        "model_class": model.__class__.__name__,
        "machine": args.machine,
        "candidate_linear_module_count": len(inventory_rows),
        "candidate_linear_module_names": [row["module_name"] for row in inventory_rows],
        "selected_target_module_count": len(selected_names),
        "selected_target_module_names": selected_names,
        "selected_base_parameter_count": int(selected_base_params),
        "selected_lora_parameter_count_estimate": int(selected_lora_params_estimate),
        "lora_config": asdict(lora_cfg),
        "parameter_audit": audit_summary,
        "blocked_actions": [
            "no forward",
            "no training",
            "no embedding extraction",
            "no scoring",
        ],
    }
    write_json(out_root / "inspect_lora_targets.json", summary)
    write_markdown(Path(args.doc_path), build_inspect_markdown(summary))


def run_one_machine_lora_sanity(args):
    import torch

    set_seed(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (out_root / "scores").mkdir(parents=True, exist_ok=True)

    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    train_cfg = Stage17BTrainConfig()

    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data_root: {data_root}")

    all_items = filter_items(scan_dataset(data_root), args.machine)
    source_train_items, test_items = split_train_test(all_items, use_target_train=False)
    if not source_train_items or not test_items:
        raise RuntimeError(
            f"Machine {args.machine}: source_train_normal={len(source_train_items)} test={len(test_items)}"
        )
    train_items, val_items = split_train_val_items(source_train_items, train_cfg.val_ratio, args.seed)
    if not train_items or not val_items:
        raise RuntimeError(
            f"Train/val split invalid for {args.machine}: train={len(train_items)} val={len(val_items)}"
        )

    teacher, resolved_device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(teacher)
    teacher.eval()

    student, _ = load_clean_eat_local(model_path, str(resolved_device))
    inventory_rows, selected_names = inspect_linear_modules(student, lora_cfg)
    if not selected_names:
        raise RuntimeError("No LoRA target modules were selected for one_machine_lora_sanity.")
    write_csv(out_root / "target_module_inventory.csv", inventory_rows)

    student = inject_lora_modules(student, selected_names, lora_cfg)
    parameter_rows = parameter_audit_rows(student)
    parameter_summary = summarize_parameter_audit(parameter_rows)
    if not parameter_summary["no_unexpected_trainable_base_parameters"]:
        raise RuntimeError(
            "Unexpected trainable base parameters detected before training: "
            f"{parameter_summary['unexpected_trainable_base_parameter_names']}"
        )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_ckpt_path = out_root / "checkpoints" / f"{args.machine}_lora_light_best.pt"
    log_rows = []
    gradient_rows: list[dict] | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg.epochs + 1):
        set_student_mode_for_training(student)
        train_rng = np.random.default_rng(args.seed + epoch * 1000)
        perm = train_rng.permutation(len(train_items))
        train_batch_rows = []

        for batch_index, start in enumerate(range(0, len(perm), train_cfg.batch_size), start=1):
            batch_ids = perm[start:start + train_cfg.batch_size]
            batch_paths = [Path(train_items[int(idx)].path) for idx in batch_ids]

            loss_dict = batch_loss_pass(teacher, student, batch_paths, resolved_device, train_cfg, train_rng)
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            if gradient_rows is None:
                gradient_rows = gradient_audit_rows(student, epoch=epoch, batch_index=batch_index)
            torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
            optimizer.step()

            train_batch_rows.append(
                {
                    "total": float(loss_dict["total"].item()),
                    "anchor0": float(loss_dict["anchor0"].item()),
                    "anchor1": float(loss_dict["anchor1"].item()),
                    "view": float(loss_dict["view"].item()),
                    "rel": float(loss_dict["rel"].item()),
                }
            )

        student.eval()
        val_rng = np.random.default_rng(args.seed + 500000 + epoch)
        val_batch_rows = []
        with torch.no_grad():
            for start in range(0, len(val_items), train_cfg.batch_size):
                batch_items = val_items[start:start + train_cfg.batch_size]
                batch_paths = [Path(item.path) for item in batch_items]
                loss_dict = batch_loss_pass(teacher, student, batch_paths, resolved_device, train_cfg, val_rng)
                val_batch_rows.append(
                    {
                        "total": float(loss_dict["total"].item()),
                        "anchor0": float(loss_dict["anchor0"].item()),
                        "anchor1": float(loss_dict["anchor1"].item()),
                        "view": float(loss_dict["view"].item()),
                        "rel": float(loss_dict["rel"].item()),
                    }
                )

        train_stats = {
            "total": mean_stats(train_batch_rows, "total"),
            "anchor0": mean_stats(train_batch_rows, "anchor0"),
            "anchor1": mean_stats(train_batch_rows, "anchor1"),
            "view": mean_stats(train_batch_rows, "view"),
            "rel": mean_stats(train_batch_rows, "rel"),
        }
        val_stats = {
            "total": mean_stats(val_batch_rows, "total"),
            "anchor0": mean_stats(val_batch_rows, "anchor0"),
            "anchor1": mean_stats(val_batch_rows, "anchor1"),
            "view": mean_stats(val_batch_rows, "view"),
            "rel": mean_stats(val_batch_rows, "rel"),
        }
        collect_epoch_rows(log_rows, epoch, train_stats, val_stats, optimizer.param_groups[0]["lr"])

        if val_stats["total"] < best_val_loss - 1e-7:
            best_val_loss = val_stats["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "student_state_dict": student.state_dict(),
                    "machine": args.machine,
                    "seed": int(args.seed),
                    "eat_model_path": str(model_path),
                    "selected_target_module_names": selected_names,
                    "lora_config": asdict(lora_cfg),
                    "train_config": asdict(train_cfg),
                    "parameter_audit": parameter_summary,
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val_loss),
                    "train_count": int(len(train_items)),
                    "val_count": int(len(val_items)),
                    "memory_count": int(len(source_train_items)),
                    "test_count": int(len(test_items)),
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.early_stop_patience:
                break

    if gradient_rows is None:
        raise RuntimeError("No training batches were executed; gradient audit could not be captured.")
    unexpected_base_grad_names = [
        row["parameter_name"] for row in gradient_rows if row["category"] == "unexpected_base_grad"
    ]
    if unexpected_base_grad_names:
        raise RuntimeError(
            "Unexpected base-parameter gradients detected: "
            f"{unexpected_base_grad_names}"
        )
    if best_epoch < 0 or not best_ckpt_path.exists():
        raise RuntimeError("Best checkpoint was not created during one_machine_lora_sanity.")

    write_csv(out_root / "train_log.csv", log_rows)
    write_csv(out_root / "lora_gradient_audit.csv", gradient_rows)

    best_blob = torch.load(str(best_ckpt_path), map_location=resolved_device)
    eval_student, _ = load_clean_eat_local(model_path, str(resolved_device))
    eval_student = inject_lora_modules(eval_student, selected_names, lora_cfg)
    eval_student.load_state_dict(best_blob["student_state_dict"], strict=True)
    eval_student.eval()

    parameter_rows_after = parameter_audit_rows(eval_student)
    parameter_summary_after = summarize_parameter_audit(parameter_rows_after)
    if not parameter_summary_after["no_unexpected_trainable_base_parameters"]:
        raise RuntimeError(
            "Unexpected trainable base parameters detected after reload: "
            f"{parameter_summary_after['unexpected_trainable_base_parameter_names']}"
        )
    write_csv(out_root / "parameter_audit_after_training.csv", parameter_rows_after)

    train_embeddings = extract_p1_embeddings(eval_student, source_train_items, resolved_device, train_cfg.batch_size)
    test_embeddings = extract_p1_embeddings(eval_student, test_items, resolved_device, train_cfg.batch_size)
    y_test = np.asarray([int(item.label) for item in test_items], dtype=np.int64)
    d_test = np.asarray([item.domain for item in test_items], dtype=str)
    train_paths = np.asarray([str(item.path) for item in source_train_items], dtype=object)
    test_paths = np.asarray([str(item.path) for item in test_items], dtype=object)

    adapted_scores = knn_cosine_scores(train_embeddings, test_embeddings, k=args.latent_knn_k)
    np.savez_compressed(
        out_root / "embeddings" / f"{args.machine}_lora_p1_embeddings.npz",
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        y_test=y_test,
        d_test=d_test.astype("U16"),
        train_paths=train_paths,
        test_paths=test_paths,
        checkpoint_path=str(best_ckpt_path),
        eat_model_path=str(model_path),
        pooling="P1_mean_std",
        machine=args.machine,
    )
    np.savez(
        out_root / "scores" / f"{args.machine}_lora_p1_scores.npz",
        test_scores=adapted_scores,
        y_test=y_test,
        d_test=d_test.astype("U16"),
        test_paths=test_paths,
        score_mode="lora_p1_knn",
        latent_knn_k=int(args.latent_knn_k),
        checkpoint_path=str(best_ckpt_path),
        eat_model_path=str(model_path),
    )

    frozen_train_embeddings = extract_p1_embeddings(teacher, source_train_items, resolved_device, train_cfg.batch_size)
    frozen_test_embeddings = extract_p1_embeddings(teacher, test_items, resolved_device, train_cfg.batch_size)
    runtime_frozen_scores = knn_cosine_scores(frozen_train_embeddings, frozen_test_embeddings, k=args.latent_knn_k)

    adapted_summary, adapted_tail, adapted_geom = evaluate_candidate(
        "LORA_P1_mean_std", args.machine, adapted_scores, y_test, d_test
    )
    runtime_frozen_summary, runtime_frozen_tail, runtime_frozen_geom = evaluate_candidate(
        "FROZEN_P1_mean_std_RUNTIME", args.machine, runtime_frozen_scores, y_test, d_test
    )

    summary_rows = [adapted_summary, runtime_frozen_summary]
    tail_rows = [adapted_tail, runtime_frozen_tail]
    geom_rows = [adapted_geom, runtime_frozen_geom]
    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    comparison_row = {
        "machine": args.machine,
        "adapted_candidate": adapted_summary["candidate"],
        "runtime_frozen_candidate": runtime_frozen_summary["candidate"],
        "adapted_hmean": adapted_summary["hmean"],
        "adapted_pAUC": adapted_summary["pAUC"],
        "adapted_TPR@99n": adapted_summary["TPR@99n"],
        "runtime_frozen_hmean": runtime_frozen_summary["hmean"],
        "runtime_frozen_pAUC": runtime_frozen_summary["pAUC"],
        "runtime_frozen_TPR@99n": runtime_frozen_summary["TPR@99n"],
        "registered_frozen_hmean": REGISTERED_FROZEN_REFERENCE["hmean"],
        "registered_frozen_pAUC": REGISTERED_FROZEN_REFERENCE["pAUC"],
        "registered_frozen_TPR@99n": REGISTERED_FROZEN_REFERENCE["TPR@99n"],
        "registered_fusion_hmean": REGISTERED_PREVIOUS_FUSION_REFERENCE["hmean"],
        "registered_fusion_pAUC": REGISTERED_PREVIOUS_FUSION_REFERENCE["pAUC"],
        "registered_fusion_TPR@99n": REGISTERED_PREVIOUS_FUSION_REFERENCE["TPR@99n"],
        "delta_hmean_vs_runtime_frozen": adapted_summary["hmean"] - runtime_frozen_summary["hmean"],
        "delta_pAUC_vs_runtime_frozen": adapted_summary["pAUC"] - runtime_frozen_summary["pAUC"],
        "delta_TPR99_vs_runtime_frozen": adapted_summary["TPR@99n"] - runtime_frozen_summary["TPR@99n"],
        "delta_hmean_vs_registered_frozen": adapted_summary["hmean"] - REGISTERED_FROZEN_REFERENCE["hmean"],
        "delta_pAUC_vs_registered_frozen": adapted_summary["pAUC"] - REGISTERED_FROZEN_REFERENCE["pAUC"],
        "delta_TPR99_vs_registered_frozen": adapted_summary["TPR@99n"] - REGISTERED_FROZEN_REFERENCE["TPR@99n"],
        "delta_hmean_vs_registered_fusion": adapted_summary["hmean"] - REGISTERED_PREVIOUS_FUSION_REFERENCE["hmean"],
        "delta_pAUC_vs_registered_fusion": adapted_summary["pAUC"] - REGISTERED_PREVIOUS_FUSION_REFERENCE["pAUC"],
        "delta_TPR99_vs_registered_fusion": adapted_summary["TPR@99n"] - REGISTERED_PREVIOUS_FUSION_REFERENCE["TPR@99n"],
        "adapted_score_compression_flag": adapted_summary["score_compression_flag"],
        "no_unexpected_trainable_base_parameters": parameter_summary_after["no_unexpected_trainable_base_parameters"],
    }
    write_csv(out_root / "frozen_reference_comparison.csv", [comparison_row])

    result = {
        "mode": "one_machine_lora_sanity",
        "machine": args.machine,
        "eat_model_path": str(model_path),
        "resolved_device": str(resolved_device),
        "split": {
            "source_train_normal_total": int(len(source_train_items)),
            "train_count": int(len(train_items)),
            "val_count": int(len(val_items)),
            "test_count": int(len(test_items)),
        },
        "selected_target_module_count": int(len(selected_names)),
        "selected_target_module_names": selected_names,
        "lora_config": asdict(lora_cfg),
        "train_config": asdict(train_cfg),
        "parameter_audit": parameter_summary_after,
        "training": {
            "best_epoch": int(best_blob["best_epoch"]),
            "best_val_loss": float(best_blob["best_val_loss"]),
            "best_checkpoint": str(best_ckpt_path),
        },
        "adapted_summary": adapted_summary,
        "runtime_frozen_summary": runtime_frozen_summary,
        "comparison": comparison_row,
        "output_files": {
            "train_log": str(out_root / "train_log.csv"),
            "gradient_audit": str(out_root / "lora_gradient_audit.csv"),
            "parameter_audit_after_training": str(out_root / "parameter_audit_after_training.csv"),
            "checkpoint": str(best_ckpt_path),
            "embedding_npz": str(out_root / "embeddings" / f"{args.machine}_lora_p1_embeddings.npz"),
            "score_npz": str(out_root / "scores" / f"{args.machine}_lora_p1_scores.npz"),
            "summary_csv": str(out_root / "summary.csv"),
            "tail_table_csv": str(out_root / "tail_table.csv"),
            "score_geometry_csv": str(out_root / "score_geometry.csv"),
            "frozen_reference_comparison_csv": str(out_root / "frozen_reference_comparison.csv"),
        },
    }
    write_json(out_root / "one_machine_lora_sanity_result.json", result)
    write_markdown(Path(args.doc_path), build_audit_markdown(result))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_lora_targets":
        run_inspect_lora_targets(args)
        return
    if args.mode == "one_machine_lora_sanity":
        run_one_machine_lora_sanity(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
