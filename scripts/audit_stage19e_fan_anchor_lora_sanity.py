#!/usr/bin/env python
"""Stage 19E: fan anchor-regularized LoRA adaptation sanity.

Planning/training scaffold only.

Allowed:
- clean EAT load
- fan source train-normal split
- LoRA on last-block attention qkv/proj only
- masked reconstruction + anchor embedding preservation

Blocked:
- no ASD scoring
- no test-label checkpoint selection
- no anomaly-label training
- no full7
- no selector
- no fusion
- no Stage19B checkpoint continuation
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
except Exception:  # pragma: no cover - check mode should still report import status
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test


CHECKPOINT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE19D_ROOT_DEFAULT = "results/stage19d_anchor_regularized_adaptation_design"
OUTPUT_ROOT_DEFAULT = "results/stage19e_fan_anchor_lora_sanity"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19E_FAN_ANCHOR_LORA_SANITY_AUDIT.md"
MACHINE_DEFAULT = "fan"

STAGE19D_DECISION_DEFAULT = "results/stage19d_anchor_regularized_adaptation_design/stage19d_design_decision.json"
STAGE19D_PROTOCOL_DEFAULT = "results/stage19d_anchor_regularized_adaptation_design/stage19e_recommended_protocol.json"


@dataclass
class Stage19ETrainConfig:
    trainable_scope: str = "last_block_attention_lora"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.05
    mask_ratio: float = 0.30
    lambda_anchor: float = 1.0
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 2
    val_ratio: float = 0.10
    seed: int = 13711
    norm_mean: float = -4.268
    norm_std: float = 4.569
    target_length: int = 1024
    num_mel_bins: int = 128
    sample_rate: int = 16000
    frame_shift_ms: int = 10
    hidden_size: int = 768
    recon_dim: int = 128


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="Stage19E fan anchor-regularized LoRA sanity")
    p.add_argument("--mode", required=True, choices=["check", "inspect_lora", "train_fan_anchor_lora"])
    p.add_argument("--eat_model_path", default=CHECKPOINT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage19d_root", default=STAGE19D_ROOT_DEFAULT)
    p.add_argument("--stage19d_decision", default=STAGE19D_DECISION_DEFAULT)
    p.add_argument("--stage19d_protocol", default=STAGE19D_PROTOCOL_DEFAULT)
    p.add_argument("--trainable_scope", default="last_block_attention_lora")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--mask_ratio", type=float, default=0.30)
    p.add_argument("--lambda_anchor", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=2)
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
    status = {"json": True, "csv": True, "pathlib": True}
    try:
        import numpy  # noqa: F401

        status["numpy"] = True
    except Exception:
        status["numpy"] = False
    return status


def set_seed(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


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


def split_train_val(items: list, val_ratio: float, seed: int) -> tuple[list, list]:
    if not items:
        return [], []
    total = len(items)
    val_count = max(1, int(total * val_ratio))
    val_count = min(val_count, total - 1) if total > 1 else 1
    if np is not None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(total)
        val_idx = set(int(x) for x in perm[:val_count])
    else:
        rng = random.Random(seed)
        perm = list(range(total))
        rng.shuffle(perm)
        val_idx = set(int(x) for x in perm[:val_count])
    train_items = [item for idx, item in enumerate(items) if idx not in val_idx]
    val_items = [item for idx, item in enumerate(items) if idx in val_idx]
    return train_items, val_items


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
        return model.model.blocks, "model.blocks"
    if hasattr(model, "blocks"):
        return model.blocks, "blocks"
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

    blocks, _block_prefix = get_block_container(model)
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


def parameter_audit_rows(model, recon_head) -> list[dict]:
    rows = []
    for scope, module in [("eat", model), ("recon_head", recon_head)]:
        for name, param in module.named_parameters():
            full_name = f"{scope}.{name}"
            if name.startswith("recon_head."):
                category = "unexpected_nested_recon_head"
            elif full_name.startswith("recon_head."):
                category = "recon_head_trainable" if param.requires_grad else "frozen_recon_head"
            elif is_lora_parameter_name(name):
                category = "lora_trainable" if param.requires_grad else "unexpected_frozen_lora"
            elif param.requires_grad:
                category = "unexpected_trainable_base"
            else:
                category = "frozen_base"
            if full_name.startswith("recon_head."):
                category = "recon_head_trainable" if param.requires_grad else "frozen_recon_head"
            rows.append(
                {
                    "parameter_name": full_name,
                    "numel": int(param.numel()),
                    "requires_grad": bool(param.requires_grad),
                    "category": category,
                }
            )
    return rows


def summarize_parameter_audit(rows: list[dict], selected_names: list[str], lora_cfg: LoRAConfig) -> dict:
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
        "lora_trainable_parameter_count": int(lora_trainable),
        "recon_head_trainable_parameter_count": int(recon_head_trainable),
        "trainable_ratio": float(trainable / total) if total > 0 else 0.0,
        "selected_lora_target_names": selected_names,
        "lora_config": asdict(lora_cfg),
        "no_unexpected_trainable_base_params": len(unexpected) == 0,
        "unexpected_trainable_base_parameter_names": unexpected,
        "trainable_scope_counts": {
            "lora": int(lora_trainable),
            "recon_head": int(recon_head_trainable),
        },
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


def waveform_to_eat_tensor(wav_16k: np.ndarray, cfg: Stage19ETrainConfig):
    import torch
    import torchaudio

    wav = torch.tensor(wav_16k, dtype=torch.float32)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=cfg.sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=cfg.num_mel_bins,
        dither=0.0,
        frame_shift=cfg.frame_shift_ms,
    )
    if mel.shape[0] < cfg.target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, cfg.target_length - mel.shape[0]))
    else:
        mel = mel[:cfg.target_length, :]
    mel = (mel - cfg.norm_mean) / (cfg.norm_std * 2)
    return mel.to(dtype=torch.float32)


def build_batch(paths: list[Path], cfg: Stage19ETrainConfig, device):
    import torch

    feats = [waveform_to_eat_tensor(load_waveform_16k(path), cfg) for path in paths]
    mel = torch.stack(feats, dim=0)
    return mel.to(device)


def random_time_mask(mel, mask_ratio: float, rng: np.random.Generator):
    import torch

    B, T, _ = mel.shape
    n_mask = max(1, int(T * mask_ratio))
    mask = torch.zeros((B, T), dtype=torch.bool, device=mel.device)
    for b in range(B):
        idx = rng.choice(T, size=n_mask, replace=False)
        mask[b, idx] = True
    masked = mel.clone()
    masked[mask] = 0.0
    return masked, mask


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


def min_clamp_frames(frames):
    import torch

    return torch.where(torch.abs(frames) < 0.1, torch.zeros_like(frames), frames)


def mean_pool_frames(frames):
    return frames.mean(dim=1)


def l2_normalize(x):
    import torch.nn.functional as F

    return F.normalize(x, dim=1)


def set_student_mode_for_training(student):
    student.eval()
    for module in student.modules():
        if is_lora_wrapper(module):
            module.train()


def compute_loss(student, teacher, recon_head, mel_batch, cfg: Stage19ETrainConfig, device, rng: np.random.Generator):
    import torch
    import torch.nn.functional as F

    masked_mel, time_mask = random_time_mask(mel_batch, cfg.mask_ratio, rng)
    student_frames_masked = extract_frame_output(student, masked_mel, device)
    pred_mel = recon_head(student_frames_masked)
    target_mel = mel_batch

    t_out = int(pred_mel.shape[1])
    target_down = F.interpolate(
        target_mel.transpose(1, 2),
        size=t_out,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    mask_down = F.interpolate(
        time_mask.float().unsqueeze(1),
        size=t_out,
        mode="nearest",
    ).squeeze(1).bool()
    if not torch.any(mask_down):
        mask_down[:, 0] = True

    recon_loss = F.mse_loss(pred_mel[mask_down], target_down[mask_down])

    with torch.no_grad():
        teacher_frames = extract_frame_output(teacher, mel_batch, device)
        teacher_emb = l2_normalize(mean_pool_frames(min_clamp_frames(teacher_frames)))
    student_frames_anchor = extract_frame_output(student, mel_batch, device)
    student_emb = l2_normalize(mean_pool_frames(min_clamp_frames(student_frames_anchor)))
    anchor_loss = (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=1)).mean()
    total = recon_loss + cfg.lambda_anchor * anchor_loss

    return {
        "total": total,
        "recon_loss": recon_loss,
        "anchor_loss": anchor_loss,
    }


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


def build_inventory_rows(train_items: list, test_items: list) -> list[dict]:
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
    return rows


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage19E Fan Anchor LoRA Sanity Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no training",
        "- no embedding extraction",
        "- no ASD scoring",
        "",
        "## Status",
        "",
        f"- clean EAT checkpoint exists: `{status['eat_model_path_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- stage19d decision file exists: `{status['stage19d_decision_exists']}`",
        f"- stage19d protocol file exists: `{status['stage19d_protocol_exists']}`",
        f"- output root writable: `{status['output_root_writable']}`",
        f"- fan source train-normal count: `{status['source_train_normal_count']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_audit_markdown(verdict: dict, summary: dict) -> list[str]:
    return [
        "# CTAF Stage19E Fan Anchor LoRA Sanity Audit",
        "",
        "## Protocol",
        "",
        "- machine: `fan`",
        "- training data: `source train-normal only`",
        "- split: `90/10 by seed 13711`",
        "- teacher: frozen clean EAT",
        "- student: clean EAT with LoRA only on last block attention qkv/proj",
        "- LoRA rank/alpha/dropout: `4 / 8 / 0.05`",
        "- mask ratio: `0.30`",
        "- loss: `L = L_recon_masked + lambda_anchor * L_anchor_embedding`",
        "- lambda anchor: `1.0`",
        "- checkpoint selection: `lowest validation total loss`",
        "",
        "## Gates",
        "",
        "- minimum pass: `adapted_ldn fan hmean > 0.5527191284766243`",
        "- strong pass: `adapted_ldn fan hmean >= 0.5820070015781968`",
        "- hard fail: `adapted_ldn fan hmean <= 0.5527191284766243`",
        "- additional guardrail: `adapted_baseline_knn_k5 should not be far below Stage18 baseline`",
        "",
        "## Training Verdict",
        "",
        f"- best checkpoint: `{verdict['best_checkpoint']}`",
        f"- best epoch: `{verdict['best_epoch']}`",
        f"- best validation total loss: `{verdict['best_val_loss']:.8f}`",
        f"- train count: `{verdict['train_count']}`",
        f"- val count: `{verdict['val_count']}`",
        f"- no unexpected base trainable params: `{summary['no_unexpected_trainable_base_params']}`",
        "",
        "## Guardrails",
        "",
        "- no ASD scoring in Stage19E",
        "- no test-label checkpoint selection",
        "- no anomaly-label training",
        "- no full7",
        "- no selector",
        "- no fusion",
        "- do not continue Stage19B checkpoint",
        "",
        "Stage19F will evaluate adapted embeddings later with a fixed backend.",
    ]


def build_trainable_rows(model, recon_head) -> list[dict]:
    rows = []
    for scope, module in [("eat", model), ("recon_head", recon_head)]:
        for name, param in module.named_parameters():
            full_name = f"{scope}.{name}"
            if is_lora_parameter_name(name):
                category = "lora_trainable" if param.requires_grad else "unexpected_frozen_lora"
            elif full_name.startswith("recon_head.") and param.requires_grad:
                category = "recon_head_trainable"
            elif full_name.startswith("recon_head."):
                category = "frozen_recon_head"
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


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    model_path = Path(args.eat_model_path)
    stage19d_root = Path(args.stage19d_root)
    decision_path = Path(args.stage19d_decision)
    protocol_path = Path(args.stage19d_protocol)

    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, args.machine) if items else []
    train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
    train_split, val_split = split_train_val(train_items, 0.10, args.seed) if train_items else ([], [])

    check_status = {
        "mode": "check",
        "machine": args.machine,
        "eat_model_path_exists": model_path.exists(),
        "data_root_exists": data_root.exists(),
        "stage19d_root_exists": stage19d_root.exists(),
        "stage19d_decision_exists": decision_path.exists(),
        "stage19d_protocol_exists": protocol_path.exists(),
        "output_root_writable": ensure_writable(out_root),
        "source_train_normal_count": len(train_items),
        "train_split_count": len(train_split),
        "val_split_count": len(val_split),
        "test_count": len(test_items),
        "imports": import_status(),
        "notes": "No model load, no forward, no training, no embedding extraction, no ASD scoring.",
    }
    write_json(out_root / "check_status.json", check_status)
    write_csv(out_root / "fan_split_inventory.csv", build_inventory_rows(train_items, test_items))


def run_inspect_lora(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.trainable_scope != "last_block_attention_lora":
        raise ValueError(f"Unsupported trainable_scope for Stage19E: {args.trainable_scope}")
    model_path = Path(args.eat_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")

    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model, device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(model)
    target_rows, selected_names = inspect_lora_targets(model, lora_cfg)
    if not selected_names:
        raise RuntimeError("No LoRA targets selected for Stage19E.")
    model = inject_lora_modules(model, selected_names, lora_cfg)
    recon_head = build_reconstruction_head(device)
    trainable_rows = parameter_audit_rows(model, recon_head)
    summary = summarize_trainable(trainable_rows, selected_names, lora_cfg)

    write_csv(out_root / "lora_target_inventory.csv", target_rows)
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_json(out_root / "trainable_summary.json", summary)


def run_train_fan_anchor_lora(args):
    import torch

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    if args.trainable_scope != "last_block_attention_lora":
        raise ValueError(f"Unsupported trainable_scope for Stage19E: {args.trainable_scope}")

    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data_root: {data_root}")

    cfg = Stage19ETrainConfig(
        trainable_scope=args.trainable_scope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mask_ratio=args.mask_ratio,
        lambda_anchor=args.lambda_anchor,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    all_items = filter_items(scan_dataset(data_root), args.machine)
    source_train_items, test_items = split_train_test(all_items, use_target_train=False)
    train_items, val_items = split_train_val(source_train_items, cfg.val_ratio, cfg.seed)
    if not train_items or not val_items:
        raise RuntimeError(
            f"Invalid fan source train/val split: train={len(train_items)} val={len(val_items)}"
        )

    train_inventory = build_inventory_rows(source_train_items, test_items)
    write_csv(out_root / "fan_split_inventory.csv", train_inventory)

    lora_cfg = LoRAConfig(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    teacher, device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(teacher)
    teacher.eval()

    student, _ = load_clean_eat_local(model_path, str(device))
    freeze_all_parameters(student)
    target_rows, selected_names = inspect_lora_targets(student, lora_cfg)
    if not selected_names:
        raise RuntimeError("No LoRA target modules selected for Stage19E training.")
    write_csv(out_root / "lora_target_inventory.csv", target_rows)
    student = inject_lora_modules(student, selected_names, lora_cfg)
    recon_head = build_reconstruction_head(device, hidden_size=cfg.hidden_size, recon_dim=cfg.recon_dim)

    trainable_rows = build_trainable_rows(student, recon_head)
    summary = summarize_trainable(trainable_rows, selected_names, lora_cfg)
    if not summary["no_unexpected_trainable_base_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_base_parameter_names']}")
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_json(out_root / "trainable_summary.json", summary)
    write_json(out_root / "stage19e_train_config.json", asdict(cfg))

    trainable_params = [p for p in student.parameters() if p.requires_grad] + list(recon_head.parameters())
    if not trainable_params:
        raise RuntimeError("No trainable LoRA or reconstruction-head parameters found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_path = out_root / "checkpoints" / "fan_stage19e_anchor_lora_best.pt"
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    log_rows = []

    val_batches = [val_items[i:i + cfg.batch_size] for i in range(0, len(val_items), cfg.batch_size)]

    for epoch in range(1, cfg.epochs + 1):
        set_student_mode_for_training(student)
        recon_head.train()
        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_anchor_sum = 0.0
        train_count = 0
        train_rng = np.random.default_rng(cfg.seed + epoch * 1000)

        perm = train_rng.permutation(len(train_items))
        for start in range(0, len(perm), cfg.batch_size):
            batch_ids = perm[start:start + cfg.batch_size]
            batch = [train_items[int(idx)] for idx in batch_ids]
            paths = [Path(item.path) for item in batch]
            mel_batch = build_batch(paths, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_loss(student, teacher, recon_head, mel_batch, cfg, device, train_rng)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
            train_total_sum += float(loss_dict["total"].item()) * bsz
            train_recon_sum += float(loss_dict["recon_loss"].item()) * bsz
            train_anchor_sum += float(loss_dict["anchor_loss"].item()) * bsz
            train_count += bsz

        student.eval()
        recon_head.eval()
        val_total_sum = 0.0
        val_recon_sum = 0.0
        val_anchor_sum = 0.0
        val_count = 0
        val_rng = np.random.default_rng(cfg.seed + 500000 + epoch)
        with torch.no_grad():
            for batch in val_batches:
                paths = [Path(item.path) for item in batch]
                mel_batch = build_batch(paths, cfg, device)
                loss_dict = compute_loss(student, teacher, recon_head, mel_batch, cfg, device, val_rng)
                bsz = len(batch)
                val_total_sum += float(loss_dict["total"].item()) * bsz
                val_recon_sum += float(loss_dict["recon_loss"].item()) * bsz
                val_anchor_sum += float(loss_dict["anchor_loss"].item()) * bsz
                val_count += bsz

        train_total = train_total_sum / max(train_count, 1)
        train_recon = train_recon_sum / max(train_count, 1)
        train_anchor = train_anchor_sum / max(train_count, 1)
        val_total = val_total_sum / max(val_count, 1)
        val_recon = val_recon_sum / max(val_count, 1)
        val_anchor = val_anchor_sum / max(val_count, 1)
        log_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total,
                "train_recon_loss": train_recon,
                "train_anchor_loss": train_anchor,
                "val_total_loss": val_total,
                "val_recon_loss": val_recon,
                "val_anchor_loss": val_anchor,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_total < best_val - 1e-8:
            best_val = val_total
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "student_state_dict": student.state_dict(),
                    "recon_head_state_dict": recon_head.state_dict(),
                    "train_config": asdict(cfg),
                    "lora_config": asdict(lora_cfg),
                    "lora_target_names": selected_names,
                    "trainable_summary": summary,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    write_csv(out_root / "train_log.csv", log_rows)
    verdict = {
        "machine": args.machine,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "trainable_scope": cfg.trainable_scope,
        "lora_config": asdict(lora_cfg),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "no_unexpected_trainable_base_params": summary["no_unexpected_trainable_base_params"],
        "notes": [
            "Training only.",
            "No ASD scoring performed in Stage19E.",
            "Stage19F should evaluate adapted embeddings with the fixed backend.",
        ],
    }
    write_json(out_root / "stage19e_training_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_markdown(verdict, summary))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_lora":
        run_inspect_lora(args)
        return
    if args.mode == "train_fan_anchor_lora":
        run_train_fan_anchor_lora(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
