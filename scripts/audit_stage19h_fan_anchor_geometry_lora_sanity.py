#!/usr/bin/env python
"""Stage 19H: fan anchor + distance-geometry regularized LoRA training.

Planning/training scaffold only.

Allowed:
- clean EAT load
- fan source train-normal split
- LoRA on last-block attention qkv/proj only
- masked reconstruction + anchor embedding preservation + pairwise geometry preservation

Blocked:
- no ASD scoring
- no test-label checkpoint selection
- no anomaly-label training
- no full7
- no selector
- no fusion
- no Stage19B checkpoint continuation
- no Stage19E checkpoint continuation
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


def load_stage19e_helpers():
    stage19e_path = ROOT / "scripts" / "audit_stage19e_fan_anchor_lora_sanity.py"
    spec = importlib.util.spec_from_file_location("stage19e_helpers", stage19e_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Stage19E helpers from {stage19e_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


stage19e = load_stage19e_helpers()


CHECKPOINT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE19G_ROOT_DEFAULT = "results/stage19g_fan_lora_ldn_failure_analysis"
STAGE19G_RECOMMENDATION_DEFAULT = "results/stage19g_fan_lora_ldn_failure_analysis/stage19h_recommendation.json"
OUTPUT_ROOT_DEFAULT = "results/stage19h_fan_anchor_geometry_lora_sanity"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19H_FAN_ANCHOR_GEOMETRY_LORA_SANITY_AUDIT.md"
MACHINE_DEFAULT = "fan"

EXPECTED_LORA_TARGET_NAMES = [
    "model.blocks.11.attn.qkv",
    "model.blocks.11.attn.proj",
]


@dataclass
class Stage19HTrainConfig(stage19e.Stage19ETrainConfig):
    lambda_dist: float = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Stage19H fan anchor+geometry regularized LoRA sanity")
    p.add_argument(
        "--mode",
        required=True,
        choices=["check", "inspect_lora", "debug_one_batch", "train_fan_anchor_geometry_lora"],
    )
    p.add_argument("--eat_model_path", default=CHECKPOINT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage19g_root", default=STAGE19G_ROOT_DEFAULT)
    p.add_argument("--stage19g_recommendation", default=STAGE19G_RECOMMENDATION_DEFAULT)
    p.add_argument("--trainable_scope", default="last_block_attention_lora")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--mask_ratio", type=float, default=0.30)
    p.add_argument("--lambda_anchor", type=float, default=1.0)
    p.add_argument("--lambda_dist", type=float, default=0.1)
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


def write_loss_diagnostic(path: Path, payload: dict):
    write_json(path, payload)


def import_status():
    status = {"json": True, "csv": True, "pathlib": True}
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
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


def ensure_fixed_protocol(args):
    expected = {
        "machine": "fan",
        "trainable_scope": "last_block_attention_lora",
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_dropout": 0.05,
        "mask_ratio": 0.30,
        "lambda_anchor": 1.0,
        "lambda_dist": 0.1,
        "batch_size": 8,
        "epochs": 5,
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "early_stop_patience": 2,
        "seed": 13711,
    }
    for key, value in expected.items():
        actual = getattr(args, key)
        if isinstance(value, float):
            if not math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Fixed protocol mismatch for {key}: expected {value}, got {actual}")
        elif actual != value:
            raise ValueError(f"Fixed protocol mismatch for {key}: expected {value}, got {actual}")


def isclose_float(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def build_check_status(args, data_root: Path, out_root: Path) -> dict:
    items = []
    train_items = []
    test_items = []
    if data_root.exists():
        items, train_items, test_items = stage19e.expected_items(data_root, args.machine)
    train_split, val_split = stage19e.split_train_val(train_items, 0.10, args.seed) if train_items else ([], [])
    return {
        "mode": "check",
        "machine": args.machine,
        "eat_model_path_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "stage19g_root_exists": Path(args.stage19g_root).exists(),
        "stage19g_recommendation_exists": Path(args.stage19g_recommendation).exists(),
        "output_root_writable": ensure_writable(out_root),
        "source_train_normal_count": len(train_items),
        "train_split_count": len(train_split),
        "val_split_count": len(val_split),
        "test_count": len(test_items),
        "imports": import_status(),
        "notes": "No model load, no forward, no training, no embedding extraction, no ASD scoring.",
    }


def build_model_load_status(
    model_path: Path,
    device: str,
    lora_cfg,
    selected_names: list[str],
    summary: dict,
    trainable_rows: list[dict],
    recon_head_created: bool,
):
    return {
        "clean_eat_checkpoint": str(model_path),
        "device": device,
        "selected_lora_target_names": selected_names,
        "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
        "lora_target_match": set(selected_names) == set(EXPECTED_LORA_TARGET_NAMES),
        "selected_lora_target_count": len(selected_names),
        "no_mlp_lora": all("mlp" not in name.lower() for name in selected_names),
        "no_earlier_block_lora": all("blocks.11." in name for name in selected_names),
        "recon_head_created": recon_head_created,
        "no_unexpected_trainable_base_params": summary["no_unexpected_trainable_base_params"],
        "unexpected_trainable_base_parameter_names": summary["unexpected_trainable_base_parameter_names"],
        "trainable_scope": "last_block_attention_lora",
        "lora_config": asdict(lora_cfg),
        "trainable_summary": summary,
    }


def safe_l2_normalize(embeddings, eps: float = 1e-12):
    import torch

    norm = torch.linalg.norm(embeddings, dim=1, keepdim=True).clamp_min(eps)
    return embeddings / norm


def safe_pairwise_chord_distance_matrix(embeddings):
    import torch

    emb = safe_l2_normalize(embeddings)
    norm2 = (emb * emb).sum(dim=1, keepdim=True)
    dist2 = norm2 + norm2.transpose(0, 1) - 2.0 * (emb @ emb.transpose(0, 1))
    dist2 = torch.clamp(dist2, min=0.0)
    return torch.sqrt(dist2 + 1e-12)


def tensor_stats(x):
    import torch

    if x.numel() == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(torch.amin(x).item()),
        "max": float(torch.amax(x).item()),
        "mean": float(torch.mean(x).item()),
    }


def offdiag_stats(matrix):
    import torch

    if matrix.shape[0] <= 1:
        return {"min": None, "max": None, "mean": None}
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    values = matrix[mask]
    if values.numel() == 0:
        return {"min": None, "max": None, "mean": None}
    return tensor_stats(values)


def pairwise_distance_preservation_loss(student_emb, teacher_emb):
    import torch
    import torch.nn.functional as F

    if student_emb.shape[0] < 2:
        return torch.zeros((), device=student_emb.device, dtype=student_emb.dtype)
    student_dist = safe_pairwise_chord_distance_matrix(student_emb)
    teacher_dist = safe_pairwise_chord_distance_matrix(teacher_emb)
    mask = ~torch.eye(student_dist.shape[0], dtype=torch.bool, device=student_dist.device)
    student_values = student_dist[mask]
    teacher_values = teacher_dist[mask]
    if student_values.numel() == 0 or teacher_values.numel() == 0:
        return torch.zeros((), device=student_emb.device, dtype=student_emb.dtype)
    return F.mse_loss(student_values, teacher_values)


def finite_loss_guard(loss_dict: dict, diagnostic_path: Path | None = None, extra: dict | None = None):
    import torch

    total = loss_dict.get("total")
    recon = loss_dict.get("recon_loss")
    anchor = loss_dict.get("anchor_loss")
    pairwise = loss_dict.get("pairwise_loss")
    finite_flags = {
        "recon_loss": bool(torch.isfinite(recon).item()),
        "anchor_loss": bool(torch.isfinite(anchor).item()),
        "pairwise_loss": bool(torch.isfinite(pairwise).item()),
        "total_loss": bool(torch.isfinite(total).item()),
    }
    if all(finite_flags.values()):
        return True

    payload = {
        "recon_loss": float(recon.item()),
        "anchor_loss": float(anchor.item()),
        "pairwise_loss": float(pairwise.item()),
        "total_loss": float(total.item()),
        "all_losses_finite": False,
    }
    payload.update(finite_flags)
    if extra:
        payload.update(extra)
    if diagnostic_path is not None:
        write_loss_diagnostic(diagnostic_path, payload)
    raise RuntimeError(
        "Non-finite Stage19H loss encountered: "
        f"recon={payload['recon_loss']} anchor={payload['anchor_loss']} "
        f"pairwise={payload['pairwise_loss']} total={payload['total_loss']}"
    )


def compute_loss(
    student,
    teacher,
    recon_head,
    mel_batch,
    cfg: Stage19HTrainConfig,
    device,
    rng: np.random.Generator,
    diagnostic_path: Path | None = None,
    extra_diagnostic: dict | None = None,
):
    import torch
    import torch.nn.functional as F

    masked_mel, time_mask = stage19e.random_time_mask(mel_batch, cfg.mask_ratio, rng)
    student_frames_masked = stage19e.extract_frame_output(student, masked_mel, device)
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
        teacher_frames = stage19e.extract_frame_output(teacher, mel_batch, device)
        teacher_emb = safe_l2_normalize(stage19e.mean_pool_frames(stage19e.min_clamp_frames(teacher_frames)))
    student_frames_anchor = stage19e.extract_frame_output(student, mel_batch, device)
    student_emb = safe_l2_normalize(stage19e.mean_pool_frames(stage19e.min_clamp_frames(student_frames_anchor)))

    anchor_loss = (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=1)).mean()
    pairwise_loss = pairwise_distance_preservation_loss(student_emb, teacher_emb)
    total = recon_loss + cfg.lambda_anchor * anchor_loss + cfg.lambda_dist * pairwise_loss

    loss_dict = {
        "total": total,
        "recon_loss": recon_loss,
        "anchor_loss": anchor_loss,
        "pairwise_loss": pairwise_loss,
    }
    finite_loss_guard(loss_dict, diagnostic_path=diagnostic_path, extra=extra_diagnostic)
    return loss_dict


def build_debug_payload(
    loss_dict: dict,
    teacher_emb,
    student_emb,
    teacher_pairwise,
    student_pairwise,
    batch_size: int,
    cfg: Stage19HTrainConfig,
):
    import torch

    teacher_norm = torch_compat_norm(teacher_emb)
    student_norm = torch_compat_norm(student_emb)
    teacher_pairwise_stats = offdiag_stats(teacher_pairwise)
    student_pairwise_stats = offdiag_stats(student_pairwise)
    return {
        "recon_loss": float(loss_dict["recon_loss"].item()),
        "anchor_loss": float(loss_dict["anchor_loss"].item()),
        "pairwise_loss": float(loss_dict["pairwise_loss"].item()),
        "total_loss": float(loss_dict["total"].item()),
        "teacher_embedding_norm_min": float(teacher_norm.min().item()),
        "teacher_embedding_norm_max": float(teacher_norm.max().item()),
        "teacher_embedding_norm_mean": float(teacher_norm.mean().item()),
        "student_embedding_norm_min": float(student_norm.min().item()),
        "student_embedding_norm_max": float(student_norm.max().item()),
        "student_embedding_norm_mean": float(student_norm.mean().item()),
        "teacher_pairwise_dist_min": teacher_pairwise_stats["min"],
        "teacher_pairwise_dist_max": teacher_pairwise_stats["max"],
        "teacher_pairwise_dist_mean": teacher_pairwise_stats["mean"],
        "student_pairwise_dist_min": student_pairwise_stats["min"],
        "student_pairwise_dist_max": student_pairwise_stats["max"],
        "student_pairwise_dist_mean": student_pairwise_stats["mean"],
        "all_losses_finite": bool(
            torch.isfinite(loss_dict["recon_loss"]).item()
            and torch.isfinite(loss_dict["anchor_loss"]).item()
            and torch.isfinite(loss_dict["pairwise_loss"]).item()
            and torch.isfinite(loss_dict["total"]).item()
        ),
        "batch_size": batch_size,
        "mask_ratio": cfg.mask_ratio,
        "lambda_anchor": cfg.lambda_anchor,
        "lambda_dist": cfg.lambda_dist,
    }


def torch_compat_norm(embeddings):
    import torch

    return torch.linalg.norm(embeddings, dim=1)


def build_audit_markdown(verdict: dict, summary: dict) -> list[str]:
    return [
        "# CTAF Stage19H Fan Anchor Geometry LoRA Sanity Audit",
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
        "- loss: `L = L_recon_masked + lambda_anchor * L_anchor_embedding + lambda_dist * L_pairwise_distance_preservation`",
        "- lambda anchor: `1.0`",
        "- lambda dist: `0.1`",
        "- checkpoint selection: `lowest validation total loss`",
        "",
        "## Guardrails",
        "",
        "- no ASD scoring in Stage19H",
        "- no test-label checkpoint selection",
        "- no anomaly-label training",
        "- no full7",
        "- no selector",
        "- no fusion",
        "- do not continue Stage19B checkpoint",
        "- do not continue Stage19E checkpoint",
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
        "Stage19I will evaluate adapted embeddings later with the fixed backend.",
    ]


def check_fixed_paths(args):
    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    check_status = build_check_status(args, data_root, out_root)
    items = []
    train_items = []
    test_items = []
    if data_root.exists():
        items, train_items, test_items = stage19e.expected_items(data_root, args.machine)
    write_json(out_root / "check_status.json", check_status)
    write_csv(out_root / "fan_split_inventory.csv", stage19e.build_inventory_rows(train_items, test_items))


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    check_fixed_paths(args)


def run_inspect_lora(args):
    ensure_fixed_protocol(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")

    lora_cfg = stage19e.LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model, device = stage19e.load_clean_eat_local(model_path, args.device)
    stage19e.freeze_all_parameters(model)
    target_rows, selected_names = stage19e.inspect_lora_targets(model, lora_cfg)
    if set(selected_names) != set(EXPECTED_LORA_TARGET_NAMES):
        raise RuntimeError(
            f"Stage19H LoRA targets must match {EXPECTED_LORA_TARGET_NAMES}, got {selected_names}"
        )
    model = stage19e.inject_lora_modules(model, selected_names, lora_cfg)
    recon_head = stage19e.build_reconstruction_head(device)
    trainable_rows = stage19e.build_trainable_rows(model, recon_head)
    summary = stage19e.summarize_trainable(trainable_rows, selected_names, lora_cfg)
    if not summary["no_unexpected_trainable_base_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_base_parameter_names']}")

    model_load_status = build_model_load_status(
        model_path=model_path,
        device=str(device),
        lora_cfg=lora_cfg,
        selected_names=selected_names,
        summary=summary,
        trainable_rows=trainable_rows,
        recon_head_created=True,
    )
    write_csv(out_root / "lora_target_inventory.csv", target_rows)
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_json(out_root / "trainable_summary.json", summary)
    write_json(out_root / "model_load_status.json", model_load_status)


def run_debug_one_batch(args):
    import torch

    ensure_fixed_protocol(args)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data_root: {data_root}")

    cfg = Stage19HTrainConfig(
        trainable_scope=args.trainable_scope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mask_ratio=args.mask_ratio,
        lambda_anchor=args.lambda_anchor,
        lambda_dist=args.lambda_dist,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    all_items = stage19e.filter_items(stage19e.scan_dataset(data_root), args.machine)
    source_train_items, _test_items = stage19e.split_train_test(all_items, use_target_train=False)
    train_items, _val_items = stage19e.split_train_val(source_train_items, cfg.val_ratio, cfg.seed)
    if not train_items:
        raise RuntimeError("No fan source train-normal items available for debug_one_batch.")

    lora_cfg = stage19e.LoRAConfig(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    teacher, device = stage19e.load_clean_eat_local(model_path, args.device)
    stage19e.freeze_all_parameters(teacher)
    teacher.eval()

    student, _ = stage19e.load_clean_eat_local(model_path, str(device))
    stage19e.freeze_all_parameters(student)
    _target_rows, selected_names = stage19e.inspect_lora_targets(student, lora_cfg)
    if set(selected_names) != set(EXPECTED_LORA_TARGET_NAMES):
        raise RuntimeError(
            f"Stage19H LoRA targets must match {EXPECTED_LORA_TARGET_NAMES}, got {selected_names}"
        )
    student = stage19e.inject_lora_modules(student, selected_names, lora_cfg)
    recon_head = stage19e.build_reconstruction_head(device, hidden_size=cfg.hidden_size, recon_dim=cfg.recon_dim)
    student.eval()
    recon_head.eval()

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(train_items))
    batch_ids = perm[: cfg.batch_size]
    batch = [train_items[int(idx)] for idx in batch_ids]
    paths = [Path(item.path) for item in batch]
    mel_batch = stage19e.build_batch(paths, cfg, device)
    rng = np.random.default_rng(cfg.seed + 12345)
    debug_diagnostic_path = out_root / "debug_one_batch_loss.json"

    loss_dict = compute_loss(student, teacher, recon_head, mel_batch, cfg, device, rng, diagnostic_path=debug_diagnostic_path)

    with torch.no_grad():
        teacher_frames = stage19e.extract_frame_output(teacher, mel_batch, device)
        student_frames = stage19e.extract_frame_output(student, mel_batch, device)
        teacher_emb = safe_l2_normalize(stage19e.mean_pool_frames(stage19e.min_clamp_frames(teacher_frames)))
        student_emb = safe_l2_normalize(stage19e.mean_pool_frames(stage19e.min_clamp_frames(student_frames)))
        teacher_pairwise = safe_pairwise_chord_distance_matrix(teacher_emb)
        student_pairwise = safe_pairwise_chord_distance_matrix(student_emb)

    payload = build_debug_payload(
        loss_dict=loss_dict,
        teacher_emb=teacher_emb,
        student_emb=student_emb,
        teacher_pairwise=teacher_pairwise,
        student_pairwise=student_pairwise,
        batch_size=len(batch),
        cfg=cfg,
    )
    write_json(debug_diagnostic_path, payload)


def run_train_fan_anchor_geometry_lora(args):
    import torch

    ensure_fixed_protocol(args)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    stage19g_rec = Path(args.stage19g_recommendation)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data_root: {data_root}")
    if not stage19g_rec.exists():
        raise FileNotFoundError(f"Missing Stage19G recommendation: {stage19g_rec}")

    cfg = Stage19HTrainConfig(
        trainable_scope=args.trainable_scope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mask_ratio=args.mask_ratio,
        lambda_anchor=args.lambda_anchor,
        lambda_dist=args.lambda_dist,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    all_items = stage19e.filter_items(stage19e.scan_dataset(data_root), args.machine)
    source_train_items, test_items = stage19e.split_train_test(all_items, use_target_train=False)
    train_items, val_items = stage19e.split_train_val(source_train_items, cfg.val_ratio, cfg.seed)
    if not train_items or not val_items:
        raise RuntimeError(
            f"Invalid fan source train/val split: train={len(train_items)} val={len(val_items)}"
        )

    write_json(out_root / "check_status.json", build_check_status(args, data_root, out_root))
    write_csv(out_root / "fan_split_inventory.csv", stage19e.build_inventory_rows(source_train_items, test_items))

    lora_cfg = stage19e.LoRAConfig(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    teacher, device = stage19e.load_clean_eat_local(model_path, args.device)
    stage19e.freeze_all_parameters(teacher)
    teacher.eval()

    student, _ = stage19e.load_clean_eat_local(model_path, str(device))
    stage19e.freeze_all_parameters(student)
    target_rows, selected_names = stage19e.inspect_lora_targets(student, lora_cfg)
    if set(selected_names) != set(EXPECTED_LORA_TARGET_NAMES):
        raise RuntimeError(
            f"Stage19H LoRA targets must match {EXPECTED_LORA_TARGET_NAMES}, got {selected_names}"
        )
    student = stage19e.inject_lora_modules(student, selected_names, lora_cfg)
    recon_head = stage19e.build_reconstruction_head(device, hidden_size=cfg.hidden_size, recon_dim=cfg.recon_dim)

    trainable_rows = stage19e.build_trainable_rows(student, recon_head)
    summary = stage19e.summarize_trainable(trainable_rows, selected_names, lora_cfg)
    if not summary["no_unexpected_trainable_base_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_base_parameter_names']}")
    write_json(
        out_root / "model_load_status.json",
        build_model_load_status(
            model_path=model_path,
            device=str(device),
            lora_cfg=lora_cfg,
            selected_names=selected_names,
            summary=summary,
            trainable_rows=trainable_rows,
            recon_head_created=True,
        ),
    )
    write_csv(out_root / "lora_target_inventory.csv", target_rows)
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_json(out_root / "trainable_summary.json", summary)
    write_json(out_root / "stage19h_train_config.json", asdict(cfg))

    trainable_params = [p for p in student.parameters() if p.requires_grad] + [p for p in recon_head.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA or reconstruction-head parameters found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_path = out_root / "checkpoints" / "fan_stage19h_anchor_geometry_lora_best.pt"
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    log_rows = []

    val_batches = [val_items[i : i + cfg.batch_size] for i in range(0, len(val_items), cfg.batch_size)]

    for epoch in range(1, cfg.epochs + 1):
        stage19e.set_student_mode_for_training(student)
        recon_head.train()
        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_anchor_sum = 0.0
        train_pairwise_sum = 0.0
        train_count = 0
        train_rng = np.random.default_rng(cfg.seed + epoch * 1000)

        perm = train_rng.permutation(len(train_items))
        for start in range(0, len(perm), cfg.batch_size):
            batch_ids = perm[start : start + cfg.batch_size]
            batch = [train_items[int(idx)] for idx in batch_ids]
            paths = [Path(item.path) for item in batch]
            mel_batch = stage19e.build_batch(paths, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss_diag_path = out_root / "diagnostics" / f"nonfinite_train_epoch{epoch:02d}_step{start // max(cfg.batch_size, 1):03d}.json"
            loss_dict = compute_loss(
                student,
                teacher,
                recon_head,
                mel_batch,
                cfg,
                device,
                train_rng,
                diagnostic_path=loss_diag_path,
                extra_diagnostic={"epoch": epoch, "batch_start_index": int(start), "phase": "train"},
            )
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
            train_total_sum += float(loss_dict["total"].item()) * bsz
            train_recon_sum += float(loss_dict["recon_loss"].item()) * bsz
            train_anchor_sum += float(loss_dict["anchor_loss"].item()) * bsz
            train_pairwise_sum += float(loss_dict["pairwise_loss"].item()) * bsz
            train_count += bsz

        student.eval()
        recon_head.eval()
        val_total_sum = 0.0
        val_recon_sum = 0.0
        val_anchor_sum = 0.0
        val_pairwise_sum = 0.0
        val_count = 0
        val_rng = np.random.default_rng(cfg.seed + 500000 + epoch)
        with torch.no_grad():
            for batch in val_batches:
                paths = [Path(item.path) for item in batch]
                mel_batch = stage19e.build_batch(paths, cfg, device)
                loss_diag_path = out_root / "diagnostics" / f"nonfinite_val_epoch{epoch:02d}.json"
                loss_dict = compute_loss(
                    student,
                    teacher,
                    recon_head,
                    mel_batch,
                    cfg,
                    device,
                    val_rng,
                    diagnostic_path=loss_diag_path,
                    extra_diagnostic={"epoch": epoch, "phase": "val"},
                )
                bsz = len(batch)
                val_total_sum += float(loss_dict["total"].item()) * bsz
                val_recon_sum += float(loss_dict["recon_loss"].item()) * bsz
                val_anchor_sum += float(loss_dict["anchor_loss"].item()) * bsz
                val_pairwise_sum += float(loss_dict["pairwise_loss"].item()) * bsz
                val_count += bsz

        train_total = train_total_sum / max(train_count, 1)
        train_recon = train_recon_sum / max(train_count, 1)
        train_anchor = train_anchor_sum / max(train_count, 1)
        train_pairwise = train_pairwise_sum / max(train_count, 1)
        val_total = val_total_sum / max(val_count, 1)
        val_recon = val_recon_sum / max(val_count, 1)
        val_anchor = val_anchor_sum / max(val_count, 1)
        val_pairwise = val_pairwise_sum / max(val_count, 1)
        log_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total,
                "train_recon_loss": train_recon,
                "train_anchor_loss": train_anchor,
                "train_pairwise_loss": train_pairwise,
                "val_total_loss": val_total,
                "val_recon_loss": val_recon,
                "val_anchor_loss": val_anchor,
                "val_pairwise_loss": val_pairwise,
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
        "lambda_anchor": cfg.lambda_anchor,
        "lambda_dist": cfg.lambda_dist,
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "no_unexpected_trainable_base_params": summary["no_unexpected_trainable_base_params"],
        "notes": [
            "Training only.",
            "No ASD scoring performed in Stage19H.",
            "Stage19I should evaluate adapted embeddings with the fixed backend.",
        ],
    }
    write_json(out_root / "stage19h_training_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_markdown(verdict, summary))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_lora":
        run_inspect_lora(args)
        return
    if args.mode == "debug_one_batch":
        run_debug_one_batch(args)
        return
    if args.mode == "train_fan_anchor_geometry_lora":
        run_train_fan_anchor_geometry_lora(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
