#!/usr/bin/env python
"""Stage22B/C: geometry-aligned normal-only objective for clean EAT.

This script writes a Stage21E-compatible adapted EAT checkpoint, but changes
the training objective from masked reconstruction to backend-geometry alignment:

- clean-EAT neighborhood preservation
- light normal-safe augmentation consistency
- frozen LDN-proxy consistency
- dispersion guard against collapse

It uses only normal training audio from the Stage21C allowlist. It does not use
anomaly labels, test labels, selectors, or LDN parameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from audit_stage21d_global_dap_ssl_sanity import (
    build_batch,
    enable_last_n_blocks,
    extract_frame_output,
    freeze_all_parameters,
    load_allowlist_rows,
    load_clean_eat_local,
    split_train_val,
    summarize_allowlist,
    summarize_trainable,
    validate_allowlist_rows,
)


CHECKPOINT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ALLOWLIST_DEFAULT = "results/stage21c_dcase2022_additional_integrity_audit/allowed_train_normal_filelist.csv"
STAGE21C_ROOT_DEFAULT = "results/stage21c_dcase2022_additional_integrity_audit"
OUTPUT_ROOT_DEFAULT = "results/stage22b_geometry_objective_sanity"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE22B_GEOMETRY_OBJECTIVE_SANITY_AUDIT.md"
EPS = 1e-12


@dataclass
class Stage22GeometryConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 16
    epochs: int = 5
    lr: float = 5e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 2
    val_ratio: float = 0.10
    norm_mean: float = -4.268
    norm_std: float = 4.569
    target_length: int = 1024
    num_mel_bins: int = 128
    sample_rate: int = 16000
    frame_shift_ms: int = 10
    neighbor_k: int = 5
    ldn_K: int = 1
    score_k: int = 5
    neighbor_temperature: float = 0.05
    neighbor_loss_weight: float = 1.0
    anchor_loss_weight: float = 0.05
    aug_loss_weight: float = 0.20
    ldn_proxy_loss_weight: float = 0.50
    dispersion_loss_weight: float = 0.10
    min_std_ratio: float = 0.80
    aug_gain_min: float = 0.90
    aug_gain_max: float = 1.10
    aug_noise_std: float = 0.01
    aug_max_time_shift: int = 12
    aug_freq_mask_width: int = 8


def parse_args():
    p = argparse.ArgumentParser(description="Stage22B/C geometry objective sanity training")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_geometry_ssl"])
    p.add_argument("--eat_model_path", default=CHECKPOINT_DEFAULT)
    p.add_argument("--allowlist_csv", default=ALLOWLIST_DEFAULT)
    p.add_argument("--stage21c_root", default=STAGE21C_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--neighbor_loss_weight", type=float, default=1.0)
    p.add_argument("--anchor_loss_weight", type=float, default=0.05)
    p.add_argument("--aug_loss_weight", type=float, default=0.20)
    p.add_argument("--ldn_proxy_loss_weight", type=float, default=0.50)
    p.add_argument("--dispersion_loss_weight", type=float, default=0.10)
    p.add_argument("--neighbor_k", type=int, default=5)
    p.add_argument("--score_k", type=int, default=5)
    p.add_argument("--ldn_K", type=int, default=1)
    p.add_argument("--max_train_items", type=int, default=0, help="Optional smoke-test cap. 0 means all allowlist rows.")
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


def parameter_audit_rows_no_head(model) -> tuple[list[dict], list[dict]]:
    trainable_rows = []
    frozen_rows = []
    for name, param in model.named_parameters():
        row = {
            "parameter_name": f"eat.{name}",
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
        }
        if param.requires_grad:
            trainable_rows.append(row)
        else:
            frozen_rows.append(row)
    return trainable_rows, frozen_rows


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


def build_config(args) -> Stage22GeometryConfig:
    return Stage22GeometryConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        neighbor_loss_weight=args.neighbor_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        aug_loss_weight=args.aug_loss_weight,
        ldn_proxy_loss_weight=args.ldn_proxy_loss_weight,
        dispersion_loss_weight=args.dispersion_loss_weight,
        neighbor_k=args.neighbor_k,
        score_k=args.score_k,
        ldn_K=args.ldn_K,
    )


def torch_min_clamp(frames):
    import torch

    return torch.where(torch.abs(frames) < 0.1, torch.zeros_like(frames), frames)


def backend_embedding(model, mel_batch, device):
    import torch.nn.functional as F

    frames = extract_frame_output(model, mel_batch, device)
    pooled = torch_min_clamp(frames).mean(dim=1)
    return F.normalize(pooled, dim=1)


def chord_distance_torch(z):
    import torch
    import torch.nn.functional as F

    z = F.normalize(z, dim=1)
    sim = z @ z.T
    dist2 = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    return torch.sqrt(dist2 + 1e-12)


def diagonal_mask(n: int, device):
    import torch

    return torch.eye(n, dtype=torch.bool, device=device)


def neighbor_distribution(dist, k: int, temperature: float):
    import torch
    import torch.nn.functional as F

    n = dist.shape[0]
    kk = min(k, max(1, n - 1))
    diag = diagonal_mask(n, dist.device)
    masked = dist.masked_fill(diag, float("inf"))
    idx = torch.topk(masked, k=kk, dim=1, largest=False).indices
    logits = torch.full_like(dist, -1e9)
    logits.scatter_(1, idx, -dist.gather(1, idx) / max(temperature, 1e-6))
    return F.softmax(logits, dim=1)


def neighborhood_preservation_loss(teacher_z, student_z, cfg: Stage22GeometryConfig):
    import torch
    import torch.nn.functional as F

    if teacher_z.shape[0] <= 1:
        return torch.zeros([], device=student_z.device)
    with torch.no_grad():
        teacher_dist = chord_distance_torch(teacher_z)
        teacher_prob = neighbor_distribution(teacher_dist, cfg.neighbor_k, cfg.neighbor_temperature)
    student_dist = chord_distance_torch(student_z)
    student_prob = neighbor_distribution(student_dist, cfg.neighbor_k, cfg.neighbor_temperature)
    return F.kl_div(torch.log(student_prob + 1e-12), teacher_prob, reduction="batchmean")


def ldn_proxy_scores(z, cfg: Stage22GeometryConfig):
    import torch

    n = z.shape[0]
    if n <= 1:
        return torch.zeros((n,), device=z.device)
    dist = chord_distance_torch(z)
    diag = diagonal_mask(n, dist.device)
    masked = dist.masked_fill(diag, float("inf"))
    rho_k = min(cfg.ldn_K, max(1, n - 1))
    rho = torch.topk(masked, k=rho_k, dim=1, largest=False).values.mean(dim=1)
    adjusted = (dist - rho.unsqueeze(0)).masked_fill(diag, float("inf"))
    score_k = min(cfg.score_k, max(1, n - 1))
    return torch.topk(adjusted, k=score_k, dim=1, largest=False).values.mean(dim=1)


def zscore_like_student(student_scores, teacher_scores):
    teacher_mean = teacher_scores.mean().detach()
    teacher_std = teacher_scores.std(unbiased=False).detach().clamp_min(1e-6)
    return (student_scores - teacher_mean) / teacher_std


def ldn_proxy_consistency_loss(teacher_z, student_z, cfg: Stage22GeometryConfig):
    import torch
    import torch.nn.functional as F

    if teacher_z.shape[0] <= 2:
        return torch.zeros([], device=student_z.device)
    with torch.no_grad():
        teacher_scores = ldn_proxy_scores(teacher_z, cfg)
        teacher_norm = zscore_like_student(teacher_scores, teacher_scores)
    student_scores = ldn_proxy_scores(student_z, cfg)
    student_norm = zscore_like_student(student_scores, teacher_scores)
    return F.smooth_l1_loss(student_norm, teacher_norm)


def dispersion_guard_loss(teacher_z, student_z, cfg: Stage22GeometryConfig):
    import torch

    if teacher_z.shape[0] <= 2:
        return torch.zeros([], device=student_z.device)
    with torch.no_grad():
        teacher_std = teacher_z.std(dim=0, unbiased=False)
        floor = cfg.min_std_ratio * teacher_std
    student_std = student_z.std(dim=0, unbiased=False)
    return torch.relu(floor - student_std).mean()


def augment_mel_batch(mel_batch, cfg: Stage22GeometryConfig, rng: np.random.Generator):
    import torch

    aug = mel_batch.clone()
    bsz, n_frames, n_bins = aug.shape
    gains = rng.uniform(cfg.aug_gain_min, cfg.aug_gain_max, size=bsz).astype(np.float32)
    gain_tensor = torch.tensor(gains, dtype=aug.dtype, device=aug.device).view(bsz, 1, 1)
    aug = aug * gain_tensor
    if cfg.aug_noise_std > 0:
        aug = aug + torch.randn_like(aug) * cfg.aug_noise_std
    if cfg.aug_max_time_shift > 0:
        shifted = []
        for b in range(bsz):
            shift = int(rng.integers(-cfg.aug_max_time_shift, cfg.aug_max_time_shift + 1))
            shifted.append(torch.roll(aug[b], shifts=shift, dims=0))
        aug = torch.stack(shifted, dim=0)
    if cfg.aug_freq_mask_width > 0 and n_bins > cfg.aug_freq_mask_width:
        for b in range(bsz):
            start = int(rng.integers(0, n_bins - cfg.aug_freq_mask_width + 1))
            aug[b, :, start:start + cfg.aug_freq_mask_width] = 0.0
    return aug


def compute_geometry_loss(student, teacher, mel_batch, cfg: Stage22GeometryConfig, device, rng: np.random.Generator):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        teacher_z = backend_embedding(teacher, mel_batch, device)
    student_z = backend_embedding(student, mel_batch, device)

    neigh = neighborhood_preservation_loss(teacher_z, student_z, cfg)
    anchor = (1.0 - F.cosine_similarity(student_z, teacher_z, dim=1)).mean()
    ldn_proxy = ldn_proxy_consistency_loss(teacher_z, student_z, cfg)
    dispersion = dispersion_guard_loss(teacher_z, student_z, cfg)

    aug_consistency = torch.zeros([], device=mel_batch.device)
    if cfg.aug_loss_weight > 0:
        aug_mel = augment_mel_batch(mel_batch, cfg, rng)
        aug_z = backend_embedding(student, aug_mel, device)
        aug_consistency = (1.0 - F.cosine_similarity(aug_z, student_z.detach(), dim=1)).mean()

    total = (
        cfg.neighbor_loss_weight * neigh
        + cfg.anchor_loss_weight * anchor
        + cfg.aug_loss_weight * aug_consistency
        + cfg.ldn_proxy_loss_weight * ldn_proxy
        + cfg.dispersion_loss_weight * dispersion
    )
    return {
        "total": total,
        "neighbor_loss": neigh,
        "anchor_loss": anchor,
        "aug_consistency_loss": aug_consistency,
        "ldn_proxy_loss": ldn_proxy,
        "dispersion_guard_loss": dispersion,
    }


def rows_with_optional_cap(rows: list[dict], max_items: int) -> list[dict]:
    if max_items <= 0:
        return rows
    return rows[:max_items]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    allowlist_path = Path(args.allowlist_csv)
    rows = load_allowlist_rows(allowlist_path) if allowlist_path.exists() else []
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_model_path_exists": Path(args.eat_model_path).exists(),
        "allowlist_csv_exists": allowlist_path.exists(),
        "stage21c_root_exists": Path(args.stage21c_root).exists(),
        "stage21c_integrity_exists": (Path(args.stage21c_root) / "integrity_verdict.json").exists(),
        "allowlist_count": len(rows),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Training mode uses only Stage21C normal train allowlist rows.",
        ],
    })


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    model_path = Path(args.eat_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")

    model, device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(model)
    allowed_blocks = enable_last_n_blocks(model, cfg.trainable_last_blocks)
    trainable_rows, frozen_rows = parameter_audit_rows_no_head(model)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    write_json(out_root / "model_load_status.json", {
        "mode": "inspect_trainable",
        "eat_model_path": str(model_path),
        "device": str(device),
        "model_class": model.__class__.__name__,
        "trainable_last_blocks": cfg.trainable_last_blocks,
        "allowed_block_names": allowed_blocks,
    })
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)


def run_train_geometry_ssl(args):
    import torch

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    allowlist_path = Path(args.allowlist_csv)
    model_path = Path(args.eat_model_path)
    stage21c_root = Path(args.stage21c_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not allowlist_path.exists():
        raise FileNotFoundError(f"Missing allowlist csv: {allowlist_path}")

    integrity_path = stage21c_root / "integrity_verdict.json"
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text())
        if not integrity.get("ready_for_stage21d"):
            raise RuntimeError("Stage21C integrity verdict is not ready_for_stage21d.")

    rows = rows_with_optional_cap(load_allowlist_rows(allowlist_path), args.max_train_items)
    validate_allowlist_rows(rows)
    cfg = build_config(args)
    set_seed(args.seed)
    train_rows, val_rows = split_train_val(rows, cfg.val_ratio, args.seed)
    if not train_rows or not val_rows:
        raise RuntimeError(f"Invalid train/val split: train={len(train_rows)} val={len(val_rows)}")

    split_inventory = []
    for row in train_rows:
        split_inventory.append({**row, "split_assignment": "train"})
    for row in val_rows:
        split_inventory.append({**row, "split_assignment": "val"})
    write_csv(out_root / "global_split_inventory.csv", split_inventory)

    teacher, device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(teacher)
    teacher.eval()

    student, _ = load_clean_eat_local(model_path, str(device))
    freeze_all_parameters(student)
    allowed_blocks = enable_last_n_blocks(student, cfg.trainable_last_blocks)
    trainable_rows, frozen_rows = parameter_audit_rows_no_head(student)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    if not summary["no_unexpected_earlier_layer_trainable_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_parameter_names']}")

    write_json(out_root / "model_load_status.json", {
        "mode": "train_geometry_ssl",
        "eat_model_path": str(model_path),
        "allowlist_csv": str(allowlist_path),
        "device": str(device),
        "model_class": student.__class__.__name__,
        "trainable_last_blocks": cfg.trainable_last_blocks,
        "allowed_block_names": allowed_blocks,
    })
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_path = out_root / "checkpoints" / "stage22_geometry_best.pt"
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    log_rows = []

    train_batches = [train_rows[i:i + cfg.batch_size] for i in range(0, len(train_rows), cfg.batch_size)]
    val_batches = [val_rows[i:i + cfg.batch_size] for i in range(0, len(val_rows), cfg.batch_size)]

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        train_sums = {k: 0.0 for k in ["total", "neighbor_loss", "anchor_loss", "aug_consistency_loss", "ldn_proxy_loss", "dispersion_guard_loss"]}
        train_count = 0
        rng = np.random.default_rng(args.seed + epoch * 1000)
        for batch in train_batches:
            paths = [Path(row["path"]) for row in batch]
            mel_batch = build_batch(paths, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_geometry_loss(student, teacher, mel_batch, cfg, device, rng)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()

            bsz = len(batch)
            for key in train_sums:
                train_sums[key] += float(loss_dict[key].item()) * bsz
            train_count += bsz

        student.eval()
        val_sums = {k: 0.0 for k in train_sums}
        val_count = 0
        val_rng = np.random.default_rng(args.seed + 500000 + epoch)
        with torch.no_grad():
            for batch in val_batches:
                paths = [Path(row["path"]) for row in batch]
                mel_batch = build_batch(paths, cfg, device)
                loss_dict = compute_geometry_loss(student, teacher, mel_batch, cfg, device, val_rng)
                bsz = len(batch)
                for key in val_sums:
                    val_sums[key] += float(loss_dict[key].item()) * bsz
                val_count += bsz

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for key in train_sums:
            row[f"train_{key}"] = train_sums[key] / max(train_count, 1)
            row[f"val_{key}"] = val_sums[key] / max(val_count, 1)
        log_rows.append(row)
        val_loss = row["val_total"]

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save({
                "student_state_dict": student.state_dict(),
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "allowed_block_names": allowed_blocks,
                "trainable_summary": summary,
                "allowlist_csv": str(allowlist_path),
                "stage": "stage22_geometry_objective",
                "uses_anomaly_labels": False,
                "uses_test_labels_for_checkpoint_selection": False,
                "ldn_parameters_tuned": False,
            }, best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    group_rows, machine_rows = summarize_allowlist(rows)
    write_csv(out_root / "allowlist_group_summary.csv", group_rows)
    write_csv(out_root / "allowlist_machine_summary.csv", machine_rows)
    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage22_geometry_train_config.json", asdict(cfg))
    write_json(out_root / "stage22_geometry_training_verdict.json", {
        "allowlist_csv": str(allowlist_path),
        "total_allowlist_count": len(rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "no_unexpected_earlier_layer_trainable_params": summary["no_unexpected_earlier_layer_trainable_params"],
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "stage21e_compatible_checkpoint": True,
        "notes": [
            "Training objective is geometry-aligned normal-only SSL.",
            "No ASD scoring performed in this stage.",
            "Evaluate fan first with Stage21E and the frozen min_clamp__mean + ldn_diff_K1_scorek5 backend.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage22B Geometry Objective Sanity Audit",
        "",
        f"- allowlist csv: `{allowlist_path}`",
        f"- total allowlist count: `{len(rows)}`",
        f"- train count: `{len(train_rows)}`",
        f"- val count: `{len(val_rows)}`",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation loss: `{best_val:.8f}`",
        "",
        "Objective components:",
        "",
        f"- neighborhood preservation weight: `{cfg.neighbor_loss_weight}`",
        f"- anchor weight: `{cfg.anchor_loss_weight}`",
        f"- augmentation consistency weight: `{cfg.aug_loss_weight}`",
        f"- LDN proxy consistency weight: `{cfg.ldn_proxy_loss_weight}`",
        f"- dispersion guard weight: `{cfg.dispersion_loss_weight}`",
        "",
        "No anomaly labels, test-label checkpoint selection, selectors, or LDN parameter tuning were used.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_trainable":
        run_inspect_trainable(args)
        return
    if args.mode == "train_geometry_ssl":
        run_train_geometry_ssl(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
