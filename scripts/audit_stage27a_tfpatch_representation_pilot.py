#!/usr/bin/env python
"""Stage27A: TF-patch representation-level pilot for clean EAT.

Stage24/25/26 showed that raw segment scoring and local-head-only pseudo
boundary scoring are not enough. Stage27A moves the TF-patch objective into the
representation itself:

- source train-normal only
- clean EAT teacher frozen
- EAT student with last N blocks trainable
- same-machine mel TF patch replacement
- local boundary head on student chunks guides patch sensitivity
- clean anchor keeps student close to clean EAT geometry
- non-overlap pseudo chunks are kept consistent with clean teacher chunks

The saved checkpoint is compatible with adapted EAT evaluators:
`student_state_dict` + `train_config`. The boundary head is saved only for audit
and is not used for ASD scoring.

No anomaly labels, no test-label checkpoint selection, no LDN tuning, no
selector, no fusion.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21e_adapted_eval import (
    extract_frame_output,
    load_clean_eat_local,
    load_waveform_16k,
    waveform_to_eat_tensor,
)
from audit_stage21d_global_dap_ssl_sanity import (
    enable_last_n_blocks,
    freeze_all_parameters,
    summarize_trainable as base_summarize_trainable,
)


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage27a_tfpatch_representation_pilot"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE27A_TFPATCH_REPRESENTATION_PILOT.md"
MACHINES_DEFAULT = "fan,valve"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage27AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 12
    epochs: int = 3
    lr: float = 2e-6
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    segment_len: int = 4
    segment_stride: int = 2
    patch_time_frac: float = 0.08
    patch_freq_frac: float = 0.10
    hidden_dim: int = 128
    boundary_loss_weight: float = 1.0
    clean_anchor_loss_weight: float = 0.20
    nonoverlap_consistency_loss_weight: float = 0.20
    positive_loss_weight: float = 4.0


def parse_args():
    p = argparse.ArgumentParser(description="Stage27A TF-patch representation-level pilot")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_tfpatch_representation"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--segment_len", type=int, default=4)
    p.add_argument("--segment_stride", type=int, default=2)
    p.add_argument("--patch_time_frac", type=float, default=0.08)
    p.add_argument("--patch_freq_frac", type=float, default=0.10)
    p.add_argument("--boundary_loss_weight", type=float, default=1.0)
    p.add_argument("--clean_anchor_loss_weight", type=float, default=0.20)
    p.add_argument("--nonoverlap_consistency_loss_weight", type=float, default=0.20)
    p.add_argument("--positive_loss_weight", type=float, default=4.0)
    return p.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
        for key in row:
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage27AConfig:
    return Stage27AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        segment_len=args.segment_len,
        segment_stride=args.segment_stride,
        patch_time_frac=args.patch_time_frac,
        patch_freq_frac=args.patch_freq_frac,
        boundary_loss_weight=args.boundary_loss_weight,
        clean_anchor_loss_weight=args.clean_anchor_loss_weight,
        nonoverlap_consistency_loss_weight=args.nonoverlap_consistency_loss_weight,
        positive_loss_weight=args.positive_loss_weight,
    )


def parameter_audit_rows_no_head(model, head) -> tuple[list[dict], list[dict]]:
    trainable_rows = []
    frozen_rows = []
    for scope, module in [("eat", model), ("boundary_head", head)]:
        for name, param in module.named_parameters():
            row = {
                "parameter_name": f"{scope}.{name}",
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
            }
            if param.requires_grad:
                trainable_rows.append(row)
            else:
                frozen_rows.append(row)
    return trainable_rows, frozen_rows


def summarize_trainable(trainable_rows: list[dict], frozen_rows: list[dict], allowed_block_names: list[str]) -> dict:
    """Stage27A allows the TF-patch boundary head in addition to the last EAT blocks."""
    summary = base_summarize_trainable(trainable_rows, frozen_rows, allowed_block_names)
    filtered_unexpected = [
        name
        for name in summary["unexpected_trainable_parameter_names"]
        if not name.startswith("boundary_head.")
    ]

    scope_counts = {}
    for row in trainable_rows:
        name = row["parameter_name"]
        if name.startswith("eat.model.blocks."):
            prefix = ".".join(name.split(".")[:4])
        elif name.startswith("boundary_head."):
            prefix = "boundary_head"
        else:
            prefix = "other"
        scope_counts[prefix] = scope_counts.get(prefix, 0) + 1

    summary["trainable_scope_counts"] = scope_counts
    summary["no_unexpected_earlier_layer_trainable_params"] = len(filtered_unexpected) == 0
    summary["unexpected_trainable_parameter_names"] = filtered_unexpected
    return summary


class LocalBoundaryHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_grouped_rows(rows: list[dict], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_rows = []
    val_rows = []
    machines = sorted({r["machine"] for r in rows})
    for machine in machines:
        rr = [r for r in rows if r["machine"] == machine]
        idx = rng.permutation(len(rr))
        val_n = max(1, int(len(rr) * val_ratio))
        val_n = min(val_n, len(rr) - 1)
        val_idx = set(int(i) for i in idx[:val_n])
        for i, row in enumerate(rr):
            (val_rows if i in val_idx else train_rows).append(row)
    return train_rows, val_rows


def load_mel(path: str, device):
    wav = load_waveform_16k(Path(path))
    return waveform_to_eat_tensor(wav).to(device)


def batch_rows_to_mel(rows: list[dict], device):
    return torch.stack([load_mel(row["path"], device) for row in rows], dim=0)


def make_pseudo_mel_batch(clean_mel, rng: np.random.Generator, patch_time_frac: float, patch_freq_frac: float):
    bsz, t, f = clean_mel.shape
    donor_idx = np.arange(bsz)
    rng.shuffle(donor_idx)
    if bsz > 1:
        for i in range(bsz):
            if donor_idx[i] == i:
                donor_idx[i] = (i + 1) % bsz
    pseudo = clean_mel.clone()
    t_len = max(1, int(t * patch_time_frac))
    f_len = max(1, int(f * patch_freq_frac))
    patch_rows = []
    for i in range(bsz):
        t0 = int(rng.integers(0, t - t_len + 1))
        f0 = int(rng.integers(0, f - f_len + 1))
        pseudo[i, t0:t0 + t_len, f0:f0 + f_len] = clean_mel[int(donor_idx[i]), t0:t0 + t_len, f0:f0 + f_len]
        patch_rows.append({"t0": t0, "t1": t0 + t_len, "f0": f0, "f1": f0 + f_len})
    return pseudo, patch_rows


def min_clamp_torch(frames):
    return torch.where(torch.abs(frames) < 0.1, torch.zeros_like(frames), frames)


def chunk_frames_with_patch_labels(frames, patch_rows, mel_t: int, segment_len: int, stride: int):
    chunks = []
    labels = []
    bsz, out_t, _ = frames.shape
    for b in range(bsz):
        patch = patch_rows[b]
        out_p0 = int(np.floor(patch["t0"] / mel_t * out_t))
        out_p1 = int(np.ceil(patch["t1"] / mel_t * out_t))
        out_p1 = max(out_p0 + 1, min(out_t, out_p1))
        starts = list(range(0, max(1, out_t - segment_len + 1), stride))
        if not starts or starts[-1] != max(0, out_t - segment_len):
            starts.append(max(0, out_t - segment_len))
        for start in starts:
            end = min(out_t, start + segment_len)
            chunks.append(frames[b, start:end].mean(dim=0))
            overlap = max(0, min(end, out_p1) - max(start, out_p0))
            labels.append(1.0 if overlap > 0 else 0.0)
    return torch.stack(chunks, dim=0), torch.tensor(labels, dtype=torch.float32, device=frames.device)


def chunk_frames_no_labels(frames, segment_len: int, stride: int):
    chunks = []
    bsz, out_t, _ = frames.shape
    for b in range(bsz):
        starts = list(range(0, max(1, out_t - segment_len + 1), stride))
        if not starts or starts[-1] != max(0, out_t - segment_len):
            starts.append(max(0, out_t - segment_len))
        for start in starts:
            end = min(out_t, start + segment_len)
            chunks.append(frames[b, start:end].mean(dim=0))
    return torch.stack(chunks, dim=0)


def clip_backend_pool(frames):
    return F.normalize(min_clamp_torch(frames).mean(dim=1), dim=1)


def compute_loss(student, teacher, head, clean_mel, rng: np.random.Generator, cfg: Stage27AConfig, device):
    pseudo_mel, patch_rows = make_pseudo_mel_batch(clean_mel, rng, cfg.patch_time_frac, cfg.patch_freq_frac)

    with torch.no_grad():
        teacher_clean_frames = min_clamp_torch(extract_frame_output(teacher, clean_mel, device))
        teacher_clean_chunks = chunk_frames_no_labels(teacher_clean_frames, cfg.segment_len, cfg.segment_stride)
        teacher_clip = clip_backend_pool(teacher_clean_frames)

    student_clean_frames = min_clamp_torch(extract_frame_output(student, clean_mel, device))
    student_pseudo_frames = min_clamp_torch(extract_frame_output(student, pseudo_mel, device))
    student_clean_chunks = chunk_frames_no_labels(student_clean_frames, cfg.segment_len, cfg.segment_stride)
    student_pseudo_chunks, pseudo_labels = chunk_frames_with_patch_labels(
        student_pseudo_frames,
        patch_rows,
        mel_t=clean_mel.shape[1],
        segment_len=cfg.segment_len,
        stride=cfg.segment_stride,
    )

    clean_labels = torch.zeros(student_clean_chunks.shape[0], dtype=torch.float32, device=device)
    boundary_x = torch.cat([student_clean_chunks, student_pseudo_chunks], dim=0)
    boundary_y = torch.cat([clean_labels, pseudo_labels], dim=0)
    pos_weight = torch.tensor(cfg.positive_loss_weight, dtype=torch.float32, device=device)
    boundary_loss = F.binary_cross_entropy_with_logits(head(boundary_x), boundary_y, pos_weight=pos_weight)

    student_clip = clip_backend_pool(student_clean_frames)
    clean_anchor = (1.0 - F.cosine_similarity(student_clip, teacher_clip, dim=1)).mean()

    nonoverlap_mask = pseudo_labels < 0.5
    if torch.any(nonoverlap_mask):
        nonoverlap_consistency = (
            1.0 - F.cosine_similarity(
                F.normalize(student_pseudo_chunks[nonoverlap_mask], dim=1),
                F.normalize(teacher_clean_chunks[nonoverlap_mask].detach(), dim=1),
                dim=1,
            )
        ).mean()
    else:
        nonoverlap_consistency = torch.zeros([], device=device)

    total = (
        cfg.boundary_loss_weight * boundary_loss
        + cfg.clean_anchor_loss_weight * clean_anchor
        + cfg.nonoverlap_consistency_loss_weight * nonoverlap_consistency
    )
    return {
        "total": total,
        "boundary_loss": boundary_loss,
        "clean_anchor_loss": clean_anchor,
        "nonoverlap_consistency_loss": nonoverlap_consistency,
        "positive_chunk_fraction": pseudo_labels.mean().detach(),
    }


def load_train_rows(data_root: Path, machines: list[str]) -> list[dict]:
    items = scan_dataset(data_root)
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine)
        train_items, _ = split_train_test(machine_items, use_target_train=False)
        for item in train_items:
            rows.append({"machine": machine, "path": str(item.path)})
    return rows


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine) if items else []
        train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
        rows.append({"machine": machine, "source_train_normal_count": len(train_items), "test_count": len(test_items)})
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "machines": machines,
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage27A saves an adapted EAT checkpoint for existing frozen backend evaluation.",
        ],
    })


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    model, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(model)
    allowed_blocks = enable_last_n_blocks(model, cfg.trainable_last_blocks)
    head = LocalBoundaryHead(768, cfg.hidden_dim).to(device)
    trainable_rows, frozen_rows = parameter_audit_rows_no_head(model, head)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)


def run_train_tfpatch_representation(args):
    set_seed(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    machines = parse_csv_list(args.machines)
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machines: {machines}")

    train_rows_all = load_train_rows(Path(args.data_root), machines)
    train_rows, val_rows = split_grouped_rows(train_rows_all, cfg.val_ratio, args.seed)
    write_csv(out_root / "train_val_inventory.csv", [
        {**row, "split_assignment": "train"} for row in train_rows
    ] + [
        {**row, "split_assignment": "val"} for row in val_rows
    ])

    teacher, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(teacher)
    teacher.eval()
    student, _ = load_clean_eat_local(Path(args.eat_model_path), str(device))
    freeze_all_parameters(student)
    allowed_blocks = enable_last_n_blocks(student, cfg.trainable_last_blocks)
    head = LocalBoundaryHead(768, cfg.hidden_dim).to(device)

    trainable_rows, frozen_rows = parameter_audit_rows_no_head(student, head)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    if not summary["no_unexpected_earlier_layer_trainable_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_parameter_names']}")
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)

    student_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": student_params, "lr": cfg.lr},
            {"params": head.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    best_path = out_root / "checkpoints" / "stage27a_tfpatch_representation_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, cfg.epochs + 1):
        student.train()
        head.train()
        order = rng.permutation(len(train_rows))
        train_sums = {k: 0.0 for k in ["total", "boundary_loss", "clean_anchor_loss", "nonoverlap_consistency_loss", "positive_chunk_fraction"]}
        train_count = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + cfg.batch_size]]
            clean_mel = batch_rows_to_mel(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_loss(student, teacher, head, clean_mel, rng, cfg, device)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(student_params + list(head.parameters()), cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
            for key in train_sums:
                train_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
            train_count += bsz

        student.eval()
        head.eval()
        val_sums = {k: 0.0 for k in train_sums}
        val_count = 0
        with torch.no_grad():
            for start in range(0, len(val_rows), cfg.batch_size):
                batch = val_rows[start:start + cfg.batch_size]
                clean_mel = batch_rows_to_mel(batch, device)
                loss_dict = compute_loss(student, teacher, head, clean_mel, rng, cfg, device)
                bsz = len(batch)
                for key in val_sums:
                    val_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
                val_count += bsz

        row = {"epoch": epoch}
        for key in train_sums:
            row[f"train_{key}"] = train_sums[key] / max(train_count, 1)
            row[f"val_{key}"] = val_sums[key] / max(val_count, 1)
        log_rows.append(row)
        val_loss = row["val_total"]
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "student_state_dict": student.state_dict(),
                "boundary_head_state_dict": head.state_dict(),
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "allowed_block_names": allowed_blocks,
                "trainable_summary": summary,
                "machines": machines,
                "stage": "stage27a_tfpatch_representation_pilot",
                "uses_anomaly_labels": False,
                "uses_test_labels_for_checkpoint_selection": False,
                "ldn_parameters_tuned": False,
            }, best_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage27a_train_config.json", asdict(cfg))
    write_json(out_root / "stage27a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Representation-level TF-patch pilot.",
            "Boundary head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage27A TF-Patch Representation Pilot",
        "",
        f"- machines: `{','.join(machines)}`",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation loss: `{best_val}`",
        "",
        "No ASD scoring was performed in this stage. Evaluate fan/valve first with the frozen adapted backend.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_trainable":
        run_inspect_trainable(args)
        return
    if args.mode == "train_tfpatch_representation":
        run_train_tfpatch_representation(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
