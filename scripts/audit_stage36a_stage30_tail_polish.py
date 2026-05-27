#!/usr/bin/env python
"""Stage36A: Stage30 checkpoint tail-polish pilot.

Stage34/35 showed that the normal-only tail objective improves pAUC/TPR@99n,
but training pseudo-attribute CE from the clean EAT start again caused large
geometry drift and kept hmean below Stage30. Stage36A starts from the Stage30
hmean-best checkpoint and applies only normal-label-free geometry polishing:

- clean-EAT teacher anchor
- batch pairwise cosine-geometry preservation
- same-machine batch-center preservation
- mild high-tail same-machine kNN dispersion preservation

There is no pseudo-attribute CE/head in this stage. Checkpoint selection uses
normal-only validation loss and writes a Stage22D-compatible `student_state_dict`.
No anomaly labels, test-label checkpoint selection, selector/fusion, or LDN
parameter tuning are used.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21e_adapted_eval import extract_frame_output, load_clean_eat_local
from audit_stage21d_global_dap_ssl_sanity import enable_last_n_blocks, freeze_all_parameters
from audit_stage29a_pseudo_attribute_representation import (
    batch_rows_to_mel,
    import_status,
    read_csv,
    split_stratified,
    write_csv,
    write_json,
    write_markdown,
)
from audit_stage30a_guardrailed_pseudo_attribute import (
    compute_machine_center_loss,
    compute_pairwise_geometry_loss,
    machine_to_id_from_rows,
)
from audit_stage34a_tail_guardrailed_pseudo_attribute import compute_tail_dispersion_loss


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
INIT_CKPT_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute/checkpoints/stage30a_guardrailed_pseudo_attribute_best.pt"
OUTPUT_ROOT_DEFAULT = "results/stage36a_stage30_tail_polish"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE36A_STAGE30_TAIL_POLISH.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage36AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 64
    epochs: int = 3
    lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    anchor_loss_weight: float = 0.50
    pairwise_geometry_loss_weight: float = 0.50
    machine_center_loss_weight: float = 0.25
    tail_dispersion_loss_weight: float = 0.03
    tail_knn_k: int = 3
    tail_fraction: float = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="Stage36A Stage30 checkpoint tail polish")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_stage30_tail_polish"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--init_ckpt_path", default=INIT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--pseudo_inventory", default=PSEUDO_INVENTORY_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--anchor_loss_weight", type=float, default=0.50)
    p.add_argument("--pairwise_geometry_loss_weight", type=float, default=0.50)
    p.add_argument("--machine_center_loss_weight", type=float, default=0.25)
    p.add_argument("--tail_dispersion_loss_weight", type=float, default=0.03)
    p.add_argument("--tail_knn_k", type=int, default=3)
    p.add_argument("--tail_fraction", type=float, default=0.25)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage36AConfig:
    return Stage36AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        anchor_loss_weight=args.anchor_loss_weight,
        pairwise_geometry_loss_weight=args.pairwise_geometry_loss_weight,
        machine_center_loss_weight=args.machine_center_loss_weight,
        tail_dispersion_loss_weight=args.tail_dispersion_loss_weight,
        tail_knn_k=args.tail_knn_k,
        tail_fraction=args.tail_fraction,
    )


def write_trainable_audit(path: Path, model, allowed_blocks: list[str]):
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
    allowed = tuple(allowed_blocks)
    unexpected = [
        row["parameter_name"] for row in trainable_rows
        if allowed and not row["parameter_name"].startswith(allowed)
    ]
    summary = {
        "trainable_parameter_count": int(sum(int(r["numel"]) for r in trainable_rows)),
        "frozen_parameter_count": int(sum(int(r["numel"]) for r in frozen_rows)),
        "trainable_tensor_count": len(trainable_rows),
        "frozen_tensor_count": len(frozen_rows),
        "allowed_block_names": allowed_blocks,
        "unexpected_trainable_parameter_names": unexpected,
        "no_unexpected_earlier_layer_trainable_params": len(unexpected) == 0,
    }
    write_csv(path / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(path / "frozen_parameter_audit.csv", frozen_rows)
    write_json(path / "trainable_summary.json", summary)
    return summary


def load_student_from_checkpoint(eat_path: Path, ckpt_path: Path, device_arg: str):
    model, device = load_clean_eat_local(eat_path, device_arg)
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("student_state_dict", blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load init checkpoint cleanly: missing={missing}, unexpected={unexpected}")
    return model, device, blob


def compute_polish_loss(student, teacher, mel_batch, machine_ids, cfg: Stage36AConfig, device):
    with torch.no_grad():
        teacher_frames = extract_frame_output(teacher, mel_batch, device)
        teacher_clip = teacher_frames.where(torch.abs(teacher_frames) >= 0.1, torch.zeros_like(teacher_frames)).mean(dim=1)
    student_frames = extract_frame_output(student, mel_batch, device)
    student_clip = student_frames.where(torch.abs(student_frames) >= 0.1, torch.zeros_like(student_frames)).mean(dim=1)
    anchor = (1.0 - F.cosine_similarity(
        F.normalize(student_clip, dim=1),
        F.normalize(teacher_clip, dim=1),
        dim=1,
    )).mean()
    pairwise = compute_pairwise_geometry_loss(student_clip, teacher_clip)
    machine_center = compute_machine_center_loss(student_clip, teacher_clip, machine_ids)
    tail_dispersion, tail_dispersion_ratio = compute_tail_dispersion_loss(
        student_clip,
        teacher_clip,
        machine_ids,
        cfg.tail_knn_k,
        cfg.tail_fraction,
    )
    total = (
        cfg.anchor_loss_weight * anchor
        + cfg.pairwise_geometry_loss_weight * pairwise
        + cfg.machine_center_loss_weight * machine_center
        + cfg.tail_dispersion_loss_weight * tail_dispersion
    )
    return {
        "total": total,
        "anchor_loss": anchor,
        "pairwise_geometry_loss": pairwise,
        "machine_center_loss": machine_center,
        "tail_dispersion_loss": tail_dispersion,
        "tail_dispersion_ratio": tail_dispersion_ratio,
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    inventory_path = Path(args.pseudo_inventory)
    inventory = read_csv(inventory_path) if inventory_path.exists() else []
    machines = sorted({r["machine"] for r in inventory}) if inventory else MACHINES_ALL
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine) if items else []
        train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(train_items),
            "pseudo_inventory_count": sum(1 for r in inventory if r.get("machine") == machine),
            "test_count": len(test_items),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "init_ckpt_exists": Path(args.init_ckpt_path).exists(),
        "data_root_exists": data_root.exists(),
        "pseudo_inventory": str(inventory_path),
        "pseudo_inventory_exists": inventory_path.exists(),
        "pseudo_inventory_count": len(inventory),
        "machines": machines,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage36A starts from Stage30 checkpoint and applies normal-only tail polish.",
        ],
    })


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    model, _, _ = load_student_from_checkpoint(Path(args.eat_model_path), Path(args.init_ckpt_path), args.device)
    freeze_all_parameters(model)
    allowed_blocks = enable_last_n_blocks(model, cfg.trainable_last_blocks)
    summary = write_trainable_audit(out_root, model, allowed_blocks)
    if not summary["no_unexpected_earlier_layer_trainable_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_parameter_names']}")


def average_losses(rows, prefix: str, sums: dict[str, float], count: int):
    row = {}
    for key, value in sums.items():
        row[f"{prefix}_{key}"] = value / max(count, 1)
    return row


def evaluate_split(student, teacher, rows: list[dict], machine_to_id: dict[str, int], cfg: Stage36AConfig, device, loss_keys):
    sums = {k: 0.0 for k in loss_keys}
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), cfg.batch_size):
            batch = rows[start:start + cfg.batch_size]
            mel = batch_rows_to_mel(batch, device)
            machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in batch], dtype=torch.long, device=device)
            loss_dict = compute_polish_loss(student, teacher, mel, machine_ids, cfg, device)
            bsz = len(batch)
            for key in loss_keys:
                sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
            count += bsz
    return sums, count


def save_checkpoint(path: Path, student, cfg, epoch, val_loss, allowed_blocks, summary, machines, machine_to_id, init_blob, inventory_path):
    torch.save({
        "student_state_dict": student.state_dict(),
        "train_config": asdict(cfg),
        "best_epoch": epoch,
        "best_val_loss": val_loss,
        "allowed_block_names": allowed_blocks,
        "trainable_summary": summary,
        "machines": machines,
        "machine_to_id": machine_to_id,
        "num_pseudo_attributes": init_blob.get("num_pseudo_attributes"),
        "pseudo_inventory": str(inventory_path),
        "init_checkpoint_stage": init_blob.get("stage"),
        "init_checkpoint_best_epoch": init_blob.get("best_epoch"),
        "stage": "stage36a_stage30_tail_polish",
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "tail_objective": "Stage30-init normal-only high-tail kNN dispersion polish",
    }, path)


def run_train_stage30_tail_polish(args):
    set_seed(args.seed)
    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    inventory_path = Path(args.pseudo_inventory)
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing pseudo inventory: {inventory_path}")
    rows = read_csv(inventory_path)
    machines = sorted({r["machine"] for r in rows})
    machine_to_id = machine_to_id_from_rows(rows)
    train_rows, val_rows = split_stratified(rows, cfg.val_ratio, args.seed)
    write_csv(out_root / "train_val_inventory.csv", [
        {**row, "split_assignment": "train"} for row in train_rows
    ] + [
        {**row, "split_assignment": "val"} for row in val_rows
    ])

    teacher, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(teacher)
    teacher.eval()
    student, _, init_blob = load_student_from_checkpoint(Path(args.eat_model_path), Path(args.init_ckpt_path), str(device))
    freeze_all_parameters(student)
    allowed_blocks = enable_last_n_blocks(student, cfg.trainable_last_blocks)
    summary = write_trainable_audit(out_root, student, allowed_blocks)
    if not summary["no_unexpected_earlier_layer_trainable_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_parameter_names']}")

    student_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": student_params, "lr": cfg.lr}], weight_decay=cfg.weight_decay)

    best_path = ckpt_root / "stage36a_stage30_tail_polish_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    loss_keys = ["total", "anchor_loss", "pairwise_geometry_loss", "machine_center_loss", "tail_dispersion_loss", "tail_dispersion_ratio"]

    student.eval()
    val_sums, val_count = evaluate_split(student, teacher, val_rows, machine_to_id, cfg, device, loss_keys)
    epoch0 = {"epoch": 0}
    epoch0.update(average_losses(val_rows, "val", val_sums, val_count))
    log_rows.append(epoch0)
    best_val = epoch0["val_total"]
    best_epoch = 0
    save_checkpoint(best_path, student, cfg, best_epoch, best_val, allowed_blocks, summary, machines, machine_to_id, init_blob, inventory_path)
    save_checkpoint(ckpt_root / "stage36a_stage30_tail_polish_epoch000.pt", student, cfg, 0, epoch0["val_total"], allowed_blocks, summary, machines, machine_to_id, init_blob, inventory_path)

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        order = rng.permutation(len(train_rows))
        train_sums = {k: 0.0 for k in loss_keys}
        train_count = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + cfg.batch_size]]
            mel = batch_rows_to_mel(batch, device)
            machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_polish_loss(student, teacher, mel, machine_ids, cfg, device)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(student_params, cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
            for key in loss_keys:
                train_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
            train_count += bsz

        student.eval()
        val_sums, val_count = evaluate_split(student, teacher, val_rows, machine_to_id, cfg, device, loss_keys)
        row = {"epoch": epoch}
        row.update(average_losses(train_rows, "train", train_sums, train_count))
        row.update(average_losses(val_rows, "val", val_sums, val_count))
        log_rows.append(row)
        epoch_path = ckpt_root / f"stage36a_stage30_tail_polish_epoch{epoch:03d}.pt"
        save_checkpoint(epoch_path, student, cfg, epoch, row["val_total"], allowed_blocks, summary, machines, machine_to_id, init_blob, inventory_path)
        if row["val_total"] < best_val:
            best_val = row["val_total"]
            best_epoch = epoch
            save_checkpoint(best_path, student, cfg, best_epoch, best_val, allowed_blocks, summary, machines, machine_to_id, init_blob, inventory_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage36a_train_config.json", asdict(cfg))
    write_json(out_root / "stage36a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "init_checkpoint": args.init_ckpt_path,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "stage21e_compatible_checkpoint": True,
        "uses_pseudo_attribute_ce": False,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Stage30-init normal-only tail polish.",
            "No attribute head or CE is used; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage36A Stage30 Tail Polish",
        "",
        f"- init checkpoint: `{args.init_ckpt_path}`",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation loss: `{best_val}`",
        f"- tail dispersion loss weight: `{cfg.tail_dispersion_loss_weight}`",
        f"- tail kNN k: `{cfg.tail_knn_k}`",
        f"- tail fraction: `{cfg.tail_fraction}`",
        "",
        "No ASD scoring was performed in this stage. Evaluate with the frozen adapted backend.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_trainable":
        run_inspect_trainable(args)
        return
    if args.mode == "train_stage30_tail_polish":
        run_train_stage30_tail_polish(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
