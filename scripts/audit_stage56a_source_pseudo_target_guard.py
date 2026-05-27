#!/usr/bin/env python
"""Stage56A: source pseudo-attribute training with target-normal guard.

Stage55 showed that adding target train-normal clips directly to the pseudo
attribute inventory hurts the Stage35 target-memory route. Stage56A keeps the
Stage35 source-only pseudo CE objective, but adds a small unlabeled guard on
target-domain train-normal clips. Target clips never receive pseudo labels and
never enter the attribute CE; they only anchor target-memory geometry to the
clean teacher.

It does not use anomaly labels, tune LDN parameters, select checkpoints by test
labels, or perform ASD scoring. The attribute head is training-only; all saved
checkpoints remain Stage21E/22D/48A-compatible through `student_state_dict`.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21e_adapted_eval import load_clean_eat_local
from audit_stage21d_global_dap_ssl_sanity import enable_last_n_blocks, freeze_all_parameters
from audit_stage29a_pseudo_attribute_representation import (
    ArcAttributeHead,
    batch_rows_to_mel,
    import_status,
    parameter_audit_rows,
    read_csv,
    split_stratified,
    summarize_trainable,
    write_csv,
    write_json,
    write_markdown,
)
from audit_stage30a_guardrailed_pseudo_attribute import machine_to_id_from_rows
from audit_stage34a_tail_guardrailed_pseudo_attribute import compute_loss, compute_tail_dispersion_loss
from audit_stage21e_adapted_eval import extract_frame_output
from audit_stage29a_pseudo_attribute_representation import clean_clip_pool_torch
from audit_stage30a_guardrailed_pseudo_attribute import (
    compute_machine_center_loss,
    compute_pairwise_geometry_loss,
)
import torch.nn.functional as F


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
OUTPUT_ROOT_DEFAULT = "results/stage56a_source_pseudo_target_guard"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE56A_SOURCE_PSEUDO_TARGET_GUARD.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage35AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 64
    epochs: int = 5
    lr: float = 2e-6
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    ce_loss_weight: float = 1.0
    anchor_loss_weight: float = 0.50
    pairwise_geometry_loss_weight: float = 0.50
    machine_center_loss_weight: float = 0.25
    tail_dispersion_loss_weight: float = 0.05
    tail_start_epoch: int = 3
    tail_ramp_epochs: int = 2
    tail_knn_k: int = 3
    tail_fraction: float = 0.25
    target_guard_loss_weight: float = 0.10
    target_anchor_loss_weight: float = 0.50
    target_pairwise_geometry_loss_weight: float = 0.25
    target_machine_center_loss_weight: float = 0.25
    target_tail_dispersion_loss_weight: float = 0.01
    target_batch_size: int = 32
    logit_scale: float = 20.0
    arc_margin: float = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Stage56A source pseudo training with target-normal guard")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_source_pseudo_target_guard"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--pseudo_inventory", default=PSEUDO_INVENTORY_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ce_loss_weight", type=float, default=1.0)
    p.add_argument("--anchor_loss_weight", type=float, default=0.50)
    p.add_argument("--pairwise_geometry_loss_weight", type=float, default=0.50)
    p.add_argument("--machine_center_loss_weight", type=float, default=0.25)
    p.add_argument("--tail_dispersion_loss_weight", type=float, default=0.05)
    p.add_argument("--tail_start_epoch", type=int, default=3)
    p.add_argument("--tail_ramp_epochs", type=int, default=2)
    p.add_argument("--tail_knn_k", type=int, default=3)
    p.add_argument("--tail_fraction", type=float, default=0.25)
    p.add_argument("--target_guard_loss_weight", type=float, default=0.10)
    p.add_argument("--target_anchor_loss_weight", type=float, default=0.50)
    p.add_argument("--target_pairwise_geometry_loss_weight", type=float, default=0.25)
    p.add_argument("--target_machine_center_loss_weight", type=float, default=0.25)
    p.add_argument("--target_tail_dispersion_loss_weight", type=float, default=0.01)
    p.add_argument("--target_batch_size", type=int, default=32)
    p.add_argument("--logit_scale", type=float, default=20.0)
    p.add_argument("--arc_margin", type=float, default=0.10)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage35AConfig:
    return Stage35AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        ce_loss_weight=args.ce_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        pairwise_geometry_loss_weight=args.pairwise_geometry_loss_weight,
        machine_center_loss_weight=args.machine_center_loss_weight,
        tail_dispersion_loss_weight=args.tail_dispersion_loss_weight,
        tail_start_epoch=args.tail_start_epoch,
        tail_ramp_epochs=args.tail_ramp_epochs,
        tail_knn_k=args.tail_knn_k,
        tail_fraction=args.tail_fraction,
        target_guard_loss_weight=args.target_guard_loss_weight,
        target_anchor_loss_weight=args.target_anchor_loss_weight,
        target_pairwise_geometry_loss_weight=args.target_pairwise_geometry_loss_weight,
        target_machine_center_loss_weight=args.target_machine_center_loss_weight,
        target_tail_dispersion_loss_weight=args.target_tail_dispersion_loss_weight,
        target_batch_size=args.target_batch_size,
        logit_scale=args.logit_scale,
        arc_margin=args.arc_margin,
    )


def tail_weight_for_epoch(cfg: Stage35AConfig, epoch: int) -> float:
    if epoch < cfg.tail_start_epoch:
        return 0.0
    if cfg.tail_ramp_epochs <= 1:
        return cfg.tail_dispersion_loss_weight
    progress = min(1.0, (epoch - cfg.tail_start_epoch + 1) / float(cfg.tail_ramp_epochs))
    return cfg.tail_dispersion_loss_weight * progress


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
        source_target_train, _ = split_train_test(machine_items, use_target_train=True) if machine_items else ([], [])
        target_train_count = sum(1 for item in source_target_train if item.domain == "target")
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(train_items),
            "target_train_normal_count": target_train_count,
            "pseudo_inventory_count": sum(1 for r in inventory if r.get("machine") == machine),
            "test_count": len(test_items),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "pseudo_inventory": str(inventory_path),
        "pseudo_inventory_exists": inventory_path.exists(),
        "pseudo_inventory_count": len(inventory),
        "target_guard_count": sum(row["target_train_normal_count"] for row in rows),
        "pseudo_attribute_count": len({r["pseudo_attribute_id"] for r in inventory}) if inventory else 0,
        "machines": machines,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage56A uses source-only pseudo CE plus target train-normal unlabeled guard.",
        ],
    })


def load_target_train_rows(data_root: Path, machines: list[str]) -> list[dict]:
    items = scan_dataset(data_root)
    target_rows = []
    for machine in machines:
        machine_items = filter_items(items, machine)
        train_items, _ = split_train_test(machine_items, use_target_train=True)
        for item in train_items:
            if item.domain != "target":
                continue
            target_rows.append({
                "machine": machine,
                "path": str(item.path),
                "domain": item.domain,
                "section": item.section,
            })
    return target_rows


def sample_target_batch(target_rows: list[dict], rng: np.random.Generator, batch_size: int) -> list[dict]:
    if not target_rows or batch_size <= 0:
        return []
    indices = rng.choice(len(target_rows), size=batch_size, replace=len(target_rows) < batch_size)
    return [target_rows[int(i)] for i in indices]


def compute_target_guard_loss(student, teacher, mel_batch, machine_ids, cfg: Stage35AConfig, device):
    if mel_batch.shape[0] == 0:
        zero = torch.zeros([], device=device)
        return {
            "target_total": zero,
            "target_anchor_loss": zero,
            "target_pairwise_geometry_loss": zero,
            "target_machine_center_loss": zero,
            "target_tail_dispersion_loss": zero,
            "target_tail_dispersion_ratio": zero,
        }
    with torch.no_grad():
        teacher_frames = extract_frame_output(teacher, mel_batch, device)
        teacher_clip = clean_clip_pool_torch(teacher_frames)
    student_frames = extract_frame_output(student, mel_batch, device)
    student_clip = clean_clip_pool_torch(student_frames)

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
        cfg.target_anchor_loss_weight * anchor
        + cfg.target_pairwise_geometry_loss_weight * pairwise
        + cfg.target_machine_center_loss_weight * machine_center
        + cfg.target_tail_dispersion_loss_weight * tail_dispersion
    )
    return {
        "target_total": total,
        "target_anchor_loss": anchor,
        "target_pairwise_geometry_loss": pairwise,
        "target_machine_center_loss": machine_center,
        "target_tail_dispersion_loss": tail_dispersion,
        "target_tail_dispersion_ratio": tail_dispersion_ratio,
    }


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    inventory = read_csv(Path(args.pseudo_inventory))
    num_classes = len({int(r["pseudo_attribute_id"]) for r in inventory})
    model, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(model)
    allowed_blocks = enable_last_n_blocks(model, cfg.trainable_last_blocks)
    head = ArcAttributeHead(768, num_classes, cfg.logit_scale, cfg.arc_margin).to(device)
    trainable_rows, frozen_rows = parameter_audit_rows(model, head)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)


def run_train_source_pseudo_target_guard(args):
    set_seed(args.seed)
    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    inventory_path = Path(args.pseudo_inventory)
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing pseudo inventory: {inventory_path}")
    rows = read_csv(inventory_path)
    machines = sorted({r["machine"] for r in rows})
    machine_to_id = machine_to_id_from_rows(rows)
    num_classes = len({int(r["pseudo_attribute_id"]) for r in rows})
    train_rows, val_rows = split_stratified(rows, cfg.val_ratio, args.seed)
    target_rows = load_target_train_rows(Path(args.data_root), machines)
    if not target_rows:
        raise RuntimeError("No target-domain train-normal rows found for Stage56 target guard.")
    write_csv(out_root / "train_val_inventory.csv", [
        {**row, "split_assignment": "train"} for row in train_rows
    ] + [
        {**row, "split_assignment": "val"} for row in val_rows
    ])
    write_csv(out_root / "target_guard_inventory.csv", target_rows)

    teacher, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(teacher)
    teacher.eval()
    student, _ = load_clean_eat_local(Path(args.eat_model_path), str(device))
    freeze_all_parameters(student)
    allowed_blocks = enable_last_n_blocks(student, cfg.trainable_last_blocks)
    head = ArcAttributeHead(768, num_classes, cfg.logit_scale, cfg.arc_margin).to(device)

    trainable_rows, frozen_rows = parameter_audit_rows(student, head)
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

    best_path = out_root / "checkpoints" / "stage56a_source_pseudo_target_guard_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    loss_keys = [
        "total",
        "ce_loss",
        "anchor_loss",
        "pairwise_geometry_loss",
        "machine_center_loss",
        "tail_dispersion_loss",
        "tail_dispersion_ratio",
        "acc",
        "target_total",
        "target_anchor_loss",
        "target_pairwise_geometry_loss",
        "target_machine_center_loss",
        "target_tail_dispersion_loss",
        "target_tail_dispersion_ratio",
    ]
    for epoch in range(1, cfg.epochs + 1):
        epoch_tail_weight = tail_weight_for_epoch(cfg, epoch)
        epoch_cfg = replace(cfg, tail_dispersion_loss_weight=epoch_tail_weight)
        student.train()
        head.train()
        order = rng.permutation(len(train_rows))
        train_sums = {k: 0.0 for k in loss_keys}
        train_count = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + cfg.batch_size]]
            mel = batch_rows_to_mel(batch, device)
            labels = torch.tensor([int(r["pseudo_attribute_id"]) for r in batch], dtype=torch.long, device=device)
            machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, epoch_cfg, device)
            target_batch = sample_target_batch(target_rows, rng, cfg.target_batch_size)
            if target_batch and cfg.target_guard_loss_weight > 0:
                target_mel = batch_rows_to_mel(target_batch, device)
                target_machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in target_batch], dtype=torch.long, device=device)
                target_loss = compute_target_guard_loss(student, teacher, target_mel, target_machine_ids, epoch_cfg, device)
            else:
                zero = torch.zeros([], device=device)
                target_loss = {
                    "target_total": zero,
                    "target_anchor_loss": zero,
                    "target_pairwise_geometry_loss": zero,
                    "target_machine_center_loss": zero,
                    "target_tail_dispersion_loss": zero,
                    "target_tail_dispersion_ratio": zero,
                }
            total = loss_dict["total"] + cfg.target_guard_loss_weight * target_loss["target_total"]
            total.backward()
            torch.nn.utils.clip_grad_norm_(student_params + list(head.parameters()), cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
            loss_dict = {**loss_dict, **target_loss}
            loss_dict["total"] = total
            for key in loss_keys:
                train_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
            train_count += bsz

        student.eval()
        head.eval()
        val_sums = {k: 0.0 for k in loss_keys}
        val_count = 0
        with torch.no_grad():
            for start in range(0, len(val_rows), cfg.batch_size):
                batch = val_rows[start:start + cfg.batch_size]
                mel = batch_rows_to_mel(batch, device)
                labels = torch.tensor([int(r["pseudo_attribute_id"]) for r in batch], dtype=torch.long, device=device)
                machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in batch], dtype=torch.long, device=device)
                loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, epoch_cfg, device)
                target_batch = sample_target_batch(target_rows, rng, cfg.target_batch_size)
                if target_batch and cfg.target_guard_loss_weight > 0:
                    target_mel = batch_rows_to_mel(target_batch, device)
                    target_machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in target_batch], dtype=torch.long, device=device)
                    target_loss = compute_target_guard_loss(student, teacher, target_mel, target_machine_ids, epoch_cfg, device)
                else:
                    zero = torch.zeros([], device=device)
                    target_loss = {
                        "target_total": zero,
                        "target_anchor_loss": zero,
                        "target_pairwise_geometry_loss": zero,
                        "target_machine_center_loss": zero,
                        "target_tail_dispersion_loss": zero,
                        "target_tail_dispersion_ratio": zero,
                    }
                total = loss_dict["total"] + cfg.target_guard_loss_weight * target_loss["target_total"]
                loss_dict = {**loss_dict, **target_loss}
                loss_dict["total"] = total
                bsz = len(batch)
                for key in loss_keys:
                    val_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
                val_count += bsz

        row = {"epoch": epoch, "tail_dispersion_loss_weight": epoch_tail_weight}
        for key in loss_keys:
            row[f"train_{key}"] = train_sums[key] / max(train_count, 1)
            row[f"val_{key}"] = val_sums[key] / max(val_count, 1)
        log_rows.append(row)
        val_loss = row["val_total"]
        epoch_path = out_root / "checkpoints" / f"stage56a_source_pseudo_target_guard_epoch{epoch:03d}.pt"
        checkpoint_blob = {
            "student_state_dict": student.state_dict(),
            "attribute_head_state_dict": head.state_dict(),
            "train_config": asdict(cfg),
            "epoch": epoch,
            "tail_dispersion_loss_weight_this_epoch": epoch_tail_weight,
            "val_loss": val_loss,
            "allowed_block_names": allowed_blocks,
            "trainable_summary": summary,
            "machines": machines,
            "machine_to_id": machine_to_id,
            "num_pseudo_attributes": num_classes,
            "pseudo_inventory": str(inventory_path),
            "stage": "stage56a_source_pseudo_target_guard",
            "uses_anomaly_labels": False,
            "uses_test_labels_for_checkpoint_selection": False,
            "ldn_parameters_tuned": False,
            "tail_objective": "Stage35 delayed mild same-machine teacher high-tail kNN dispersion preservation",
            "target_guard": "target train-normal unlabeled clean-teacher anchor/geometry/tail guard",
        }
        torch.save(checkpoint_blob, epoch_path)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_blob = dict(checkpoint_blob)
            best_blob["best_epoch"] = best_epoch
            best_blob["best_val_loss"] = best_val
            torch.save(best_blob, best_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage56a_train_config.json", asdict(cfg))
    write_json(out_root / "stage56a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epoch_checkpoints": [
            str(out_root / "checkpoints" / f"stage56a_source_pseudo_target_guard_epoch{epoch:03d}.pt")
            for epoch in range(1, cfg.epochs + 1)
        ],
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "target_guard_count": len(target_rows),
        "num_pseudo_attributes": num_classes,
        "pseudo_inventory": str(inventory_path),
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "tail_objective": "delayed mild same-machine teacher high-tail kNN dispersion preservation",
        "target_guard": "target train-normal unlabeled clean-teacher anchor/geometry/tail guard",
        "notes": [
            "Source-only pseudo CE with Stage35 scheduled tail plus target train-normal unlabeled guard.",
            "Attribute head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage56A Source Pseudo Target Guard",
        "",
        f"- machines: `{','.join(machines)}`",
        f"- pseudo attributes: `{num_classes}`",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation loss: `{best_val}`",
        f"- epoch checkpoints saved: `{cfg.epochs}`",
        f"- target guard train-normal clips: `{len(target_rows)}`",
        f"- target guard loss weight: `{cfg.target_guard_loss_weight}`",
        f"- max tail dispersion loss weight: `{cfg.tail_dispersion_loss_weight}`",
        f"- tail start epoch: `{cfg.tail_start_epoch}`",
        f"- tail ramp epochs: `{cfg.tail_ramp_epochs}`",
        f"- tail kNN k: `{cfg.tail_knn_k}`",
        f"- tail fraction: `{cfg.tail_fraction}`",
        "",
        "No ASD scoring was performed in this stage. Evaluate each epoch with the frozen adapted backend.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_trainable":
        run_inspect_trainable(args)
        return
    if args.mode == "train_source_pseudo_target_guard":
        run_train_source_pseudo_target_guard(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
