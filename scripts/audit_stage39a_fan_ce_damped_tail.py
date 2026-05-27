#!/usr/bin/env python
"""Stage39A: fan-CE-damped guardrail-constrained tail training.

Stage38A protected fan from the tail loss and slightly improved the Stage37
hmean/tail tradeoff, but fan hmean still lagged Stage30. Stage39A keeps the
Stage38 structure and makes fan protection more direct:

- fan samples receive a smaller pseudo-attribute CE weight
- fan gets a stronger clean-teacher anchor/center guard
- high-tail kNN dispersion remains excluded from fan by default
- checkpoint selection remains normal-validation guardrail constrained

This is still one adapted EAT checkpoint, not a selector or fusion route.
No anomaly labels, test-label checkpoint selection, or LDN tuning are used.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, replace
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
    ArcAttributeHead,
    batch_rows_to_mel,
    clean_clip_pool_torch,
    import_status,
    parameter_audit_rows,
    read_csv,
    split_stratified,
    summarize_trainable,
    write_csv,
    write_json,
    write_markdown,
)
from audit_stage30a_guardrailed_pseudo_attribute import (
    compute_machine_center_loss,
    compute_pairwise_geometry_loss,
    machine_to_id_from_rows,
)
from audit_stage38a_fan_protected_tail import (
    compute_fan_guard_loss,
    compute_tail_loss_maybe_excluding_fan,
)


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
OUTPUT_ROOT_DEFAULT = "results/stage39a_fan_ce_damped_tail"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE39A_FAN_CE_DAMPED_TAIL.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage39AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 64
    epochs: int = 5
    lr: float = 2e-6
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    ce_loss_weight: float = 1.0
    fan_ce_loss_weight: float = 0.25
    anchor_loss_weight: float = 0.50
    pairwise_geometry_loss_weight: float = 0.50
    machine_center_loss_weight: float = 0.25
    tail_dispersion_loss_weight: float = 0.05
    tail_start_epoch: int = 3
    tail_ramp_epochs: int = 2
    tail_knn_k: int = 3
    tail_fraction: float = 0.25
    logit_scale: float = 20.0
    arc_margin: float = 0.10
    fan_extra_guard_loss_weight: float = 0.50
    exclude_fan_from_tail: bool = True
    select_max_anchor_loss: float = 0.12
    select_max_pairwise_geometry_loss: float = 0.02
    select_max_machine_center_loss: float = 0.11


def parse_args():
    p = argparse.ArgumentParser(description="Stage39A fan-CE-damped tail training")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_fan_ce_damped_tail"])
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
    p.add_argument("--fan_ce_loss_weight", type=float, default=0.25)
    p.add_argument("--tail_dispersion_loss_weight", type=float, default=0.05)
    p.add_argument("--tail_start_epoch", type=int, default=3)
    p.add_argument("--tail_ramp_epochs", type=int, default=2)
    p.add_argument("--tail_knn_k", type=int, default=3)
    p.add_argument("--tail_fraction", type=float, default=0.25)
    p.add_argument("--fan_extra_guard_loss_weight", type=float, default=0.50)
    p.add_argument("--include_fan_in_tail", action="store_true")
    p.add_argument("--select_max_anchor_loss", type=float, default=0.12)
    p.add_argument("--select_max_pairwise_geometry_loss", type=float, default=0.02)
    p.add_argument("--select_max_machine_center_loss", type=float, default=0.11)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage39AConfig:
    return Stage39AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        fan_ce_loss_weight=args.fan_ce_loss_weight,
        tail_dispersion_loss_weight=args.tail_dispersion_loss_weight,
        tail_start_epoch=args.tail_start_epoch,
        tail_ramp_epochs=args.tail_ramp_epochs,
        tail_knn_k=args.tail_knn_k,
        tail_fraction=args.tail_fraction,
        fan_extra_guard_loss_weight=args.fan_extra_guard_loss_weight,
        exclude_fan_from_tail=not args.include_fan_in_tail,
        select_max_anchor_loss=args.select_max_anchor_loss,
        select_max_pairwise_geometry_loss=args.select_max_pairwise_geometry_loss,
        select_max_machine_center_loss=args.select_max_machine_center_loss,
    )


def tail_weight_for_epoch(cfg: Stage39AConfig, epoch: int) -> float:
    if epoch < cfg.tail_start_epoch:
        return 0.0
    if cfg.tail_ramp_epochs <= 1:
        return cfg.tail_dispersion_loss_weight
    progress = min(1.0, (epoch - cfg.tail_start_epoch + 1) / float(cfg.tail_ramp_epochs))
    return cfg.tail_dispersion_loss_weight * progress


def selection_eligible(row: dict, cfg: Stage39AConfig) -> bool:
    return (
        row["val_anchor_loss"] <= cfg.select_max_anchor_loss
        and row["val_pairwise_geometry_loss"] <= cfg.select_max_pairwise_geometry_loss
        and row["val_machine_center_loss"] <= cfg.select_max_machine_center_loss
    )


def compute_weighted_ce(logits, labels, machine_ids, cfg: Stage39AConfig, fan_machine_id: int | None):
    ce_per = F.cross_entropy(logits, labels, reduction="none")
    weights = torch.full_like(ce_per, cfg.ce_loss_weight)
    if fan_machine_id is not None:
        weights = torch.where(machine_ids == fan_machine_id, torch.full_like(weights, cfg.fan_ce_loss_weight), weights)
    return (ce_per * weights).mean(), ce_per.mean()


def compute_loss(student, teacher, head, mel_batch, labels, machine_ids, cfg: Stage39AConfig, device, fan_machine_id: int | None):
    with torch.no_grad():
        teacher_frames = extract_frame_output(teacher, mel_batch, device)
        teacher_clip = clean_clip_pool_torch(teacher_frames)
    student_frames = extract_frame_output(student, mel_batch, device)
    student_clip = clean_clip_pool_torch(student_frames)

    logits = head(student_clip, labels)
    weighted_ce, raw_ce = compute_weighted_ce(logits, labels, machine_ids, cfg, fan_machine_id)
    anchor = (1.0 - F.cosine_similarity(
        F.normalize(student_clip, dim=1),
        F.normalize(teacher_clip, dim=1),
        dim=1,
    )).mean()
    pairwise = compute_pairwise_geometry_loss(student_clip, teacher_clip)
    machine_center = compute_machine_center_loss(student_clip, teacher_clip, machine_ids)
    tail_dispersion, tail_dispersion_ratio = compute_tail_loss_maybe_excluding_fan(
        student_clip, teacher_clip, machine_ids, cfg, fan_machine_id
    )
    fan_anchor, fan_center = compute_fan_guard_loss(student_clip, teacher_clip, machine_ids, fan_machine_id)
    fan_guard = fan_anchor + fan_center
    total = (
        weighted_ce
        + cfg.anchor_loss_weight * anchor
        + cfg.pairwise_geometry_loss_weight * pairwise
        + cfg.machine_center_loss_weight * machine_center
        + cfg.tail_dispersion_loss_weight * tail_dispersion
        + cfg.fan_extra_guard_loss_weight * fan_guard
    )
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return {
        "total": total,
        "weighted_ce_loss": weighted_ce,
        "raw_ce_loss": raw_ce,
        "anchor_loss": anchor,
        "pairwise_geometry_loss": pairwise,
        "machine_center_loss": machine_center,
        "tail_dispersion_loss": tail_dispersion,
        "tail_dispersion_ratio": tail_dispersion_ratio,
        "fan_anchor_loss": fan_anchor,
        "fan_center_loss": fan_center,
        "fan_guard_loss": fan_guard,
        "acc": acc,
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
        "data_root_exists": data_root.exists(),
        "pseudo_inventory_exists": inventory_path.exists(),
        "pseudo_inventory_count": len(inventory),
        "machines": machines,
        "imports": import_status(),
        "notes": ["No model load, no forward, no training."],
    })


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


def checkpoint_payload(student, head, cfg, epoch, val_loss, allowed_blocks, summary, machines, machine_to_id, num_classes, inventory_path, eligible):
    return {
        "student_state_dict": student.state_dict(),
        "attribute_head_state_dict": head.state_dict(),
        "train_config": asdict(cfg),
        "best_epoch": epoch,
        "best_val_loss": val_loss,
        "selection_eligible": eligible,
        "allowed_block_names": allowed_blocks,
        "trainable_summary": summary,
        "machines": machines,
        "machine_to_id": machine_to_id,
        "num_pseudo_attributes": num_classes,
        "pseudo_inventory": str(inventory_path),
        "stage": "stage39a_fan_ce_damped_tail",
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
    }


def run_train_fan_ce_damped_tail(args):
    set_seed(args.seed)
    cfg = build_config(args)
    out_root = Path(args.output_root)
    ckpt_root = out_root / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    inventory_path = Path(args.pseudo_inventory)
    rows = read_csv(inventory_path)
    machines = sorted({r["machine"] for r in rows})
    machine_to_id = machine_to_id_from_rows(rows)
    fan_machine_id = machine_to_id.get("fan")
    num_classes = len({int(r["pseudo_attribute_id"]) for r in rows})
    train_rows, val_rows = split_stratified(rows, cfg.val_ratio, args.seed)
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

    best_path = ckpt_root / "stage39a_fan_ce_damped_tail_best.pt"
    best_eligible_val = float("inf")
    best_fallback_val = float("inf")
    best_epoch = -1
    best_selection_eligible = False
    log_rows = []
    rng = np.random.default_rng(args.seed)
    loss_keys = [
        "total",
        "weighted_ce_loss",
        "raw_ce_loss",
        "anchor_loss",
        "pairwise_geometry_loss",
        "machine_center_loss",
        "tail_dispersion_loss",
        "tail_dispersion_ratio",
        "fan_anchor_loss",
        "fan_center_loss",
        "fan_guard_loss",
        "acc",
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
            loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, epoch_cfg, device, fan_machine_id)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(student_params + list(head.parameters()), cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
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
                loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, epoch_cfg, device, fan_machine_id)
                bsz = len(batch)
                for key in loss_keys:
                    val_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
                val_count += bsz

        row = {"epoch": epoch, "tail_dispersion_loss_weight": epoch_tail_weight}
        for key in loss_keys:
            row[f"train_{key}"] = train_sums[key] / max(train_count, 1)
            row[f"val_{key}"] = val_sums[key] / max(val_count, 1)
        row["selection_eligible"] = selection_eligible(row, cfg)
        log_rows.append(row)

        payload = checkpoint_payload(
            student, head, cfg, epoch, row["val_total"], allowed_blocks, summary,
            machines, machine_to_id, num_classes, inventory_path, row["selection_eligible"],
        )
        torch.save(payload, ckpt_root / f"stage39a_fan_ce_damped_tail_epoch{epoch:03d}.pt")
        if row["selection_eligible"] and row["val_total"] < best_eligible_val:
            best_eligible_val = row["val_total"]
            best_epoch = epoch
            best_selection_eligible = True
            torch.save(payload, best_path)
        if not best_selection_eligible and row["val_total"] < best_fallback_val:
            best_fallback_val = row["val_total"]
            best_epoch = epoch
            torch.save(payload, best_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage39a_train_config.json", asdict(cfg))
    write_json(out_root / "stage39a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_selection_eligible": best_selection_eligible,
        "best_eligible_val_loss": best_eligible_val if best_selection_eligible else None,
        "best_fallback_val_loss": best_fallback_val,
        "machines": machines,
        "fan_machine_id": fan_machine_id,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "num_pseudo_attributes": num_classes,
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Stage38-like scheduled tail training with fan CE damped, fan excluded from tail loss, and stronger fan clean-teacher guard.",
            "Attribute head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage39A Fan-CE-Damped Tail",
        "",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best selection eligible: `{best_selection_eligible}`",
        f"- fan CE loss weight: `{cfg.fan_ce_loss_weight}`",
        f"- fan extra guard loss weight: `{cfg.fan_extra_guard_loss_weight}`",
        f"- exclude fan from tail: `{cfg.exclude_fan_from_tail}`",
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
    if args.mode == "train_fan_ce_damped_tail":
        run_train_fan_ce_damped_tail(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
