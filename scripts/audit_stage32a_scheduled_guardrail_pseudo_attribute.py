#!/usr/bin/env python
"""Stage32A: scheduled guardrail pseudo-attribute representation.

Stage31 showed that uniformly lowering guardrails over-constrained the model
and did not recover tail. Stage32A changes training time structure instead:

- warmup phase: Stage29-like pseudo-attribute shaping with light anchor
- guard phase: Stage30-like geometry guardrails to pull representation back
- checkpoint selection starts only in the guard phase, using normal-val loss

The attribute head is training-only. The saved checkpoint is compatible with
Stage22D frozen backend evaluation via `student_state_dict`.

No anomaly labels, no test-label checkpoint selection, no selector, no fusion,
and no LDN parameter tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

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
from audit_stage30a_guardrailed_pseudo_attribute import (
    Stage30AConfig,
    compute_loss,
    machine_to_id_from_rows,
)


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
OUTPUT_ROOT_DEFAULT = "results/stage32a_scheduled_guardrail_pseudo_attribute"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE32A_SCHEDULED_GUARDRAIL_PSEUDO_ATTRIBUTE.md"


@dataclass
class Stage32AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 64
    epochs: int = 5
    warmup_epochs: int = 2
    lr: float = 3e-6
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    ce_loss_weight: float = 1.0
    warmup_anchor_loss_weight: float = 0.10
    warmup_pairwise_geometry_loss_weight: float = 0.0
    warmup_machine_center_loss_weight: float = 0.0
    guard_anchor_loss_weight: float = 0.50
    guard_pairwise_geometry_loss_weight: float = 0.50
    guard_machine_center_loss_weight: float = 0.25
    logit_scale: float = 20.0
    arc_margin: float = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Stage32A scheduled guardrail pseudo-attribute representation")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_scheduled_guardrail"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--pseudo_inventory", default=PSEUDO_INVENTORY_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ce_loss_weight", type=float, default=1.0)
    p.add_argument("--warmup_anchor_loss_weight", type=float, default=0.10)
    p.add_argument("--warmup_pairwise_geometry_loss_weight", type=float, default=0.0)
    p.add_argument("--warmup_machine_center_loss_weight", type=float, default=0.0)
    p.add_argument("--guard_anchor_loss_weight", type=float, default=0.50)
    p.add_argument("--guard_pairwise_geometry_loss_weight", type=float, default=0.50)
    p.add_argument("--guard_machine_center_loss_weight", type=float, default=0.25)
    p.add_argument("--logit_scale", type=float, default=20.0)
    p.add_argument("--arc_margin", type=float, default=0.10)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage32AConfig:
    return Stage32AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        ce_loss_weight=args.ce_loss_weight,
        warmup_anchor_loss_weight=args.warmup_anchor_loss_weight,
        warmup_pairwise_geometry_loss_weight=args.warmup_pairwise_geometry_loss_weight,
        warmup_machine_center_loss_weight=args.warmup_machine_center_loss_weight,
        guard_anchor_loss_weight=args.guard_anchor_loss_weight,
        guard_pairwise_geometry_loss_weight=args.guard_pairwise_geometry_loss_weight,
        guard_machine_center_loss_weight=args.guard_machine_center_loss_weight,
        logit_scale=args.logit_scale,
        arc_margin=args.arc_margin,
    )


def stage30_cfg_for_epoch(cfg: Stage32AConfig, epoch: int) -> tuple[str, Stage30AConfig]:
    warmup = epoch <= cfg.warmup_epochs
    return (
        "warmup" if warmup else "guard",
        Stage30AConfig(
            trainable_last_blocks=cfg.trainable_last_blocks,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            head_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
            val_ratio=cfg.val_ratio,
            ce_loss_weight=cfg.ce_loss_weight,
            anchor_loss_weight=cfg.warmup_anchor_loss_weight if warmup else cfg.guard_anchor_loss_weight,
            pairwise_geometry_loss_weight=cfg.warmup_pairwise_geometry_loss_weight if warmup else cfg.guard_pairwise_geometry_loss_weight,
            machine_center_loss_weight=cfg.warmup_machine_center_loss_weight if warmup else cfg.guard_machine_center_loss_weight,
            logit_scale=cfg.logit_scale,
            arc_margin=cfg.arc_margin,
        ),
    )


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    inventory_path = Path(args.pseudo_inventory)
    inventory = read_csv(inventory_path) if inventory_path.exists() else []
    machines = sorted({r["machine"] for r in inventory})
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "pseudo_inventory": str(inventory_path),
        "pseudo_inventory_exists": inventory_path.exists(),
        "pseudo_inventory_count": len(inventory),
        "pseudo_attribute_count": len({r["pseudo_attribute_id"] for r in inventory}) if inventory else 0,
        "machines": machines,
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage32A uses a warmup-to-guard training schedule.",
        ],
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


def run_train_scheduled_guardrail(args):
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

    best_path = out_root / "checkpoints" / "stage32a_scheduled_guardrail_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    loss_keys = ["total", "ce_loss", "anchor_loss", "pairwise_geometry_loss", "machine_center_loss", "acc"]
    for epoch in range(1, cfg.epochs + 1):
        phase, epoch_cfg = stage30_cfg_for_epoch(cfg, epoch)
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
                loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, epoch_cfg, device)
                bsz = len(batch)
                for key in loss_keys:
                    val_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
                val_count += bsz

        row = {
            "epoch": epoch,
            "phase": phase,
            "anchor_loss_weight": epoch_cfg.anchor_loss_weight,
            "pairwise_geometry_loss_weight": epoch_cfg.pairwise_geometry_loss_weight,
            "machine_center_loss_weight": epoch_cfg.machine_center_loss_weight,
        }
        for key in loss_keys:
            row[f"train_{key}"] = train_sums[key] / max(train_count, 1)
            row[f"val_{key}"] = val_sums[key] / max(val_count, 1)
        log_rows.append(row)

        val_loss = row["val_total"]
        checkpoint_eligible = epoch > cfg.warmup_epochs
        if checkpoint_eligible and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "student_state_dict": student.state_dict(),
                "attribute_head_state_dict": head.state_dict(),
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "allowed_block_names": allowed_blocks,
                "trainable_summary": summary,
                "machines": machines,
                "machine_to_id": machine_to_id,
                "num_pseudo_attributes": num_classes,
                "pseudo_inventory": str(inventory_path),
                "stage": "stage32a_scheduled_guardrail_pseudo_attribute",
                "uses_anomaly_labels": False,
                "uses_test_labels_for_checkpoint_selection": False,
                "ldn_parameters_tuned": False,
                "checkpoint_selection": "normal_val_loss_guard_phase_only",
            }, best_path)

    if best_epoch < 0:
        raise RuntimeError("No guard-phase checkpoint was saved; increase epochs or reduce warmup_epochs.")

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage32a_train_config.json", asdict(cfg))
    write_json(out_root / "stage32a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "num_pseudo_attributes": num_classes,
        "pseudo_inventory": str(inventory_path),
        "warmup_epochs": cfg.warmup_epochs,
        "checkpoint_selection": "normal_val_loss_guard_phase_only",
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Scheduled guardrail pseudo-attribute representation pilot.",
            "Attribute head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage32A Scheduled Guardrail Pseudo-Attribute",
        "",
        f"- machines: `{','.join(machines)}`",
        f"- pseudo attributes: `{num_classes}`",
        f"- warmup epochs: `{cfg.warmup_epochs}`",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation loss: `{best_val}`",
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
    if args.mode == "train_scheduled_guardrail":
        run_train_scheduled_guardrail(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
