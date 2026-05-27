#!/usr/bin/env python
"""Stage37A: guardrail-constrained tail training.

Stage35A became the tail-best branch but selected a late checkpoint with large
geometry drift. Stage37A repeats the same mild scheduled-tail objective and
saves every epoch, but selects the best checkpoint only among epochs whose
normal-only guardrail losses remain near the Stage30 range.

Checkpoint selection is normal-validation only:
- no anomaly labels
- no test-label checkpoint selection
- no selector/fusion
- no LDN parameter tuning
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
from audit_stage34a_tail_guardrailed_pseudo_attribute import compute_loss


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
OUTPUT_ROOT_DEFAULT = "results/stage37a_guardrail_constrained_tail"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE37A_GUARDRAIL_CONSTRAINED_TAIL.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage37AConfig:
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
    logit_scale: float = 20.0
    arc_margin: float = 0.10
    select_max_anchor_loss: float = 0.12
    select_max_pairwise_geometry_loss: float = 0.02
    select_max_machine_center_loss: float = 0.11


def parse_args():
    p = argparse.ArgumentParser(description="Stage37A guardrail-constrained tail training")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_guardrail_constrained_tail"])
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
    p.add_argument("--tail_dispersion_loss_weight", type=float, default=0.05)
    p.add_argument("--tail_start_epoch", type=int, default=3)
    p.add_argument("--tail_ramp_epochs", type=int, default=2)
    p.add_argument("--tail_knn_k", type=int, default=3)
    p.add_argument("--tail_fraction", type=float, default=0.25)
    p.add_argument("--select_max_anchor_loss", type=float, default=0.12)
    p.add_argument("--select_max_pairwise_geometry_loss", type=float, default=0.02)
    p.add_argument("--select_max_machine_center_loss", type=float, default=0.11)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage37AConfig:
    return Stage37AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        tail_dispersion_loss_weight=args.tail_dispersion_loss_weight,
        tail_start_epoch=args.tail_start_epoch,
        tail_ramp_epochs=args.tail_ramp_epochs,
        tail_knn_k=args.tail_knn_k,
        tail_fraction=args.tail_fraction,
        select_max_anchor_loss=args.select_max_anchor_loss,
        select_max_pairwise_geometry_loss=args.select_max_pairwise_geometry_loss,
        select_max_machine_center_loss=args.select_max_machine_center_loss,
    )


def tail_weight_for_epoch(cfg: Stage37AConfig, epoch: int) -> float:
    if epoch < cfg.tail_start_epoch:
        return 0.0
    if cfg.tail_ramp_epochs <= 1:
        return cfg.tail_dispersion_loss_weight
    progress = min(1.0, (epoch - cfg.tail_start_epoch + 1) / float(cfg.tail_ramp_epochs))
    return cfg.tail_dispersion_loss_weight * progress


def selection_eligible(row: dict, cfg: Stage37AConfig) -> bool:
    return (
        row["val_anchor_loss"] <= cfg.select_max_anchor_loss
        and row["val_pairwise_geometry_loss"] <= cfg.select_max_pairwise_geometry_loss
        and row["val_machine_center_loss"] <= cfg.select_max_machine_center_loss
    )


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
        "stage": "stage37a_guardrail_constrained_tail",
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
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


def run_train_guardrail_constrained_tail(args):
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

    best_path = ckpt_root / "stage37a_guardrail_constrained_tail_best.pt"
    best_eligible_val = float("inf")
    best_fallback_val = float("inf")
    best_epoch = -1
    best_selection_eligible = False
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
        torch.save(payload, ckpt_root / f"stage37a_guardrail_constrained_tail_epoch{epoch:03d}.pt")
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
    write_json(out_root / "stage37a_train_config.json", asdict(cfg))
    write_json(out_root / "stage37a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_selection_eligible": best_selection_eligible,
        "best_eligible_val_loss": best_eligible_val if best_selection_eligible else None,
        "best_fallback_val_loss": best_fallback_val,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "num_pseudo_attributes": num_classes,
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Same scheduled mild-tail objective as Stage35, but checkpoint selection is constrained by normal-only Stage30-near guardrails.",
            "Attribute head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage37A Guardrail-Constrained Tail",
        "",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best selection eligible: `{best_selection_eligible}`",
        f"- max anchor loss: `{cfg.select_max_anchor_loss}`",
        f"- max pairwise geometry loss: `{cfg.select_max_pairwise_geometry_loss}`",
        f"- max machine center loss: `{cfg.select_max_machine_center_loss}`",
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
    if args.mode == "train_guardrail_constrained_tail":
        run_train_guardrail_constrained_tail(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
