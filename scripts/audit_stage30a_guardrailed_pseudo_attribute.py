#!/usr/bin/env python
"""Stage30A: guardrailed pseudo-attribute representation pilot.

Stage29A showed that pseudo-attribute/domain-adaptive representation improves
full7 hmean/pAUC/TPR@99n, but it also moves the representation strongly and
regresses fan. Stage30A keeps the same pseudo-attribute inventory and adds
normal-only geometry guardrails:

- stronger clean-teacher clip anchor
- batch pairwise cosine-geometry preservation
- same-machine batch-center preservation

The attribute head is training-only. The saved checkpoint remains compatible
with Stage22D frozen backend evaluation via `student_state_dict`.

No anomaly labels, no test-label checkpoint selection, no selector, no fusion,
and no LDN parameter tuning.
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


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
OUTPUT_ROOT_DEFAULT = "results/stage30a_guardrailed_pseudo_attribute"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE30A_GUARDRAILED_PSEUDO_ATTRIBUTE.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage30AConfig:
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
    logit_scale: float = 20.0
    arc_margin: float = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Stage30A guardrailed pseudo-attribute representation")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_guardrailed_pseudo_attribute"])
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
    p.add_argument("--logit_scale", type=float, default=20.0)
    p.add_argument("--arc_margin", type=float, default=0.10)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage30AConfig:
    return Stage30AConfig(
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
        logit_scale=args.logit_scale,
        arc_margin=args.arc_margin,
    )


def machine_to_id_from_rows(rows: list[dict]) -> dict[str, int]:
    return {m: i for i, m in enumerate(sorted({r["machine"] for r in rows}))}


def compute_pairwise_geometry_loss(student_clip: torch.Tensor, teacher_clip: torch.Tensor) -> torch.Tensor:
    if student_clip.shape[0] <= 1:
        return torch.zeros([], device=student_clip.device)
    s = F.normalize(student_clip, dim=1)
    t = F.normalize(teacher_clip, dim=1)
    s_sim = s @ s.t()
    t_sim = t @ t.t()
    mask = ~torch.eye(s_sim.shape[0], dtype=torch.bool, device=s_sim.device)
    return F.mse_loss(s_sim[mask], t_sim[mask])


def compute_machine_center_loss(student_clip: torch.Tensor, teacher_clip: torch.Tensor, machine_ids: torch.Tensor) -> torch.Tensor:
    losses = []
    for mid in torch.unique(machine_ids):
        mask = machine_ids == mid
        if int(mask.sum().item()) < 2:
            continue
        s_center = F.normalize(student_clip[mask].mean(dim=0, keepdim=True), dim=1)
        t_center = F.normalize(teacher_clip[mask].mean(dim=0, keepdim=True), dim=1)
        losses.append(1.0 - F.cosine_similarity(s_center, t_center, dim=1).mean())
    if not losses:
        return torch.zeros([], device=student_clip.device)
    return torch.stack(losses).mean()


def compute_loss(student, teacher, head, mel_batch, labels, machine_ids, cfg: Stage30AConfig, device):
    with torch.no_grad():
        teacher_frames = extract_frame_output(teacher, mel_batch, device)
        teacher_clip = clean_clip_pool_torch(teacher_frames)
    student_frames = extract_frame_output(student, mel_batch, device)
    student_clip = clean_clip_pool_torch(student_frames)

    logits = head(student_clip, labels)
    ce = F.cross_entropy(logits, labels)
    anchor = (1.0 - F.cosine_similarity(
        F.normalize(student_clip, dim=1),
        F.normalize(teacher_clip, dim=1),
        dim=1,
    )).mean()
    pairwise = compute_pairwise_geometry_loss(student_clip, teacher_clip)
    machine_center = compute_machine_center_loss(student_clip, teacher_clip, machine_ids)
    total = (
        cfg.ce_loss_weight * ce
        + cfg.anchor_loss_weight * anchor
        + cfg.pairwise_geometry_loss_weight * pairwise
        + cfg.machine_center_loss_weight * machine_center
    )
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return {
        "total": total,
        "ce_loss": ce,
        "anchor_loss": anchor,
        "pairwise_geometry_loss": pairwise,
        "machine_center_loss": machine_center,
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
        inv_count = sum(1 for r in inventory if r.get("machine") == machine)
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(train_items),
            "pseudo_inventory_count": inv_count,
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
        "pseudo_attribute_count": len({r["pseudo_attribute_id"] for r in inventory}) if inventory else 0,
        "machines": machines,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage30A reuses Stage29 pseudo attributes and adds normal-only geometry guardrails.",
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


def run_train_guardrailed_pseudo_attribute(args):
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

    best_path = out_root / "checkpoints" / "stage30a_guardrailed_pseudo_attribute_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    loss_keys = ["total", "ce_loss", "anchor_loss", "pairwise_geometry_loss", "machine_center_loss", "acc"]
    for epoch in range(1, cfg.epochs + 1):
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
            loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, cfg, device)
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
                loss_dict = compute_loss(student, teacher, head, mel, labels, machine_ids, cfg, device)
                bsz = len(batch)
                for key in loss_keys:
                    val_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
                val_count += bsz

        row = {"epoch": epoch}
        for key in loss_keys:
            row[f"train_{key}"] = train_sums[key] / max(train_count, 1)
            row[f"val_{key}"] = val_sums[key] / max(val_count, 1)
        log_rows.append(row)
        val_loss = row["val_total"]
        if val_loss < best_val:
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
                "stage": "stage30a_guardrailed_pseudo_attribute",
                "uses_anomaly_labels": False,
                "uses_test_labels_for_checkpoint_selection": False,
                "ldn_parameters_tuned": False,
            }, best_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage30a_train_config.json", asdict(cfg))
    write_json(out_root / "stage30a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "num_pseudo_attributes": num_classes,
        "pseudo_inventory": str(inventory_path),
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Guardrailed pseudo-attribute representation pilot.",
            "Attribute head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage30A Guardrailed Pseudo-Attribute Representation",
        "",
        f"- machines: `{','.join(machines)}`",
        f"- pseudo attributes: `{num_classes}`",
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
    if args.mode == "train_guardrailed_pseudo_attribute":
        run_train_guardrailed_pseudo_attribute(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
