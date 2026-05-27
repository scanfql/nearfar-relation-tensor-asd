#!/usr/bin/env python
"""Stage33A: fixed-prototype pseudo-attribute representation pilot.

Stage29/30 showed pseudo attributes are useful, but a trainable attribute head
can either move geometry too much or learn too little under guardrails. Stage33A
removes the trainable head:

- compute fixed clean-EAT prototypes for each pseudo attribute
- train only the last EAT block to align student clip embeddings to prototypes
- keep Stage30-style normal-only geometry guardrails
- save a Stage22D-compatible `student_state_dict`

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

from audit_stage21e_adapted_eval import extract_frame_output, load_clean_eat_local
from audit_stage21d_global_dap_ssl_sanity import (
    enable_last_n_blocks,
    freeze_all_parameters,
    summarize_trainable as base_summarize_trainable,
)
from audit_stage29a_pseudo_attribute_representation import (
    batch_rows_to_mel,
    clean_clip_pool_torch,
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


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
PSEUDO_INVENTORY_DEFAULT = "results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv"
PROTOTYPE_EMBEDDING_ROOT_DEFAULT = "results/stage29a_pseudo_attribute_representation"
OUTPUT_ROOT_DEFAULT = "results/stage33a_fixed_prototype_representation"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE33A_FIXED_PROTOTYPE_REPRESENTATION.md"
EPS = 1e-12


@dataclass
class Stage33AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 64
    epochs: int = 5
    lr: float = 2e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    prototype_ce_loss_weight: float = 1.0
    anchor_loss_weight: float = 0.50
    pairwise_geometry_loss_weight: float = 0.50
    machine_center_loss_weight: float = 0.25
    prototype_logit_scale: float = 20.0
    prototype_margin: float = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="Stage33A fixed-prototype pseudo-attribute representation")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_fixed_prototype"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--pseudo_inventory", default=PSEUDO_INVENTORY_DEFAULT)
    p.add_argument("--prototype_embedding_root", default=PROTOTYPE_EMBEDDING_ROOT_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--prototype_ce_loss_weight", type=float, default=1.0)
    p.add_argument("--anchor_loss_weight", type=float, default=0.50)
    p.add_argument("--pairwise_geometry_loss_weight", type=float, default=0.50)
    p.add_argument("--machine_center_loss_weight", type=float, default=0.25)
    p.add_argument("--prototype_logit_scale", type=float, default=20.0)
    p.add_argument("--prototype_margin", type=float, default=0.05)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args) -> Stage33AConfig:
    return Stage33AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        prototype_ce_loss_weight=args.prototype_ce_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        pairwise_geometry_loss_weight=args.pairwise_geometry_loss_weight,
        machine_center_loss_weight=args.machine_center_loss_weight,
        prototype_logit_scale=args.prototype_logit_scale,
        prototype_margin=args.prototype_margin,
    )


def l2_normalize_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def parameter_audit_rows(model) -> tuple[list[dict], list[dict]]:
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


def summarize_trainable(trainable_rows: list[dict], frozen_rows: list[dict], allowed_block_names: list[str]) -> dict:
    summary = base_summarize_trainable(trainable_rows, frozen_rows, allowed_block_names)
    scope_counts = {}
    for row in trainable_rows:
        name = row["parameter_name"]
        if name.startswith("eat.model.blocks."):
            prefix = ".".join(name.split(".")[:4])
        else:
            prefix = "other"
        scope_counts[prefix] = scope_counts.get(prefix, 0) + 1
    summary["trainable_scope_counts"] = scope_counts
    return summary


def load_path_embedding_map(embedding_root: Path, machines: list[str]) -> dict[str, np.ndarray]:
    path_to_embedding: dict[str, np.ndarray] = {}
    for machine in machines:
        npz_path = embedding_root / f"{machine}_clean_eat_train_embeddings.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing clean EAT train embeddings for prototypes: {npz_path}")
        arr = np.load(npz_path, allow_pickle=True)
        embs = arr["embeddings"].astype(np.float64)
        paths = arr["paths"].astype(str)
        for path, emb in zip(paths, embs):
            path_to_embedding[str(path)] = emb
    return path_to_embedding


def build_prototypes(inventory: list[dict], embedding_root: Path) -> tuple[np.ndarray, list[dict]]:
    machines = sorted({r["machine"] for r in inventory})
    path_to_embedding = load_path_embedding_map(embedding_root, machines)
    class_ids = sorted({int(r["pseudo_attribute_id"]) for r in inventory})
    id_to_row = {cid: i for i, cid in enumerate(class_ids)}
    buckets = {cid: [] for cid in class_ids}
    for row in inventory:
        path = row["path"]
        if path not in path_to_embedding:
            raise KeyError(f"Missing embedding for inventory path: {path}")
        buckets[int(row["pseudo_attribute_id"])].append(path_to_embedding[path])
    prototypes = []
    summary_rows = []
    for cid in class_ids:
        emb = np.stack(buckets[cid], axis=0).astype(np.float64)
        proto = emb.mean(axis=0)
        prototypes.append(proto)
        summary_rows.append({
            "pseudo_attribute_id": cid,
            "prototype_row": id_to_row[cid],
            "count": int(emb.shape[0]),
            "prototype_norm_before_l2": float(np.linalg.norm(proto)),
        })
    proto_arr = l2_normalize_np(np.stack(prototypes, axis=0).astype(np.float64))
    if class_ids != list(range(len(class_ids))):
        raise RuntimeError(f"Expected contiguous pseudo ids 0..N-1, got min={min(class_ids)} max={max(class_ids)} count={len(class_ids)}")
    return proto_arr.astype(np.float32), summary_rows


def prototype_ce(student_clip: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, cfg: Stage33AConfig):
    x = F.normalize(student_clip, dim=1)
    logits = x @ prototypes.t()
    if cfg.prototype_margin > 0:
        logits = logits.clone()
        logits[torch.arange(labels.shape[0], device=labels.device), labels] -= cfg.prototype_margin
    logits = logits * cfg.prototype_logit_scale
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def compute_loss(student, teacher, mel_batch, labels, machine_ids, prototypes, cfg: Stage33AConfig, device):
    with torch.no_grad():
        teacher_frames = extract_frame_output(teacher, mel_batch, device)
        teacher_clip = clean_clip_pool_torch(teacher_frames)
    student_frames = extract_frame_output(student, mel_batch, device)
    student_clip = clean_clip_pool_torch(student_frames)

    proto_loss, proto_acc = prototype_ce(student_clip, labels, prototypes, cfg)
    anchor = (1.0 - F.cosine_similarity(
        F.normalize(student_clip, dim=1),
        F.normalize(teacher_clip, dim=1),
        dim=1,
    )).mean()
    pairwise = compute_pairwise_geometry_loss(student_clip, teacher_clip)
    machine_center = compute_machine_center_loss(student_clip, teacher_clip, machine_ids)
    total = (
        cfg.prototype_ce_loss_weight * proto_loss
        + cfg.anchor_loss_weight * anchor
        + cfg.pairwise_geometry_loss_weight * pairwise
        + cfg.machine_center_loss_weight * machine_center
    )
    return {
        "total": total,
        "prototype_ce_loss": proto_loss,
        "anchor_loss": anchor,
        "pairwise_geometry_loss": pairwise,
        "machine_center_loss": machine_center,
        "prototype_acc": proto_acc,
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    inv_path = Path(args.pseudo_inventory)
    inventory = read_csv(inv_path) if inv_path.exists() else []
    machines = sorted({r["machine"] for r in inventory})
    expected_npz = [str(Path(args.prototype_embedding_root) / f"{m}_clean_eat_train_embeddings.npz") for m in machines]
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "pseudo_inventory": str(inv_path),
        "pseudo_inventory_exists": inv_path.exists(),
        "prototype_embedding_root": args.prototype_embedding_root,
        "prototype_embedding_files_exist": all(Path(p).exists() for p in expected_npz),
        "prototype_embedding_files": expected_npz,
        "pseudo_inventory_count": len(inventory),
        "pseudo_attribute_count": len({r["pseudo_attribute_id"] for r in inventory}) if inventory else 0,
        "machines": machines,
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage33A uses fixed clean-EAT pseudo-attribute prototypes.",
        ],
    })


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    model, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(model)
    allowed_blocks = enable_last_n_blocks(model, cfg.trainable_last_blocks)
    trainable_rows, frozen_rows = parameter_audit_rows(model)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)


def run_train_fixed_prototype(args):
    set_seed(args.seed)
    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    inventory_path = Path(args.pseudo_inventory)
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing pseudo inventory: {inventory_path}")
    inventory = read_csv(inventory_path)
    machines = sorted({r["machine"] for r in inventory})
    machine_to_id = machine_to_id_from_rows(inventory)
    prototypes_np, prototype_summary = build_prototypes(inventory, Path(args.prototype_embedding_root))
    num_classes = int(prototypes_np.shape[0])
    write_csv(out_root / "prototype_summary.csv", prototype_summary)

    train_rows, val_rows = split_stratified(inventory, cfg.val_ratio, args.seed)
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
    trainable_rows, frozen_rows = parameter_audit_rows(student)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    if not summary["no_unexpected_earlier_layer_trainable_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_parameter_names']}")
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)

    prototypes = torch.tensor(prototypes_np, dtype=torch.float32, device=device)
    student_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(student_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_path = out_root / "checkpoints" / "stage33a_fixed_prototype_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    loss_keys = ["total", "prototype_ce_loss", "anchor_loss", "pairwise_geometry_loss", "machine_center_loss", "prototype_acc"]
    for epoch in range(1, cfg.epochs + 1):
        student.train()
        order = rng.permutation(len(train_rows))
        train_sums = {k: 0.0 for k in loss_keys}
        train_count = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + cfg.batch_size]]
            mel = batch_rows_to_mel(batch, device)
            labels = torch.tensor([int(r["pseudo_attribute_id"]) for r in batch], dtype=torch.long, device=device)
            machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_loss(student, teacher, mel, labels, machine_ids, prototypes, cfg, device)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(student_params, cfg.grad_clip)
            optimizer.step()
            bsz = len(batch)
            for key in loss_keys:
                train_sums[key] += float(loss_dict[key].detach().cpu().item()) * bsz
            train_count += bsz

        student.eval()
        val_sums = {k: 0.0 for k in loss_keys}
        val_count = 0
        with torch.no_grad():
            for start in range(0, len(val_rows), cfg.batch_size):
                batch = val_rows[start:start + cfg.batch_size]
                mel = batch_rows_to_mel(batch, device)
                labels = torch.tensor([int(r["pseudo_attribute_id"]) for r in batch], dtype=torch.long, device=device)
                machine_ids = torch.tensor([machine_to_id[r["machine"]] for r in batch], dtype=torch.long, device=device)
                loss_dict = compute_loss(student, teacher, mel, labels, machine_ids, prototypes, cfg, device)
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
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "allowed_block_names": allowed_blocks,
                "trainable_summary": summary,
                "machines": machines,
                "machine_to_id": machine_to_id,
                "num_pseudo_attributes": num_classes,
                "pseudo_inventory": str(inventory_path),
                "prototype_embedding_root": args.prototype_embedding_root,
                "stage": "stage33a_fixed_prototype_representation",
                "uses_anomaly_labels": False,
                "uses_test_labels_for_checkpoint_selection": False,
                "ldn_parameters_tuned": False,
            }, best_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage33a_train_config.json", asdict(cfg))
    write_json(out_root / "stage33a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "num_pseudo_attributes": num_classes,
        "pseudo_inventory": str(inventory_path),
        "prototype_embedding_root": args.prototype_embedding_root,
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Fixed-prototype pseudo-attribute representation pilot.",
            "No trainable attribute head; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage33A Fixed-Prototype Representation",
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
    if args.mode == "train_fixed_prototype":
        run_train_fixed_prototype(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
