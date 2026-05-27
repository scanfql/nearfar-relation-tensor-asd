#!/usr/bin/env python
"""Stage29A: pseudo-attribute/domain-adaptive representation pilot.

Paper-inspired route:
- source train-normal only
- clean EAT teacher frozen
- cluster clean EAT clip embeddings per machine into pseudo attributes
- train an EAT student last-block adapter to classify global pseudo attributes
- keep a clean-teacher anchor on clip geometry
- save a Stage21E/22D-compatible `student_state_dict` checkpoint

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
    min_clamp,
    waveform_to_eat_tensor,
)
from audit_stage21d_global_dap_ssl_sanity import (
    enable_last_n_blocks,
    freeze_all_parameters,
    summarize_trainable as base_summarize_trainable,
)


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage29a_pseudo_attribute_representation"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE29A_PSEUDO_ATTRIBUTE_REPRESENTATION.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
EPS = 1e-12


@dataclass
class Stage29AConfig:
    trainable_last_blocks: int = 1
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-6
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    clusters_per_machine: int = 8
    min_cluster_size: int = 8
    ce_loss_weight: float = 1.0
    anchor_loss_weight: float = 0.10
    logit_scale: float = 20.0
    arc_margin: float = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Stage29A pseudo-attribute representation pilot")
    p.add_argument("--mode", required=True, choices=["check", "extract_pseudo_attributes", "inspect_trainable", "train_pseudo_attribute_representation"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--clusters_per_machine", type=int, default=8)
    p.add_argument("--min_cluster_size", type=int, default=8)
    p.add_argument("--ce_loss_weight", type=float, default=1.0)
    p.add_argument("--anchor_loss_weight", type=float, default=0.10)
    p.add_argument("--logit_scale", type=float, default=20.0)
    p.add_argument("--arc_margin", type=float, default=0.10)
    p.add_argument("--pseudo_inventory", default="", help="Optional pseudo-attribute inventory CSV. Defaults to output_root/pseudo_attribute_inventory.csv.")
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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def build_config(args) -> Stage29AConfig:
    return Stage29AConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        clusters_per_machine=args.clusters_per_machine,
        min_cluster_size=args.min_cluster_size,
        ce_loss_weight=args.ce_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        logit_scale=args.logit_scale,
        arc_margin=args.arc_margin,
    )


def pseudo_inventory_path(args) -> Path:
    if args.pseudo_inventory:
        return Path(args.pseudo_inventory)
    return Path(args.output_root) / "pseudo_attribute_inventory.csv"


def clean_clip_pool_torch(frames: torch.Tensor) -> torch.Tensor:
    frames = torch.where(torch.abs(frames) < 0.1, torch.zeros_like(frames), frames)
    return frames.mean(dim=1)


def clean_clip_pool_np(frames: np.ndarray) -> np.ndarray:
    return min_clamp(frames).mean(axis=0, dtype=np.float64).astype(np.float64)


def l2_normalize_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def parameter_audit_rows(model, head) -> tuple[list[dict], list[dict]]:
    trainable_rows = []
    frozen_rows = []
    for scope, module in [("eat", model), ("attribute_head", head)]:
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
    summary = base_summarize_trainable(trainable_rows, frozen_rows, allowed_block_names)
    filtered_unexpected = [
        name
        for name in summary["unexpected_trainable_parameter_names"]
        if not name.startswith("attribute_head.")
    ]
    scope_counts = {}
    for row in trainable_rows:
        name = row["parameter_name"]
        if name.startswith("eat.model.blocks."):
            prefix = ".".join(name.split(".")[:4])
        elif name.startswith("attribute_head."):
            prefix = "attribute_head"
        else:
            prefix = "other"
        scope_counts[prefix] = scope_counts.get(prefix, 0) + 1
    summary["trainable_scope_counts"] = scope_counts
    summary["no_unexpected_earlier_layer_trainable_params"] = len(filtered_unexpected) == 0
    summary["unexpected_trainable_parameter_names"] = filtered_unexpected
    return summary


class ArcAttributeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 20.0, margin: float = 0.10):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        logits = x @ w.t()
        if labels is not None and self.margin > 0:
            logits = logits.clone()
            logits[torch.arange(labels.shape[0], device=labels.device), labels] -= self.margin
        return logits * self.scale


def load_train_rows(data_root: Path, machines: list[str]) -> list[dict]:
    items = scan_dataset(data_root)
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine)
        train_items, _ = split_train_test(machine_items, use_target_train=False)
        for item in train_items:
            rows.append({"machine": machine, "path": str(item.path)})
    return rows


def extract_clean_embedding(model, path: str, device) -> np.ndarray:
    wav = load_waveform_16k(Path(path))
    mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        frames = extract_frame_output(model, mel, device)
    frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return clean_clip_pool_np(frame_np)


def load_mel(path: str, device):
    wav = load_waveform_16k(Path(path))
    return waveform_to_eat_tensor(wav).to(device)


def batch_rows_to_mel(rows: list[dict], device):
    return torch.stack([load_mel(row["path"], device) for row in rows], dim=0)


def split_stratified(rows: list[dict], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_rows = []
    val_rows = []
    classes = sorted({int(r["pseudo_attribute_id"]) for r in rows})
    for cls in classes:
        rr = [r for r in rows if int(r["pseudo_attribute_id"]) == cls]
        idx = rng.permutation(len(rr))
        val_n = max(1, int(round(len(rr) * val_ratio))) if len(rr) > 1 else 0
        val_n = min(val_n, max(0, len(rr) - 1))
        val_idx = set(int(i) for i in idx[:val_n])
        for i, row in enumerate(rr):
            (val_rows if i in val_idx else train_rows).append(row)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def compute_loss(student, teacher, head, mel_batch, labels, cfg: Stage29AConfig, device):
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
    total = cfg.ce_loss_weight * ce + cfg.anchor_loss_weight * anchor
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return {"total": total, "ce_loss": ce, "anchor_loss": anchor, "acc": acc}


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
        "pseudo_inventory_exists": pseudo_inventory_path(args).exists(),
        "machines": machines,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage29A trains a pseudo-attribute representation checkpoint compatible with frozen backend evaluation.",
        ],
    })


def run_extract_pseudo_attributes(args):
    from sklearn.cluster import KMeans

    set_seed(args.seed)
    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machines: {machines}")
    rows = load_train_rows(Path(args.data_root), machines)
    model, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(model)
    model.eval()

    embedding_rows = []
    all_inventory = []
    summary_rows = []
    next_attr_id = 0
    for machine in machines:
        machine_rows = [r for r in rows if r["machine"] == machine]
        embs = []
        for i, row in enumerate(machine_rows):
            emb = extract_clean_embedding(model, row["path"], device)
            embs.append(emb)
            embedding_rows.append({"machine": machine, "row_index": i, "path": row["path"]})
        emb_arr = np.stack(embs, axis=0).astype(np.float64)
        emb_norm = l2_normalize_np(emb_arr)
        n_clusters = min(cfg.clusters_per_machine, max(1, len(machine_rows) // cfg.min_cluster_size))
        n_clusters = max(2, n_clusters) if len(machine_rows) >= 2 * cfg.min_cluster_size else 1
        if n_clusters == 1:
            labels = np.zeros((len(machine_rows),), dtype=np.int64)
            inertia = 0.0
        else:
            km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
            labels = km.fit_predict(emb_norm).astype(np.int64)
            inertia = float(km.inertia_)
        counts = np.bincount(labels, minlength=n_clusters)
        for local_id in range(n_clusters):
            summary_rows.append({
                "machine": machine,
                "local_cluster_id": local_id,
                "pseudo_attribute_id": next_attr_id + local_id,
                "cluster_count": int(counts[local_id]),
                "cluster_fraction": float(counts[local_id] / max(1, len(machine_rows))),
            })
        for row, local_id in zip(machine_rows, labels):
            all_inventory.append({
                "machine": machine,
                "path": row["path"],
                "local_cluster_id": int(local_id),
                "pseudo_attribute_id": int(next_attr_id + int(local_id)),
            })
        summary_rows.append({
            "machine": machine,
            "local_cluster_id": "ALL",
            "pseudo_attribute_id": "",
            "cluster_count": len(machine_rows),
            "cluster_fraction": 1.0,
            "n_clusters": int(n_clusters),
            "kmeans_inertia": inertia,
        })
        next_attr_id += n_clusters
        np.savez_compressed(out_root / f"{machine}_clean_eat_train_embeddings.npz", embeddings=emb_arr, labels=labels, paths=np.asarray([r["path"] for r in machine_rows], dtype=object), machine=machine)

    write_csv(pseudo_inventory_path(args), all_inventory)
    write_csv(out_root / "pseudo_attribute_summary.csv", summary_rows)
    write_csv(out_root / "embedding_inventory.csv", embedding_rows)
    write_json(out_root / "pseudo_attribute_verdict.json", {
        "mode": "extract_pseudo_attributes",
        "machines": machines,
        "train_normal_count": len(all_inventory),
        "pseudo_attribute_count": next_attr_id,
        "clusters_per_machine_requested": cfg.clusters_per_machine,
        "min_cluster_size": cfg.min_cluster_size,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "notes": [
            "Pseudo attributes are KMeans clusters over clean EAT min_clamp mean embeddings within each machine.",
            "No ASD scoring or training performed in this mode.",
        ],
    })


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    inventory = read_csv(pseudo_inventory_path(args))
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


def run_train_pseudo_attribute_representation(args):
    set_seed(args.seed)
    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    inventory_path = pseudo_inventory_path(args)
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing pseudo inventory: {inventory_path}. Run --mode extract_pseudo_attributes first.")
    rows = read_csv(inventory_path)
    machines = sorted({r["machine"] for r in rows})
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

    best_path = out_root / "checkpoints" / "stage29a_pseudo_attribute_best.pt"
    best_val = float("inf")
    best_epoch = -1
    log_rows = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, cfg.epochs + 1):
        student.train()
        head.train()
        order = rng.permutation(len(train_rows))
        train_sums = {k: 0.0 for k in ["total", "ce_loss", "anchor_loss", "acc"]}
        train_count = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + cfg.batch_size]]
            mel = batch_rows_to_mel(batch, device)
            labels = torch.tensor([int(r["pseudo_attribute_id"]) for r in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_loss(student, teacher, head, mel, labels, cfg, device)
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
                mel = batch_rows_to_mel(batch, device)
                labels = torch.tensor([int(r["pseudo_attribute_id"]) for r in batch], dtype=torch.long, device=device)
                loss_dict = compute_loss(student, teacher, head, mel, labels, cfg, device)
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
                "attribute_head_state_dict": head.state_dict(),
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "allowed_block_names": allowed_blocks,
                "trainable_summary": summary,
                "machines": machines,
                "num_pseudo_attributes": num_classes,
                "pseudo_inventory": str(inventory_path),
                "stage": "stage29a_pseudo_attribute_representation",
                "uses_anomaly_labels": False,
                "uses_test_labels_for_checkpoint_selection": False,
                "ldn_parameters_tuned": False,
            }, best_path)

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage29a_train_config.json", asdict(cfg))
    write_json(out_root / "stage29a_training_verdict.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "machines": machines,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "num_pseudo_attributes": num_classes,
        "stage21e_compatible_checkpoint": True,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Pseudo-attribute/domain-adaptive representation pilot.",
            "Attribute head is training-only; ASD scoring should use frozen backend eval on student_state_dict.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage29A Pseudo-Attribute Representation Pilot",
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
    if args.mode == "extract_pseudo_attributes":
        run_extract_pseudo_attributes(args)
        return
    if args.mode == "inspect_trainable":
        run_inspect_trainable(args)
        return
    if args.mode == "train_pseudo_attribute_representation":
        run_train_pseudo_attribute_representation(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
