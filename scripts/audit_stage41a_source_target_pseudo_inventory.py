#!/usr/bin/env python
"""Stage41A: build source+target train-normal pseudo-attribute inventory.

The Stage29-40 pseudo-attribute branch used source train-normal clips only.
Stage41A keeps the same clean-EAT KMeans pseudo-attribute construction, but
allows target-domain train-normal clips into the pseudo inventory.

This script only extracts clean EAT embeddings and writes CSV/NPZ inventory
files. It does not train a model, score ASD data, use anomaly labels, or select
checkpoints with test labels.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21d_global_dap_ssl_sanity import freeze_all_parameters
from audit_stage21e_adapted_eval import load_clean_eat_local
from audit_stage29a_pseudo_attribute_representation import (
    clean_clip_pool_np,
    extract_clean_embedding,
    import_status,
    l2_normalize_np,
    parse_csv_list,
    write_csv,
    write_json,
    write_markdown,
)


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage41a_source_target_pseudo_inventory"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE41A_SOURCE_TARGET_PSEUDO_INVENTORY.md"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class Stage41AConfig:
    clusters_per_machine: int = 8
    min_cluster_size: int = 8
    use_target_train: bool = True


def parse_args():
    p = argparse.ArgumentParser(description="Stage41A source+target pseudo-attribute inventory")
    p.add_argument("--mode", required=True, choices=["check", "extract_source_target_pseudo_attributes"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--clusters_per_machine", type=int, default=8)
    p.add_argument("--min_cluster_size", type=int, default=8)
    p.add_argument("--pseudo_inventory", default="", help="Defaults to output_root/pseudo_attribute_inventory.csv.")
    p.add_argument("--source_only", action="store_true", help="Debug option: disable target train-normal inclusion.")
    return p.parse_args()


def build_config(args) -> Stage41AConfig:
    return Stage41AConfig(
        clusters_per_machine=args.clusters_per_machine,
        min_cluster_size=args.min_cluster_size,
        use_target_train=not args.source_only,
    )


def pseudo_inventory_path(args) -> Path:
    if args.pseudo_inventory:
        return Path(args.pseudo_inventory)
    return Path(args.output_root) / "pseudo_attribute_inventory.csv"


def load_train_rows(data_root: Path, machines: list[str], use_target_train: bool) -> list[dict]:
    items = scan_dataset(data_root)
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine)
        train_items, _ = split_train_test(machine_items, use_target_train=use_target_train)
        for item in train_items:
            rows.append({
                "machine": machine,
                "path": str(item.path),
                "domain": item.domain,
                "section": item.section,
            })
    return rows


def run_check(args):
    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine) if items else []
        source_train, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
        source_target_train, _ = split_train_test(machine_items, use_target_train=True) if machine_items else ([], [])
        target_train_count = len(source_target_train) - len(source_train)
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(source_train),
            "target_train_normal_count": target_train_count,
            "source_target_train_normal_count": len(source_target_train),
            "test_count": len(test_items),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "config": asdict(cfg),
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "pseudo_inventory": str(pseudo_inventory_path(args)),
        "machines": machines,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage41A inventory may include target-domain train-normal clips, but never test labels or anomaly labels.",
        ],
    })


def run_extract_source_target_pseudo_attributes(args):
    from sklearn.cluster import KMeans

    cfg = build_config(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machines: {machines}")
    rows = load_train_rows(Path(args.data_root), machines, cfg.use_target_train)
    model, device = load_clean_eat_local(Path(args.eat_model_path), args.device)
    freeze_all_parameters(model)
    model.eval()

    embedding_rows = []
    all_inventory = []
    summary_rows = []
    next_attr_id = 0
    for machine in machines:
        machine_rows = [r for r in rows if r["machine"] == machine]
        if not machine_rows:
            raise RuntimeError(f"No train-normal rows for machine={machine}")
        embs = []
        for i, row in enumerate(machine_rows):
            emb = extract_clean_embedding(model, row["path"], device)
            embs.append(emb)
            embedding_rows.append({
                "machine": machine,
                "row_index": i,
                "path": row["path"],
                "domain": row["domain"],
                "section": row["section"],
            })
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
        source_count = sum(1 for r in machine_rows if r["domain"] == "source")
        target_count = sum(1 for r in machine_rows if r["domain"] == "target")
        for local_id in range(n_clusters):
            cluster_rows = [r for r, label in zip(machine_rows, labels) if int(label) == local_id]
            summary_rows.append({
                "machine": machine,
                "local_cluster_id": local_id,
                "pseudo_attribute_id": next_attr_id + local_id,
                "cluster_count": int(counts[local_id]),
                "cluster_fraction": float(counts[local_id] / max(1, len(machine_rows))),
                "cluster_source_count": sum(1 for r in cluster_rows if r["domain"] == "source"),
                "cluster_target_count": sum(1 for r in cluster_rows if r["domain"] == "target"),
            })
        for row, local_id in zip(machine_rows, labels):
            all_inventory.append({
                "machine": machine,
                "path": row["path"],
                "domain": row["domain"],
                "section": row["section"],
                "local_cluster_id": int(local_id),
                "pseudo_attribute_id": int(next_attr_id + int(local_id)),
            })
        summary_rows.append({
            "machine": machine,
            "local_cluster_id": "ALL",
            "pseudo_attribute_id": "",
            "cluster_count": len(machine_rows),
            "cluster_fraction": 1.0,
            "source_count": source_count,
            "target_count": target_count,
            "n_clusters": int(n_clusters),
            "kmeans_inertia": inertia,
        })
        next_attr_id += n_clusters
        np.savez_compressed(
            out_root / f"{machine}_clean_eat_source_target_train_embeddings.npz",
            embeddings=emb_arr,
            labels=labels,
            paths=np.asarray([r["path"] for r in machine_rows], dtype=object),
            domains=np.asarray([r["domain"] for r in machine_rows], dtype="U16"),
            sections=np.asarray([r["section"] for r in machine_rows], dtype="U16"),
            machine=machine,
        )

    write_csv(pseudo_inventory_path(args), all_inventory)
    write_csv(out_root / "pseudo_attribute_summary.csv", summary_rows)
    write_csv(out_root / "embedding_inventory.csv", embedding_rows)
    write_json(out_root / "pseudo_attribute_verdict.json", {
        "mode": "extract_source_target_pseudo_attributes",
        "config": asdict(cfg),
        "machines": machines,
        "train_normal_count": len(all_inventory),
        "source_train_normal_count": sum(1 for r in all_inventory if r["domain"] == "source"),
        "target_train_normal_count": sum(1 for r in all_inventory if r["domain"] == "target"),
        "pseudo_attribute_count": next_attr_id,
        "clusters_per_machine_requested": cfg.clusters_per_machine,
        "min_cluster_size": cfg.min_cluster_size,
        "uses_anomaly_labels": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "notes": [
            "Pseudo attributes are KMeans clusters over clean EAT min_clamp mean embeddings within each machine.",
            "Target-domain train-normal clips may be included; test clips and anomaly labels are never used.",
            "No ASD scoring or training performed in this mode.",
        ],
    })
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage41A Source+Target Pseudo-Attribute Inventory",
        "",
        f"- machines: `{','.join(machines)}`",
        f"- train-normal rows: `{len(all_inventory)}`",
        f"- source train-normal rows: `{sum(1 for r in all_inventory if r['domain'] == 'source')}`",
        f"- target train-normal rows: `{sum(1 for r in all_inventory if r['domain'] == 'target')}`",
        f"- pseudo attributes: `{next_attr_id}`",
        f"- pseudo inventory: `{pseudo_inventory_path(args)}`",
        "",
        "No training, scoring, anomaly labels, or test-label checkpoint selection were used.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_source_target_pseudo_attributes":
        run_extract_source_target_pseudo_attributes(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
