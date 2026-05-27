#!/usr/bin/env python
"""Stage22D: adapted full7 evaluation for the frozen LDN-Diff-K1 route.

This script evaluates a Stage22 adapted EAT checkpoint on all seven DCASE
machines using the unchanged min_clamp__mean + ldn_diff_K1_scorek5 backend.

It does not train, tune LDN parameters, add selectors, or select checkpoints by
test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test

from audit_stage21e_adapted_eval import (
    baseline_knn_scores,
    chord_distance_matrix,
    evaluate,
    extract_frame_output,
    load_stage21d_adapted,
    load_waveform_16k,
    local_density_knn,
    mean_pool,
    min_clamp,
    waveform_to_eat_tensor,
)


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ADAPTED_CKPT_DEFAULT = "results/stage22b_geometry_objective_sanity/checkpoints/stage22_geometry_best.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage22d_adapted_full7_eval"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE22D_ADAPTED_FULL7_EVAL.md"

FROZEN_ROUTE = "adapted_ldn_diff_K1_scorek5"
BASELINE_ROUTE = "adapted_baseline_knn_k5"
CONFIRMED_CLEAN_BEST_MEAN_HMEAN = 0.6159467288425517
CONFIRMED_CLEAN_BEST_MEAN_PAUC = 0.5603007518796993
CONFIRMED_CLEAN_BEST_MEAN_TPR99 = 0.10571428571428572
OLD_BEST_MEAN_HMEAN = 0.607175


def parse_args():
    p = argparse.ArgumentParser(description="Stage22D adapted full7 evaluation")
    p.add_argument("--mode", required=True, choices=["check", "extract_adapted_embeddings", "score_full7"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--adapted_ckpt_path", default=ADAPTED_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--latent_knn_k", type=int, default=5)
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


def emb_path(output_root: Path, machine: str) -> Path:
    return output_root / "embeddings" / f"{machine}_adapted_minclamp_mean_embeddings.npz"


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
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(train_items),
            "test_count": len(test_items),
            "embedding_exists": emb_path(out_root, machine).exists(),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "adapted_ckpt_exists": Path(args.adapted_ckpt_path).exists(),
        "data_root_exists": data_root.exists(),
        "machines": machines,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no scoring.",
            "Full7 evaluation uses frozen min_clamp__mean + ldn_diff_K1_scorek5 only.",
        ],
    })


def extract_machine_embeddings(model, device, data_root: Path, machine: str, output_root: Path, adapted_ckpt: Path):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")

    train_embs = []
    test_embs = []
    y_test = []
    d_test = []
    test_paths = []
    sample_frame_shape = None

    for item in train_items:
        wav = load_waveform_16k(Path(item.path))
        mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            frames = extract_frame_output(model, mel, device)
        frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
        frame_np = min_clamp(frame_np)
        train_embs.append(mean_pool(frame_np))
        sample_frame_shape = list(frame_np.shape)

    for item in test_items:
        wav = load_waveform_16k(Path(item.path))
        mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            frames = extract_frame_output(model, mel, device)
        frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
        frame_np = min_clamp(frame_np)
        test_embs.append(mean_pool(frame_np))
        y_test.append(int(item.label))
        d_test.append(item.domain)
        test_paths.append(str(item.path))
        sample_frame_shape = list(frame_np.shape)

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    path = emb_path(output_root, machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=np.asarray(y_test, dtype=np.int64),
        d_test=np.asarray(d_test, dtype="U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        machine=machine,
        checkpoint_path=str(adapted_ckpt),
        preprocess="min_clamp",
        pooling="mean",
        embedding_source="stage22_adapted_eat",
    )
    return {
        "machine": machine,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "embedding_dim": int(train_emb.shape[1]),
        "frame_shape": sample_frame_shape,
        "checkpoint_path": str(adapted_ckpt),
        "embedding_path": str(path),
    }


def run_extract_adapted_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    eat_path = Path(args.eat_model_path)
    adapted_ckpt = Path(args.adapted_ckpt_path)
    data_root = Path(args.data_root)
    if not eat_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {eat_path}")
    if not adapted_ckpt.exists():
        raise FileNotFoundError(f"Missing adapted checkpoint: {adapted_ckpt}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machine in {machines}; known={MACHINES_ALL}")

    model, _, device, blob, allowed_blocks = load_stage21d_adapted(eat_path, adapted_ckpt, args.device)
    rows = []
    for machine in machines:
        rows.append(extract_machine_embeddings(model, device, data_root, machine, out_root, adapted_ckpt))
    write_csv(out_root / "embedding_summary.csv", rows)
    write_json(out_root / "checkpoint_recap.json", {
        "adapted_ckpt_path": str(adapted_ckpt),
        "device": str(device),
        "checkpoint_stage": blob.get("stage"),
        "best_epoch": blob.get("best_epoch"),
        "best_val_loss": blob.get("best_val_loss"),
        "allowed_block_names": allowed_blocks,
        "machines": machines,
    })


def score_machine(machine: str, output_root: Path, score_k: int):
    arr = np.load(emb_path(output_root, machine), allow_pickle=True)
    train_emb = arr["train_embeddings"].astype(np.float64)
    test_emb = arr["test_embeddings"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    test_paths = arr["test_paths"].astype(str)

    test_ref_dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref_dist = chord_distance_matrix(train_emb, train_emb)
    rho_k1 = local_density_knn(ref_ref_dist, K=1)
    scores = {
        BASELINE_ROUTE: baseline_knn_scores(test_ref_dist, score_k=score_k),
        FROZEN_ROUTE: baseline_knn_scores(test_ref_dist - rho_k1[None, :], score_k=score_k),
    }
    return scores, y_test, d_test, test_paths


def run_score_full7(args):
    out_root = Path(args.output_root)
    machines = parse_csv_list(args.machines)
    missing = [str(emb_path(out_root, m)) for m in machines if not emb_path(out_root, m).exists()]
    if missing:
        raise FileNotFoundError(f"Missing adapted embeddings: {missing}")

    scores_root = out_root / "scores"
    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    baseline_per_machine = {}
    collapse_rows = []

    for machine in machines:
        scores_dict, y_test, d_test, test_paths = score_machine(machine, out_root, args.latent_knn_k)
        for candidate, scores in scores_dict.items():
            summary, tail, geom = evaluate(scores, y_test, d_test)
            per_machine_rows.append({"candidate": candidate, "machine": machine, **summary})
            tail_rows.append({"candidate": candidate, "machine": machine, **tail})
            geom_rows.append({"candidate": candidate, "machine": machine, **geom})
            cand_dir = scores_root / candidate
            cand_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cand_dir / f"{machine}_scores.npz",
                test_scores=scores.astype(np.float64),
                y_test=y_test,
                d_test=d_test.astype("U16"),
                test_paths=np.asarray(test_paths, dtype=object),
                candidate=candidate,
                machine=machine,
            )
            if candidate == BASELINE_ROUTE:
                baseline_per_machine[machine] = summary

    summary_rows = []
    comparison_rows = []
    for candidate in [BASELINE_ROUTE, FROZEN_ROUTE]:
        rows = [r for r in per_machine_rows if r["candidate"] == candidate]
        mean_h = float(np.mean([r["hmean"] for r in rows]))
        mean_p = float(np.mean([r["pAUC"] for r in rows]))
        mean_ks = float(np.mean([r["KS"] for r in rows]))
        mean_t95 = float(np.mean([r["TPR@95n"] for r in rows]))
        mean_t99 = float(np.mean([r["TPR@99n"] for r in rows]))

        improved_count = 0
        strong_collapse = False
        for row in rows:
            base = baseline_per_machine[row["machine"]]
            delta_h = row["hmean"] - base["hmean"]
            if delta_h > 0:
                improved_count += 1
            severe = bool(row["machine"] in {"bearing", "toytrain", "valve"} and delta_h < -0.02)
            strong_collapse = strong_collapse or severe
            collapse_rows.append({
                "candidate": candidate,
                "machine": row["machine"],
                "strong_machine": row["machine"] in {"bearing", "toytrain", "valve"},
                "baseline_hmean": base["hmean"],
                "candidate_hmean": row["hmean"],
                "delta_hmean_vs_adapted_baseline": delta_h,
                "severe_collapse": severe,
            })

        summary_rows.append({
            "candidate": candidate,
            "mean_hmean": mean_h,
            "mean_pAUC": mean_p,
            "mean_KS": mean_ks,
            "mean_TPR@95n": mean_t95,
            "mean_TPR@99n": mean_t99,
        })
        comparison_rows.append({
            "candidate": candidate,
            "mean_hmean": mean_h,
            "mean_pAUC": mean_p,
            "mean_TPR@99n": mean_t99,
            "delta_hmean_vs_adapted_baseline": 0.0,
            "delta_hmean_vs_confirmed_clean_best": mean_h - CONFIRMED_CLEAN_BEST_MEAN_HMEAN,
            "delta_hmean_vs_old_best": mean_h - OLD_BEST_MEAN_HMEAN,
            "improved_machine_count_vs_adapted_baseline": improved_count,
            "strong_machine_severe_collapse": strong_collapse,
        })

    adapted_baseline = next(r for r in summary_rows if r["candidate"] == BASELINE_ROUTE)
    for row in comparison_rows:
        if row["candidate"] != BASELINE_ROUTE:
            row["delta_hmean_vs_adapted_baseline"] = row["mean_hmean"] - adapted_baseline["mean_hmean"]

    frozen_summary = next(r for r in summary_rows if r["candidate"] == FROZEN_ROUTE)
    frozen_comp = next(r for r in comparison_rows if r["candidate"] == FROZEN_ROUTE)
    promotable = bool(
        frozen_summary["mean_hmean"] > CONFIRMED_CLEAN_BEST_MEAN_HMEAN
        and frozen_comp["improved_machine_count_vs_adapted_baseline"] >= 4
        and not frozen_comp["strong_machine_severe_collapse"]
    )
    negative = bool(
        frozen_summary["mean_hmean"] <= CONFIRMED_CLEAN_BEST_MEAN_HMEAN
        or frozen_comp["improved_machine_count_vs_adapted_baseline"] < 4
        or frozen_comp["strong_machine_severe_collapse"]
    )
    verdict = {
        "route": FROZEN_ROUTE,
        "mean_hmean": frozen_summary["mean_hmean"],
        "mean_pAUC": frozen_summary["mean_pAUC"],
        "mean_TPR99": frozen_summary["mean_TPR@99n"],
        "adapted_baseline_mean_hmean": adapted_baseline["mean_hmean"],
        "confirmed_clean_best_mean_hmean": CONFIRMED_CLEAN_BEST_MEAN_HMEAN,
        "confirmed_clean_best_mean_pAUC": CONFIRMED_CLEAN_BEST_MEAN_PAUC,
        "confirmed_clean_best_mean_TPR99": CONFIRMED_CLEAN_BEST_MEAN_TPR99,
        "delta_hmean_vs_adapted_baseline": frozen_comp["delta_hmean_vs_adapted_baseline"],
        "delta_hmean_vs_confirmed_clean_best": frozen_comp["delta_hmean_vs_confirmed_clean_best"],
        "delta_hmean_vs_old_best": frozen_comp["delta_hmean_vs_old_best"],
        "improved_machine_count_vs_adapted_baseline": frozen_comp["improved_machine_count_vs_adapted_baseline"],
        "strong_machine_severe_collapse": frozen_comp["strong_machine_severe_collapse"],
        "promotable": promotable,
        "negative": negative,
        "notes": [
            "Frozen adapted full7 audit only.",
            "No selector, oracle, fusion, or LDN parameter tuning.",
            "Checkpoint was supplied externally; no checkpoint selection is performed here.",
        ],
    }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "comparison_vs_adapted_baseline.csv", comparison_rows)
    write_csv(out_root / "collapse_check.csv", collapse_rows)
    write_json(out_root / "stage22d_adapted_full7_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage22D Adapted Full7 Eval",
        "",
        f"- route: `{FROZEN_ROUTE}`",
        f"- mean hmean: `{verdict['mean_hmean']:.12f}`",
        f"- mean pAUC: `{verdict['mean_pAUC']:.12f}`",
        f"- mean TPR@99n: `{verdict['mean_TPR99']:.12f}`",
        f"- adapted baseline mean hmean: `{verdict['adapted_baseline_mean_hmean']:.12f}`",
        f"- delta vs adapted baseline: `{verdict['delta_hmean_vs_adapted_baseline']:.12f}`",
        f"- delta vs confirmed clean best: `{verdict['delta_hmean_vs_confirmed_clean_best']:.12f}`",
        f"- improved machine count vs adapted baseline: `{verdict['improved_machine_count_vs_adapted_baseline']}`",
        f"- strong-machine severe collapse: `{verdict['strong_machine_severe_collapse']}`",
        f"- promotable: `{verdict['promotable']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "No selector, oracle, fusion, LDN parameter tuning, or training is performed in this stage.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_adapted_embeddings":
        run_extract_adapted_embeddings(args)
        return
    if args.mode == "score_full7":
        run_score_full7(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
