#!/usr/bin/env python
"""Stage28A: clean EAT pooling and spike-suppression audit.

This is a no-training, frozen-backend audit inspired by the temporal-pooling
and robust ASD paper evidence:
- clean EAT only
- fixed frame preprocessing/pooling candidates
- frozen baseline_knn_k5 and ldn_diff_K1_scorek5 scoring
- no selector, fusion, LDN parameter tuning, or checkpoint selection
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
    load_clean_eat_local,
    load_waveform_16k,
    local_density_knn,
    min_clamp,
    waveform_to_eat_tensor,
)


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage28a_eat_pooling_spike_audit"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE28A_EAT_POOLING_SPIKE_AUDIT.md"
CLIP_REFERENCE_DEFAULT = "results/stage18e_local_density_protocol_full7/per_machine.csv"

CONFIRMED_CLEAN_LDN_HMEAN = 0.6159467288425517
CONFIRMED_CLEAN_LDN_PAUC = 0.5603007518796993
CONFIRMED_CLEAN_LDN_TPR99 = 0.10571428571428572
STAGE22_HMEAN_BEST = 0.6166917954459288
EPS = 1e-12

POOLING_CANDIDATES = [
    "minclamp_mean",
    "spike_mean",
    "minclamp_gem_p3",
    "spike_gem_p3",
    "minclamp_rdp_g2",
    "spike_rdp_g2",
    "minclamp_rdp_g2_gem_p3",
    "spike_rdp_g2_gem_p3",
]


def parse_args():
    p = argparse.ArgumentParser(description="Stage28A clean EAT pooling/spike audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_pooling_embeddings", "score_pooling"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--clip_reference_per_machine", default=CLIP_REFERENCE_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--gem_p", type=float, default=3.0)
    p.add_argument("--rdp_gamma", type=float, default=2.0)
    p.add_argument("--spike_threshold", type=float, default=0.5)
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


def emb_path(output_root: Path, machine: str, candidate: str) -> Path:
    return output_root / "embeddings" / candidate / f"{machine}_embeddings.npz"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_clip_ldn_reference(path: Path) -> dict[str, dict]:
    rows = load_csv(path)
    return {r["machine"]: r for r in rows if r.get("candidate") == "ldn_diff_K1_scorek5"}


def spike_suppress(frames: np.ndarray, threshold: float) -> np.ndarray:
    """Softly squash large-magnitude frame activations while preserving sign."""
    x = frames.astype(np.float32, copy=True)
    hi = x > threshold
    lo = x < -threshold
    x[hi] = threshold + np.tanh(x[hi] - threshold)
    x[lo] = -threshold + np.tanh(x[lo] + threshold)
    return x.astype(np.float32)


def gem_pool(frames: np.ndarray, p: float) -> np.ndarray:
    x = np.maximum(frames.astype(np.float64), 0.0)
    return np.power(np.mean(np.power(x + EPS, p), axis=0), 1.0 / p).astype(np.float64)


def rdp_weights(frames: np.ndarray, gamma: float) -> np.ndarray:
    x = frames.astype(np.float64)
    center = np.mean(x, axis=0, keepdims=True)
    d = np.linalg.norm(x - center, axis=1)
    max_d = float(np.max(d))
    if max_d <= EPS:
        return np.full((x.shape[0],), 1.0 / max(1, x.shape[0]), dtype=np.float64)
    w = np.power(1.0 + d / max_d, gamma)
    return (w / np.sum(w)).astype(np.float64)


def rdp_pool(frames: np.ndarray, gamma: float) -> np.ndarray:
    w = rdp_weights(frames, gamma)
    return np.sum(frames.astype(np.float64) * w[:, None], axis=0).astype(np.float64)


def rdp_gem_pool(frames: np.ndarray, gamma: float, p: float) -> np.ndarray:
    w = rdp_weights(frames, gamma)
    x = np.maximum(frames.astype(np.float64), 0.0)
    return np.power(np.sum(w[:, None] * np.power(x + EPS, p), axis=0), 1.0 / p).astype(np.float64)


def pool_candidate(frames: np.ndarray, candidate: str, gem_p: float, rdp_gamma: float, spike_threshold: float) -> np.ndarray:
    base = min_clamp(frames)
    if candidate.startswith("spike_"):
        base = spike_suppress(base, spike_threshold)
    if candidate.endswith("_mean"):
        return np.mean(base, axis=0, dtype=np.float64).astype(np.float64)
    if "_gem_p3" in candidate and "_rdp_" not in candidate:
        return gem_pool(base, gem_p)
    if "_rdp_g2_gem_p3" in candidate:
        return rdp_gem_pool(base, rdp_gamma, gem_p)
    if "_rdp_g2" in candidate:
        return rdp_pool(base, rdp_gamma)
    raise ValueError(f"Unknown pooling candidate: {candidate}")


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
            "embedding_exists_all_candidates": all(emb_path(out_root, machine, c).exists() for c in POOLING_CANDIDATES),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "clip_reference_exists": Path(args.clip_reference_per_machine).exists(),
        "machines": machines,
        "pooling_candidates": POOLING_CANDIDATES,
        "all_requested_machines_known": all(m in MACHINES_ALL for m in machines),
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no scoring.",
            "Candidate set is fixed from paper-inspired pooling/spike suppression.",
            "Frozen score backends are baseline_knn_k5 and ldn_diff_K1_scorek5.",
        ],
    })


def extract_machine_embeddings(model, device, data_root: Path, machine: str, output_root: Path, args):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")

    per_candidate_train = {c: [] for c in POOLING_CANDIDATES}
    per_candidate_test = {c: [] for c in POOLING_CANDIDATES}
    y_test: list[int] = []
    d_test: list[str] = []
    test_paths: list[str] = []
    sample_frame_shape = None

    for item in train_items:
        wav = load_waveform_16k(Path(item.path))
        mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            frames = extract_frame_output(model, mel, device)
        frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sample_frame_shape = list(frame_np.shape)
        for candidate in POOLING_CANDIDATES:
            per_candidate_train[candidate].append(
                pool_candidate(frame_np, candidate, args.gem_p, args.rdp_gamma, args.spike_threshold)
            )

    for item in test_items:
        wav = load_waveform_16k(Path(item.path))
        mel = waveform_to_eat_tensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            frames = extract_frame_output(model, mel, device)
        frame_np = frames.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sample_frame_shape = list(frame_np.shape)
        for candidate in POOLING_CANDIDATES:
            per_candidate_test[candidate].append(
                pool_candidate(frame_np, candidate, args.gem_p, args.rdp_gamma, args.spike_threshold)
            )
        y_test.append(int(item.label))
        d_test.append(item.domain)
        test_paths.append(str(item.path))

    rows = []
    for candidate in POOLING_CANDIDATES:
        train_emb = np.stack(per_candidate_train[candidate], axis=0).astype(np.float64)
        test_emb = np.stack(per_candidate_test[candidate], axis=0).astype(np.float64)
        path = emb_path(output_root, machine, candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            y_test=np.asarray(y_test, dtype=np.int64),
            d_test=np.asarray(d_test, dtype="U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            machine=machine,
            candidate=candidate,
            embedding_source="clean_eat",
            preprocess="min_clamp+optional_spike_suppression",
            pooling=candidate,
            gem_p=float(args.gem_p),
            rdp_gamma=float(args.rdp_gamma),
            spike_threshold=float(args.spike_threshold),
        )
        rows.append({
            "machine": machine,
            "candidate": candidate,
            "train_count": len(train_items),
            "test_count": len(test_items),
            "embedding_dim": int(train_emb.shape[1]),
            "frame_shape": sample_frame_shape,
            "embedding_path": str(path),
        })
    return rows


def run_extract_pooling_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    eat_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    if not eat_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {eat_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machine in {machines}; known={MACHINES_ALL}")

    model, device = load_clean_eat_local(eat_path, args.device)
    rows = []
    for machine in machines:
        rows.extend(extract_machine_embeddings(model, device, data_root, machine, out_root, args))
    write_csv(out_root / "embedding_summary.csv", rows)
    write_json(out_root / "extract_recap.json", {
        "mode": "extract_pooling_embeddings",
        "device": str(device),
        "machines": machines,
        "pooling_candidates": POOLING_CANDIDATES,
        "gem_p": float(args.gem_p),
        "rdp_gamma": float(args.rdp_gamma),
        "spike_threshold": float(args.spike_threshold),
        "notes": [
            "Clean EAT forward pass only.",
            "No training or checkpoint selection.",
        ],
    })


def score_embedding_file(path: Path, score_k: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    train_emb = arr["train_embeddings"].astype(np.float64)
    test_emb = arr["test_embeddings"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    test_paths = arr["test_paths"].astype(str)
    test_ref_dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref_dist = chord_distance_matrix(train_emb, train_emb)
    rho_k1 = local_density_knn(ref_ref_dist, K=1)
    base = baseline_knn_scores(test_ref_dist, score_k=score_k)
    ldn = baseline_knn_scores(test_ref_dist - rho_k1[None, :], score_k=score_k)
    return {"baseline_knn_k5": base, "ldn_diff_K1_scorek5": ldn}, y_test, d_test, test_paths


def run_score_pooling(args):
    out_root = Path(args.output_root)
    machines = parse_csv_list(args.machines)
    missing = [
        str(emb_path(out_root, machine, candidate))
        for machine in machines
        for candidate in POOLING_CANDIDATES
        if not emb_path(out_root, machine, candidate).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing pooling embeddings: {missing[:10]} total={len(missing)}")

    clip_refs = get_clip_ldn_reference(Path(args.clip_reference_per_machine))
    scores_root = out_root / "scores"
    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    comparison_rows = []

    for candidate in POOLING_CANDIDATES:
        for machine in machines:
            scores_dict, y_test, d_test, test_paths = score_embedding_file(
                emb_path(out_root, machine, candidate), args.latent_knn_k
            )
            for backend, scores in scores_dict.items():
                route = f"{candidate}__{backend}"
                summary, tail, geom = evaluate(scores, y_test, d_test)
                per_machine_rows.append({"candidate": route, "pooling": candidate, "backend": backend, "machine": machine, **summary})
                tail_rows.append({"candidate": route, "pooling": candidate, "backend": backend, "machine": machine, **tail})
                geom_rows.append({"candidate": route, "pooling": candidate, "backend": backend, "machine": machine, **geom})
                cand_dir = scores_root / route
                cand_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cand_dir / f"{machine}_scores.npz",
                    test_scores=scores.astype(np.float64),
                    y_test=y_test,
                    d_test=d_test.astype("U16"),
                    test_paths=np.asarray(test_paths, dtype=object),
                    candidate=route,
                    machine=machine,
                )
                ref = clip_refs.get(machine)
                comparison_rows.append({
                    "candidate": route,
                    "pooling": candidate,
                    "backend": backend,
                    "machine": machine,
                    "hmean": summary["hmean"],
                    "pAUC": summary["pAUC"],
                    "TPR@99n": summary["TPR@99n"],
                    "clip_ldn_hmean": float(ref["hmean"]) if ref else None,
                    "clip_ldn_pAUC": float(ref["pAUC"]) if ref else None,
                    "clip_ldn_TPR99": float(ref["TPR@99n"]) if ref else None,
                    "delta_hmean_vs_clean_clip_ldn": summary["hmean"] - float(ref["hmean"]) if ref else None,
                    "delta_pAUC_vs_clean_clip_ldn": summary["pAUC"] - float(ref["pAUC"]) if ref else None,
                    "delta_TPR99_vs_clean_clip_ldn": summary["TPR@99n"] - float(ref["TPR@99n"]) if ref else None,
                })

    summary_rows = []
    for route in sorted({r["candidate"] for r in per_machine_rows}):
        rows = [r for r in per_machine_rows if r["candidate"] == route]
        comps = [r for r in comparison_rows if r["candidate"] == route]
        mean_h = float(np.mean([r["hmean"] for r in rows]))
        mean_p = float(np.mean([r["pAUC"] for r in rows]))
        mean_ks = float(np.mean([r["KS"] for r in rows]))
        mean_t95 = float(np.mean([r["TPR@95n"] for r in rows]))
        mean_t99 = float(np.mean([r["TPR@99n"] for r in rows]))
        improved_count = sum(1 for r in comps if r["delta_hmean_vs_clean_clip_ldn"] is not None and r["delta_hmean_vs_clean_clip_ldn"] > 0)
        bottleneck_t99 = sum(
            1 for r in comps
            if r["machine"] in {"fan", "gearbox", "slider", "toycar", "toytrain"}
            and r["delta_TPR99_vs_clean_clip_ldn"] is not None
            and r["delta_TPR99_vs_clean_clip_ldn"] > 0
        )
        strong_collapse = any(
            r["machine"] in {"bearing", "toytrain", "valve"}
            and r["delta_hmean_vs_clean_clip_ldn"] is not None
            and r["delta_hmean_vs_clean_clip_ldn"] < -0.02
            for r in comps
        )
        summary_rows.append({
            "candidate": route,
            "pooling": rows[0]["pooling"],
            "backend": rows[0]["backend"],
            "mean_hmean": mean_h,
            "mean_pAUC": mean_p,
            "mean_KS": mean_ks,
            "mean_TPR@95n": mean_t95,
            "mean_TPR@99n": mean_t99,
            "delta_hmean_vs_clean_ldn_best": mean_h - CONFIRMED_CLEAN_LDN_HMEAN,
            "delta_pAUC_vs_clean_ldn_best": mean_p - CONFIRMED_CLEAN_LDN_PAUC,
            "delta_TPR99_vs_clean_ldn_best": mean_t99 - CONFIRMED_CLEAN_LDN_TPR99,
            "delta_hmean_vs_stage22_hmean_best": mean_h - STAGE22_HMEAN_BEST,
            "improved_machine_count_vs_clean_clip_ldn": improved_count,
            "bottleneck_TPR99_improved_count_vs_clean_clip_ldn": bottleneck_t99,
            "strong_machine_severe_collapse": strong_collapse,
        })

    best = max(summary_rows, key=lambda r: r["mean_hmean"])
    best_ldn = max(
        [r for r in summary_rows if r["backend"] == "ldn_diff_K1_scorek5"],
        key=lambda r: r["mean_hmean"],
    )
    promotable = bool(
        best_ldn["mean_hmean"] > STAGE22_HMEAN_BEST
        and best_ldn["improved_machine_count_vs_clean_clip_ldn"] >= 4
        and not best_ldn["strong_machine_severe_collapse"]
    )
    pooling_positive = bool(
        best_ldn["mean_hmean"] > CONFIRMED_CLEAN_LDN_HMEAN
        or best_ldn["bottleneck_TPR99_improved_count_vs_clean_clip_ldn"] >= 2
        or best_ldn["improved_machine_count_vs_clean_clip_ldn"] >= 4
    )
    verdict = {
        "stage": "Stage28A EAT pooling/spike audit",
        "best_overall_candidate": best["candidate"],
        "best_overall_mean_hmean": best["mean_hmean"],
        "best_ldn_candidate": best_ldn["candidate"],
        "best_ldn_mean_hmean": best_ldn["mean_hmean"],
        "best_ldn_mean_pAUC": best_ldn["mean_pAUC"],
        "best_ldn_mean_TPR99": best_ldn["mean_TPR@99n"],
        "delta_hmean_vs_clean_ldn_best": best_ldn["delta_hmean_vs_clean_ldn_best"],
        "delta_hmean_vs_stage22_hmean_best": best_ldn["delta_hmean_vs_stage22_hmean_best"],
        "improved_machine_count_vs_clean_clip_ldn": best_ldn["improved_machine_count_vs_clean_clip_ldn"],
        "bottleneck_TPR99_improved_count_vs_clean_clip_ldn": best_ldn["bottleneck_TPR99_improved_count_vs_clean_clip_ldn"],
        "strong_machine_severe_collapse": best_ldn["strong_machine_severe_collapse"],
        "pooling_positive": pooling_positive,
        "promotable": promotable,
        "policy": {
            "no_training": True,
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_fusion": True,
            "no_test_label_checkpoint_selection": True,
        },
        "notes": [
            "Fixed candidate set from temporal-pooling/spike-suppression paper evidence.",
            "LDN backend remains frozen at K1 score_k5 diff form.",
            "This is architecture evidence only; selecting a global pooling route requires full7 result review.",
        ],
    }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "comparison_vs_clean_clip_ldn.csv", comparison_rows)
    write_json(out_root / "stage28a_eat_pooling_spike_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage28A EAT Pooling Spike Audit",
        "",
        f"- best LDN candidate: `{verdict['best_ldn_candidate']}`",
        f"- best LDN mean hmean: `{verdict['best_ldn_mean_hmean']:.12f}`",
        f"- best LDN mean pAUC: `{verdict['best_ldn_mean_pAUC']:.12f}`",
        f"- best LDN mean TPR@99n: `{verdict['best_ldn_mean_TPR99']:.12f}`",
        f"- delta hmean vs clean LDN best: `{verdict['delta_hmean_vs_clean_ldn_best']:.12f}`",
        f"- delta hmean vs Stage22 hmean best: `{verdict['delta_hmean_vs_stage22_hmean_best']:.12f}`",
        f"- improved machines vs clean clip LDN: `{verdict['improved_machine_count_vs_clean_clip_ldn']}`",
        f"- bottleneck TPR@99n improved count: `{verdict['bottleneck_TPR99_improved_count_vs_clean_clip_ldn']}`",
        f"- strong-machine severe collapse: `{verdict['strong_machine_severe_collapse']}`",
        f"- pooling positive: `{verdict['pooling_positive']}`",
        f"- promotable: `{verdict['promotable']}`",
        "",
        "No training, LDN parameter tuning, selector, fusion, or test-label checkpoint selection is performed.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_pooling_embeddings":
        run_extract_pooling_embeddings(args)
        return
    if args.mode == "score_pooling":
        run_score_pooling(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
