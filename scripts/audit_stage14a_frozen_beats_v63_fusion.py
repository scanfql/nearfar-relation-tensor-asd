#!/usr/bin/env python
"""Stage 14A frozen BEATs baseline and V63+BEATs fixed-fusion audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.features.logmel import load_audio


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]
DEFAULT_BEATS_CKPT = "/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 14A frozen BEATs + V63 fusion audit")
    p.add_argument("--mode", required=True, choices=["check", "a0_frozen_beats", "a1_v63_beats_fusion"])
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--beats_checkpoint", default=DEFAULT_BEATS_CKPT)
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--v37_root", default="results/v37_teacher_student_pretrain_seedpanel")
    p.add_argument("--v61_root", default="results/v61_cross_year_original_encoder_seedpanel")
    p.add_argument("--seeds", default="13711,13721,13731")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def tail_stats(scores: np.ndarray, y_true: np.ndarray) -> dict:
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    q95 = float(np.percentile(normal_scores, 95)) if len(normal_scores) else 0.0
    q99 = float(np.percentile(normal_scores, 99)) if len(normal_scores) else 0.0
    tpr95 = float(np.mean(anomaly_scores > q95)) if len(anomaly_scores) else 0.0
    tpr99 = float(np.mean(anomaly_scores > q99)) if len(anomaly_scores) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "TPR@95n": tpr95,
        "TPR@99n": tpr99,
        "anomaly_median": float(np.median(anomaly_scores)) if len(anomaly_scores) else 0.0,
        "anomaly_q75": float(np.percentile(anomaly_scores, 75)) if len(anomaly_scores) else 0.0,
    }


def fmt(x): return f"{x:.6f}"


def find_beats_checkpoint(explicit: str) -> Path | None:
    p = Path(explicit)
    return p if p.exists() else None


def load_beats(ckpt_path: Path, device: str):
    beats_dir = ckpt_path.parent
    if str(beats_dir) not in sys.path:
        sys.path.insert(0, str(beats_dir))
    from BEATs import BEATs, BEATsConfig

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=requested)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(requested)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def beats_embed(model, waveform_list: list[np.ndarray], device: torch.device, batch_size: int = 16):
    embs = []
    with torch.no_grad():
        for start in range(0, len(waveform_list), batch_size):
            batch = waveform_list[start:start + batch_size]
            max_len = max(len(w) for w in batch)
            x = np.zeros((len(batch), max_len), dtype=np.float32)
            for i, w in enumerate(batch):
                x[i, :len(w)] = w.astype(np.float32)
            wav = torch.from_numpy(x).to(device)
            feats, _ = model.extract_features(wav)
            emb = feats.mean(dim=1)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float64)


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5):
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, len(train_n))
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


def expected_test_reference(data_root: Path, machine: str) -> dict:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return {
        "paths": np.asarray([str(x.path) for x in test_items], dtype=str),
        "y_test": np.asarray([x.label for x in test_items], dtype=np.int64),
        "domains": np.asarray([x.domain for x in test_items], dtype=str),
        "test_items": test_items,
    }


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def evaluate_route(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_stats(scores, y_test)
    return {
        "hmean": met["hmean"],
        "pAUC": met["pauc"],
        "KS": geom["source_target_ks"],
        "TPR@95n": tail["TPR@95n"],
        "TPR@99n": tail["TPR@99n"],
        "q95": tail["q95"],
        "q99": tail["q99"],
        "anomaly_median": tail["anomaly_median"],
        "anomaly_q75": tail["anomaly_q75"],
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    status = {
        "mode": "check",
        "beats_checkpoint_exists": Path(args.beats_checkpoint).exists(),
        "v63_root_exists": Path(args.v63_root).exists(),
        "data_root_exists": Path(args.data_root).exists(),
        "v37_root_exists": Path(args.v37_root).exists(),
        "v61_root_exists": Path(args.v61_root).exists(),
    }
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")
    Path(args.doc_path).write_text(
        "# CTAF Stage14A Frozen BEATs + V63 Fusion Audit\n\n"
        "Mode=check completed. No model loaded, no embedding extracted, no metric computed.\n"
    )


def run_a0(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = find_beats_checkpoint(args.beats_checkpoint)
    if ckpt is None:
        raise FileNotFoundError(f"BEATs checkpoint missing: {args.beats_checkpoint}")

    model, device = load_beats(ckpt, args.device)
    items = scan_dataset(Path(args.data_root))
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]

    per_machine = []
    tail_rows = []
    summary_rows = []

    for machine in machines:
        ref = expected_test_reference(Path(args.data_root), machine)
        train_items, test_items = split_train_test(filter_items(items, machine), use_target_train=False)
        train_wavs = [load_audio(item.path) for item in train_items]
        test_wavs = [load_audio(item.path) for item in test_items]
        train_emb = beats_embed(model, train_wavs, device)
        test_emb = beats_embed(model, test_wavs, device)
        scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
        ev = evaluate_route(scores, ref["y_test"], ref["domains"])

        machine_dir = out_root / "a0_frozen_beats" / machine
        machine_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            machine_dir / f"{machine}_beats_embeddings.npz",
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            test_scores=scores,
            y_test=ref["y_test"],
            domains=ref["domains"],
            test_paths=ref["paths"],
            beats_checkpoint=str(ckpt),
        )
        score_dir = out_root / "a0_frozen_beats" / "scores"
        score_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            score_dir / f"{machine}_frozen_beats_knn_scores.npz",
            test_scores=scores,
            y_test=ref["y_test"],
            domains=ref["domains"],
            test_paths=ref["paths"],
            score_mode="frozen_beats_knn",
            beats_checkpoint=str(ckpt),
        )
        per_machine.append({
            "machine": machine,
            "hmean": ev["hmean"],
            "pAUC": ev["pAUC"],
            "KS": ev["KS"],
            "TPR@99n": ev["TPR@99n"],
        })
        tail_rows.append({"machine": machine, **tail_stats(scores, ref["y_test"])})
        summary_rows.append({"machine": machine, "embedding_dim": int(train_emb.shape[1]), "train_count": len(train_items), "test_count": len(test_items)})

    write_csv(out_root / "a0_frozen_beats" / "per_machine.csv", per_machine)
    write_csv(out_root / "a0_frozen_beats" / "tail_table.csv", tail_rows)
    write_csv(out_root / "a0_frozen_beats" / "embedding_summary.csv", summary_rows)
    write_csv(out_root / "a0_frozen_beats" / "summary.csv", [{
        "route": "frozen_BEATs_knn",
        "mean_hmean": float(np.mean([r["hmean"] for r in per_machine])),
        "mean_pAUC": float(np.mean([r["pAUC"] for r in per_machine])),
        "mean_KS": float(np.mean([r["KS"] for r in per_machine])),
        "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in per_machine])),
        "beats_checkpoint": str(ckpt),
        "embedding_dim": int(summary_rows[0]["embedding_dim"]) if summary_rows else 0,
    }])

    Path(args.doc_path).write_text(
        "# CTAF Stage14A Frozen BEATs + V63 Fusion Audit\n\n"
        "Mode=a0_frozen_beats completed.\n"
    )


def run_a1(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    alignment_rows = []
    seed_rows = []
    per_machine_source_rows = []
    comp_rows = []
    alignment_issue = False

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        for seed in seeds:
            v37_npz = Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v61_npz = Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v63_npz = Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"
            beats_npz = out_root / "a0_frozen_beats" / "scores" / f"{machine}_frozen_beats_knn_scores.npz"
            row = {
                "seed": seed,
                "machine": machine,
                "v63_exists": v63_npz.exists(),
                "beats_exists": beats_npz.exists(),
                "y_test_match": False,
                "domains_match": False,
                "sample_order_match": False,
                "alignment_ok": False,
                "issue": "",
            }
            if not v63_npz.exists() or not beats_npz.exists():
                row["issue"] = "missing_score_file"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            b37 = load_bundle(v37_npz)
            b61 = load_bundle(v61_npz)
            b63 = load_bundle(v63_npz)
            bb = load_bundle(beats_npz)
            y_match = np.array_equal(b63["y_test"], bb["y_test"]) and np.array_equal(b63["y_test"], expected["y_test"])
            d_match = np.array_equal(b63["domains"], bb["domains"]) and np.array_equal(b63["domains"], expected["domains"])
            paths_match = b63["test_paths"] is not None and bb["test_paths"] is not None and np.array_equal(b63["test_paths"], bb["test_paths"]) and np.array_equal(b63["test_paths"], expected["paths"])
            row["y_test_match"] = y_match
            row["domains_match"] = d_match
            row["sample_order_match"] = paths_match
            row["alignment_ok"] = y_match and d_match and paths_match
            if not row["alignment_ok"]:
                row["issue"] = "alignment_mismatch"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            alignment_rows.append(row)

            c0 = b63["test_scores"]
            c1 = bb["test_scores"]
            c2 = 0.5 * rank_norm(c0) + 0.5 * rank_norm(c1)
            y_test = b63["y_test"]
            domains = b63["domains"]
            for cand, scores in [("V37", b37["test_scores"]), ("V61", b61["test_scores"]), ("V63", c0), ("frozen_BEATs", c1), ("V63_plus_BEATs_fixed_rankfusion", c2)]:
                ev = evaluate_route(scores, y_test, domains)
                seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, **ev})

            q99_v63 = tail_stats(c0, y_test)["q99"]
            q99_b = tail_stats(c1, y_test)["q99"]
            normal_mask = y_test == 0
            anomaly_mask = y_test == 1
            top_v63 = set(np.where(normal_mask & (c0 >= q99_v63))[0].tolist())
            top_b = set(np.where(normal_mask & (c1 >= q99_b))[0].tolist())
            missed_v63 = set(np.where(anomaly_mask & (c0 <= q99_v63))[0].tolist())
            missed_b = set(np.where(anomaly_mask & (c1 <= q99_b))[0].tolist())
            top_union = len(top_v63 | top_b)
            miss_union = len(missed_v63 | missed_b)
            sp = spearmanr(c0, c1)
            comp_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_r": float(sp.correlation) if sp.correlation == sp.correlation else 0.0,
                "spearman_p": float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0,
                "top_normal_tail_jaccard": float(len(top_v63 & top_b) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_v63 & missed_b) / miss_union) if miss_union else 1.0,
                "v63_detect_beats_miss": int(np.sum(anomaly_mask & (c0 > q99_v63) & (c1 <= q99_b))),
                "beats_detect_v63_miss": int(np.sum(anomaly_mask & (c1 > q99_b) & (c0 <= q99_v63))),
            })
            fused_dir = out_root / "a1_v63_beats_fusion" / f"seed_{seed}" / "scores"
            fused_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                fused_dir / f"{machine}_v63_beats_rankfusion_scores.npz",
                test_scores=c2,
                y_test=y_test,
                domains=domains,
                test_paths=expected["paths"],
                score_mode="v63_beats_fixed_rankfusion",
                fusion_formula="0.5 * rank_norm(V63) + 0.5 * rank_norm(BEATs)",
                source_v63_score_path=str(v63_npz),
                source_beats_score_path=str(beats_npz),
            )

    write_csv(out_root / "a1_v63_beats_fusion" / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        Path(args.doc_path).write_text(
            "# CTAF Stage14A Frozen BEATs + V63 Fusion Audit\n\n"
            "Mode=a1_v63_beats_fusion failed due to alignment issue.\n"
        )
        return

    write_csv(out_root / "a1_v63_beats_fusion" / "seed_metrics.csv", seed_rows)
    write_csv(out_root / "a1_v63_beats_fusion" / "complementarity_diagnostics.csv", comp_rows)

    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    by_machine_comp = defaultdict(list)
    for row in seed_rows:
        by_seed_cand[(row["seed"], row["candidate"])].append(row)
        by_machine_cand[(row["machine"], row["candidate"])].append(row)
    for row in comp_rows:
        by_machine_comp[row["machine"]].append(row)

    summary_rows = []
    for (seed, cand), rows in sorted(by_seed_cand.items()):
        summary_rows.append({
            "seed": seed,
            "candidate": cand,
            "hmean": float(np.mean([r["hmean"] for r in rows])),
            "pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "KS": float(np.mean([r["KS"] for r in rows])),
            "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "summary.csv", summary_rows)

    aggregate_rows = []
    for cand in ["V37", "V61", "V63", "frozen_BEATs", "V63_plus_BEATs_fixed_rankfusion"]:
        rows = [r for r in summary_rows if r["candidate"] == cand]
        aggregate_rows.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })

    per_machine_rows = []
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
            "std_TPR@99n": float(np.std([r["TPR@99n"] for r in rows], ddof=0)),
        })
    write_csv(out_root / "per_machine.csv", per_machine_rows)

    tail_rows = []
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        tail_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_q95": float(np.mean([r["q95"] for r in rows])),
            "mean_q99": float(np.mean([r["q99"] for r in rows])),
            "mean_anomaly_median": float(np.mean([r["anomaly_median"] for r in rows])),
            "mean_anomaly_q75": float(np.mean([r["anomaly_q75"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "tail_table.csv", tail_rows)

    comp_summary_rows = []
    for machine, rows in sorted(by_machine_comp.items()):
        comp_summary_rows.append({
            "machine": machine,
            "mean_spearman_r": float(np.mean([r["spearman_r"] for r in rows])),
            "mean_top_normal_tail_jaccard": float(np.mean([r["top_normal_tail_jaccard"] for r in rows])),
            "mean_missed_anomaly_jaccard": float(np.mean([r["missed_anomaly_jaccard"] for r in rows])),
            "sum_v63_detect_beats_miss": int(np.sum([r["v63_detect_beats_miss"] for r in rows])),
            "sum_beats_detect_v63_miss": int(np.sum([r["beats_detect_v63_miss"] for r in rows])),
        })
    write_csv(out_root / "complementarity_summary.csv", comp_summary_rows)

    agg = {r["candidate"]: r for r in aggregate_rows}
    pm = {(r["machine"], r["candidate"]): r for r in per_machine_rows}
    v63 = agg["V63"]
    fused = agg["V63_plus_BEATs_fixed_rankfusion"]
    bottleneck_improved = sum(
        pm[(m, "V63_plus_BEATs_fixed_rankfusion")]["mean_TPR@99n"] > pm[(m, "V63")]["mean_TPR@99n"]
        for m in BOTTLENECK
    )
    pass_flags = {
        "hmean >= V63 + 0.01": fused["mean_hmean"] >= v63["mean_hmean"] + 0.01,
        "pAUC >= V63 - 0.01": fused["mean_pAUC"] >= v63["mean_pAUC"] - 0.01,
        "improvement not from one machine only": sum(pm[(m, "V63_plus_BEATs_fixed_rankfusion")]["mean_hmean"] > pm[(m, "V63")]["mean_hmean"] for m in machines) > 1,
        "bearing / valve do not collapse": pm[("bearing", "V63_plus_BEATs_fixed_rankfusion")]["mean_hmean"] >= 0.50 and pm[("valve", "V63_plus_BEATs_fixed_rankfusion")]["mean_hmean"] >= 0.50,
        "toycar does not severely degrade": pm[("toycar", "V63_plus_BEATs_fixed_rankfusion")]["mean_hmean"] >= pm[("toycar", "V63")]["mean_hmean"] - 0.03,
        "no anomaly/test label selection": True,
        "no weight tuning": True,
        "no per-machine selection": True,
        ">=2 bottleneck TPR@99n improve over V63": bottleneck_improved >= 2,
    }
    verdict = "Stage14A A1 fail"
    if all(pass_flags.values()):
        verdict = "Stage14A A1 pass"
    elif fused["mean_hmean"] >= v63["mean_hmean"] + 0.01 and fused["mean_pAUC"] < v63["mean_pAUC"] - 0.01:
        verdict = "hmean-positive but pAUC-negative"

    lines = [
        "# CTAF Stage14A Frozen BEATs + V63 Fusion Audit",
        "",
        "## 1. Mode Summary",
        "",
        "- A0: frozen BEATs baseline",
        "- A1: fixed V63 + frozen BEATs rank fusion",
        "- no training",
        "- no weight tuning",
        "- no per-machine selection",
        "",
        "## 2. Alignment Check",
        "",
        "- V63 score files: present",
        "- frozen BEATs score files: present",
        "- y_test alignment: PASS",
        "- domains alignment: PASS",
        "- sample order alignment: PASS",
        "- test file order alignment: PASS",
        "",
        "## 3. 3-Seed Comparison",
        "",
        "| route | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        lines.append(
            f"| {r['candidate']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} | "
            f"{r['mean_TPR@95n']:.6f} | {r['mean_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 4. Per-Machine Comparison",
        "",
        "| machine | route | mean hmean | mean pAUC | mean KS | mean TPR@99n |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for r in per_machine_rows:
        lines.append(
            f"| {r['machine']} | {r['candidate']} | {r['mean_hmean']:.6f} | {r['mean_pAUC']:.6f} | "
            f"{r['mean_KS']:.6f} | {r['mean_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 5. Complementarity Diagnostics",
        "",
        "| machine | mean spearman_r | mean top normal-tail jaccard | mean missed-anomaly jaccard | V63-detect / BEATs-miss | BEATs-detect / V63-miss |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in comp_summary_rows:
        lines.append(
            f"| {r['machine']} | {r['mean_spearman_r']:.6f} | {r['mean_top_normal_tail_jaccard']:.6f} | "
            f"{r['mean_missed_anomaly_jaccard']:.6f} | {r['sum_v63_detect_beats_miss']} | {r['sum_beats_detect_v63_miss']} |"
        )
    lines.extend([
        "",
        "## 6. Pass/Fail Gates",
        "",
        "| gate | verdict |",
        "|---|---|",
    ])
    for k, v in pass_flags.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines.extend([
        "",
        "## 7. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "## 8. Confirmation",
        "",
        "- no training occurred",
        "- no BEATs fine-tuning",
        "- no original V37/V61/V63 score file modification",
        "- no weight tuning",
        "- no per-machine selection",
        "- no anomaly/test label selection",
        "- no Stage 14B opened",
    ])
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "a0_frozen_beats":
        run_a0(args)
        return
    if args.mode == "a1_v63_beats_fusion":
        run_a1(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
