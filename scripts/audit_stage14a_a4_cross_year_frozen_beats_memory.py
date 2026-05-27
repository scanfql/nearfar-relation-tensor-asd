#!/usr/bin/env python
"""Stage 14A-A4: cross-year frozen BEATs memory-support audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
    p = argparse.ArgumentParser(description="Stage 14A-A4 cross-year frozen BEATs memory audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_cross_year_beats", "score_cross_year_beats", "fuse_v63_cross_year_beats"])
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--beats_checkpoint", default=DEFAULT_BEATS_CKPT)
    p.add_argument("--manifest_path", default="results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--a0_root", default="results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats")
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


def fmt(x): return f"{x:.6f}"


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


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
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


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


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
        "train_items": [x for x in items if x.split == "train" and x.label in (0, None)],
        "test_items": test_items,
    }


def load_manifest(args) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest_path)
    manifest = manifest[manifest["usable_for_v61"] == True].copy()
    return manifest


def split_manifest_by_memory(manifest: pd.DataFrame, machine: str):
    machine_df = manifest[manifest["machine_canonical"] == machine].copy()
    source_df = machine_df[machine_df["domain"] == "source"].copy()
    full_df = machine_df.copy()
    return source_df, full_df


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    by_year_machine = []
    for machine in machines:
        src_df, full_df = split_manifest_by_memory(manifest, machine)
        src_counts = src_df.groupby("year").size().to_dict()
        full_counts = full_df.groupby("year").size().to_dict()
        years = sorted(set(src_counts) | set(full_counts))
        for year in years:
            by_year_machine.append({
                "machine": machine,
                "year": int(year),
                "source_only_count": int(src_counts.get(year, 0)),
                "full_manifest_count": int(full_counts.get(year, 0)),
            })
    write_csv(out_root / "check_manifest_counts.csv", by_year_machine)
    status = {
        "mode": "check",
        "beats_checkpoint_exists": Path(args.beats_checkpoint).exists(),
        "manifest_exists": Path(args.manifest_path).exists(),
        "v63_root_exists": Path(args.v63_root).exists(),
        "a0_root_exists": Path(args.a0_root).exists(),
        "data_root_exists": Path(args.data_root).exists(),
    }
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")
    Path(args.doc_path).write_text(
        "# CTAF Stage14A A4 Cross-Year Frozen BEATs Memory Audit\n\n"
        "Mode=check completed. No BEATs loaded, no embeddings extracted, no scoring performed.\n"
    )


def run_extract(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    ckpt = find_beats_checkpoint(args.beats_checkpoint)
    if ckpt is None:
        raise FileNotFoundError(f"Missing BEATs checkpoint: {args.beats_checkpoint}")
    model, device = load_beats(ckpt, args.device)
    summary_rows = []

    for machine in machines:
        src_df, full_df = split_manifest_by_memory(manifest, machine)
        source_paths = src_df["absolute_path"].tolist()
        full_paths = full_df["absolute_path"].tolist()
        source_wavs = [load_audio(Path(p)) for p in source_paths]
        full_wavs = [load_audio(Path(p)) for p in full_paths]
        source_emb = beats_embed(model, source_wavs, device)
        full_emb = beats_embed(model, full_wavs, device)

        machine_out = out_root / "extract_cross_year_beats" / machine
        machine_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            machine_out / f"{machine}_b2_source_memory_embeddings.npz",
            embeddings=source_emb,
            paths=np.asarray(source_paths, dtype=object),
            beats_checkpoint=str(ckpt),
            memory_mode="source_only",
        )
        np.savez_compressed(
            machine_out / f"{machine}_b2_full_memory_embeddings.npz",
            embeddings=full_emb,
            paths=np.asarray(full_paths, dtype=object),
            beats_checkpoint=str(ckpt),
            memory_mode="full_manifest",
        )

        # reuse A0 current test embeddings if available; otherwise extract
        a0_emb_path = Path(args.a0_root) / machine / f"{machine}_beats_embeddings.npz"
        if a0_emb_path.exists():
            arr = np.load(a0_emb_path, allow_pickle=True)
            test_emb = arr["test_embeddings"].astype(np.float64)
            y_test = arr["y_test"].astype(np.int64)
            domains = arr["domains"].astype(str)
            test_paths = arr["test_paths"].astype(str)
            extracted_test = False
        else:
            ref = expected_test_reference(Path(args.data_root), machine)
            test_wavs = [load_audio(item.path) for item in ref["test_items"]]
            test_emb = beats_embed(model, test_wavs, device)
            y_test = ref["y_test"]
            domains = ref["domains"]
            test_paths = ref["paths"]
            extracted_test = True
        np.savez_compressed(
            machine_out / f"{machine}_current_test_embeddings.npz",
            test_embeddings=test_emb,
            y_test=y_test,
            domains=domains,
            test_paths=test_paths,
            beats_checkpoint=str(ckpt),
            extracted_test_embeddings=extracted_test,
        )
        summary_rows.append({
            "machine": machine,
            "source_memory_count": len(source_paths),
            "full_memory_count": len(full_paths),
            "embedding_dim": int(source_emb.shape[1]) if len(source_emb) else 0,
            "reused_a0_test_embeddings": bool(not extracted_test),
        })
    write_csv(out_root / "embedding_summary.csv", summary_rows)
    Path(args.doc_path).write_text(
        "# CTAF Stage14A A4 Cross-Year Frozen BEATs Memory Audit\n\n"
        "Mode=extract_cross_year_beats completed.\n"
    )


def run_score(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    per_machine = []
    tail_rows = []
    summary_rows = []
    score_dir = out_root / "b2_cross_year_beats" / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    a0_summary = []
    for machine in machines:
        source_path = out_root / "extract_cross_year_beats" / machine / f"{machine}_b2_source_memory_embeddings.npz"
        full_path = out_root / "extract_cross_year_beats" / machine / f"{machine}_b2_full_memory_embeddings.npz"
        test_path = out_root / "extract_cross_year_beats" / machine / f"{machine}_current_test_embeddings.npz"
        if not source_path.exists() or not full_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing extracted embeddings for {machine}")
        source_arr = np.load(source_path, allow_pickle=True)
        full_arr = np.load(full_path, allow_pickle=True)
        test_arr = np.load(test_path, allow_pickle=True)
        source_emb = source_arr["embeddings"].astype(np.float64)
        full_emb = full_arr["embeddings"].astype(np.float64)
        test_emb = test_arr["test_embeddings"].astype(np.float64)
        y_test = test_arr["y_test"].astype(np.int64)
        domains = test_arr["domains"].astype(str)
        test_paths = test_arr["test_paths"].astype(str)

        score_s = knn_cosine_scores(source_emb, test_emb, k=args.latent_knn_k)
        score_f = knn_cosine_scores(full_emb, test_emb, k=args.latent_knn_k)
        ev_s = evaluate(score_s, y_test, domains)
        ev_f = evaluate(score_f, y_test, domains)
        np.savez(
            score_dir / f"{machine}_b2_source_beats_scores.npz",
            test_scores=score_s,
            y_test=y_test,
            domains=domains,
            test_paths=test_paths,
            score_mode="b2_source_beats",
        )
        np.savez(
            score_dir / f"{machine}_b2_full_beats_scores.npz",
            test_scores=score_f,
            y_test=y_test,
            domains=domains,
            test_paths=test_paths,
            score_mode="b2_full_beats",
        )
        per_machine.extend([
            {"machine": machine, "candidate": "B2s", **{k: ev_s[k] for k in ["hmean", "pAUC", "KS", "TPR@99n"]}},
            {"machine": machine, "candidate": "B2f", **{k: ev_f[k] for k in ["hmean", "pAUC", "KS", "TPR@99n"]}},
        ])
        tail_rows.extend([
            {"machine": machine, "candidate": "B2s", **tail_stats(score_s, y_test)},
            {"machine": machine, "candidate": "B2f", **tail_stats(score_f, y_test)},
        ])
        summary_rows.append({
            "machine": machine,
            "source_memory_count": source_emb.shape[0],
            "full_memory_count": full_emb.shape[0],
            "test_count": test_emb.shape[0],
        })

    write_csv(out_root / "b2_cross_year_beats" / "per_machine.csv", per_machine)
    write_csv(out_root / "b2_cross_year_beats" / "tail_table.csv", tail_rows)
    write_csv(out_root / "b2_cross_year_beats" / "embedding_summary.csv", summary_rows)
    write_csv(out_root / "b2_cross_year_beats" / "summary.csv", [
        {
            "candidate": "B2s",
            "mean_hmean": float(np.mean([r["hmean"] for r in per_machine if r["candidate"] == "B2s"])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in per_machine if r["candidate"] == "B2s"])),
            "mean_KS": float(np.mean([r["KS"] for r in per_machine if r["candidate"] == "B2s"])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in per_machine if r["candidate"] == "B2s"])),
        },
        {
            "candidate": "B2f",
            "mean_hmean": float(np.mean([r["hmean"] for r in per_machine if r["candidate"] == "B2f"])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in per_machine if r["candidate"] == "B2f"])),
            "mean_KS": float(np.mean([r["KS"] for r in per_machine if r["candidate"] == "B2f"])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in per_machine if r["candidate"] == "B2f"])),
        },
    ])
    Path(args.doc_path).write_text(
        "# CTAF Stage14A A4 Cross-Year Frozen BEATs Memory Audit\n\n"
        "Mode=score_cross_year_beats completed.\n"
    )


def run_fuse(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    alignment_rows = []
    seed_rows = []
    comp_rows = []
    alignment_issue = False

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        b2s_npz = out_root / "b2_cross_year_beats" / "scores" / f"{machine}_b2_source_beats_scores.npz"
        b2f_npz = out_root / "b2_cross_year_beats" / "scores" / f"{machine}_b2_full_beats_scores.npz"
        if not b2s_npz.exists() or not b2f_npz.exists():
            raise FileNotFoundError(f"Missing B2 score for {machine}")
        b2s = load_bundle(b2s_npz)
        b2f = load_bundle(b2f_npz)
        for seed in seeds:
            v63_npz = Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"
            row = {
                "seed": seed,
                "machine": machine,
                "v63_exists": v63_npz.exists(),
                "b2s_exists": b2s_npz.exists(),
                "b2f_exists": b2f_npz.exists(),
                "y_test_match": False,
                "domains_match": False,
                "sample_order_match": False,
                "alignment_ok": False,
                "issue": "",
            }
            if not v63_npz.exists():
                row["issue"] = "missing_score_file"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            v63 = load_bundle(v63_npz)
            y_match = (
                np.array_equal(v63["y_test"], b2s["y_test"])
                and np.array_equal(v63["y_test"], b2f["y_test"])
                and np.array_equal(v63["y_test"], expected["y_test"])
            )
            d_match = (
                np.array_equal(v63["domains"], b2s["domains"])
                and np.array_equal(v63["domains"], b2f["domains"])
                and np.array_equal(v63["domains"], expected["domains"])
            )
            paths_match = (
                v63["test_paths"] is not None and b2s["test_paths"] is not None and b2f["test_paths"] is not None
                and np.array_equal(v63["test_paths"], b2s["test_paths"])
                and np.array_equal(v63["test_paths"], b2f["test_paths"])
                and np.array_equal(v63["test_paths"], expected["paths"])
            )
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

            c_v63 = v63["test_scores"]
            c_s = b2s["test_scores"]
            c_f = b2f["test_scores"]
            y_test = v63["y_test"]
            domains = v63["domains"]
            c_f2s = 0.5 * rank_norm(c_v63) + 0.5 * rank_norm(c_s)
            c_f2f = 0.5 * rank_norm(c_v63) + 0.5 * rank_norm(c_f)
            for cand, scores in [("V63", c_v63), ("B2s", c_s), ("B2f", c_f), ("F2s", c_f2s), ("F2f", c_f2f)]:
                ev = evaluate(scores, y_test, domains)
                seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, **ev})

            q99_v63 = tail_stats(c_v63, y_test)["q99"]
            q99_s = tail_stats(c_s, y_test)["q99"]
            normal_mask = y_test == 0
            anomaly_mask = y_test == 1
            top_v63 = set(np.where(normal_mask & (c_v63 >= q99_v63))[0].tolist())
            top_s = set(np.where(normal_mask & (c_s >= q99_s))[0].tolist())
            missed_v63 = set(np.where(anomaly_mask & (c_v63 <= q99_v63))[0].tolist())
            missed_s = set(np.where(anomaly_mask & (c_s <= q99_s))[0].tolist())
            top_union = len(top_v63 | top_s)
            miss_union = len(missed_v63 | missed_s)
            sp = spearmanr(c_v63, c_s)
            comp_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_r": float(sp.correlation) if sp.correlation == sp.correlation else 0.0,
                "spearman_p": float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0,
                "top_normal_tail_jaccard": float(len(top_v63 & top_s) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_v63 & missed_s) / miss_union) if miss_union else 1.0,
                "v63_detect_b2s_miss": int(np.sum(anomaly_mask & (c_v63 > q99_v63) & (c_s <= q99_s))),
                "b2s_detect_v63_miss": int(np.sum(anomaly_mask & (c_s > q99_s) & (c_v63 <= q99_v63))),
            })
            for name, scores in [("F2s", c_f2s), ("F2f", c_f2f)]:
                fused_dir = out_root / "f2_v63_cross_year_beats_fusion" / f"seed_{seed}" / "scores"
                fused_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    fused_dir / f"{machine}_{name.lower()}_scores.npz",
                    test_scores=scores,
                    y_test=y_test,
                    domains=domains,
                    test_paths=expected["paths"],
                    score_mode=name.lower(),
                    fusion_formula=f"0.5 * rank_norm(V63) + 0.5 * rank_norm({'B2s' if name=='F2s' else 'B2f'})",
                )

    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        Path(args.doc_path).write_text(
            "# CTAF Stage14A A4 Cross-Year Frozen BEATs Memory Audit\n\n"
            "Mode=fuse_v63_cross_year_beats failed due to alignment issue.\n"
        )
        return

    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "seed_metrics.csv", seed_rows)
    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "complementarity_diagnostics.csv", comp_rows)

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
    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "summary.csv", summary_rows)

    per_machine_rows = []
    tail_rows = []
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
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
    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "tail_table.csv", tail_rows)

    comp_summary_rows = []
    for machine, rows in sorted(by_machine_comp.items()):
        comp_summary_rows.append({
            "machine": machine,
            "mean_spearman_r": float(np.mean([r["spearman_r"] for r in rows])),
            "mean_top_normal_tail_jaccard": float(np.mean([r["top_normal_tail_jaccard"] for r in rows])),
            "mean_missed_anomaly_jaccard": float(np.mean([r["missed_anomaly_jaccard"] for r in rows])),
            "sum_v63_detect_b2s_miss": int(np.sum([r["v63_detect_b2s_miss"] for r in rows])),
            "sum_b2s_detect_v63_miss": int(np.sum([r["b2s_detect_v63_miss"] for r in rows])),
        })
    write_csv(out_root / "f2_v63_cross_year_beats_fusion" / "complementarity_summary.csv", comp_summary_rows)

    Path(args.doc_path).write_text(
        "# CTAF Stage14A A4 Cross-Year Frozen BEATs Memory Audit\n\n"
        "Mode=fuse_v63_cross_year_beats completed.\n"
    )


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_cross_year_beats":
        run_extract(args)
        return
    if args.mode == "score_cross_year_beats":
        run_score(args)
        return
    if args.mode == "fuse_v63_cross_year_beats":
        run_fuse(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
