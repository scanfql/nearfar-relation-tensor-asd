#!/usr/bin/env python
"""Stage 19I: fan anchor + distance-geometry LoRA evaluation.

Fan-only evaluation of the Stage19H checkpoint.
No selector, no fusion, no full7, no training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - check mode should report import status
    np = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_stage19f_helpers():
    stage19f_path = ROOT / "scripts" / "audit_stage19f_fan_anchor_lora_eval.py"
    spec = importlib.util.spec_from_file_location("stage19f_helpers", stage19f_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Stage19F helpers from {stage19f_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


stage19f = load_stage19f_helpers()


EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ADAPTED_CKPT_DEFAULT = "results/stage19h_fan_anchor_geometry_lora_sanity/checkpoints/fan_stage19h_anchor_geometry_lora_best.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
STAGE19H_ROOT_DEFAULT = "results/stage19h_fan_anchor_geometry_lora_sanity"
STAGE18E_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
STAGE18F_ROOT_DEFAULT = "results/stage18f_ldn_diff_k1_route_freeze"
OUTPUT_ROOT_DEFAULT = "results/stage19i_fan_anchor_geometry_lora_eval"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE19I_FAN_ANCHOR_GEOMETRY_LORA_EVAL_AUDIT.md"
EMBEDDING_NPZ_DEFAULT = "results/stage19i_fan_anchor_geometry_lora_eval/embeddings/fan_anchor_geometry_lora_minclamp_mean_embeddings.npz"
MACHINE_DEFAULT = "fan"

STAGE18_BASELINE_HMEAN = 0.5820070015781968
STAGE18_BASELINE_PAUC = 0.5421052631578948
STAGE18_BASELINE_TPR99 = 0.04

STAGE18_LDN_HMEAN = 0.5527191284766243
STAGE18_LDN_PAUC = 0.5252631578947369
STAGE18_LDN_TPR99 = 0.02

STAGE19F_LDN_HMEAN = 0.5501054732664112
STAGE19F_LDN_PAUC = 0.5284210526315789
STAGE19F_LDN_TPR99 = 0.02

EXPECTED_LORA_TARGET_NAMES = [
    "model.blocks.11.attn.qkv",
    "model.blocks.11.attn.proj",
]

REPRESENTATION_GUARDRAIL_MAX_DROP = 0.02

CFG = stage19f.CFG
LoRAConfig = stage19f.LoRAConfig

write_json = stage19f.write_json
write_csv = stage19f.write_csv
write_markdown = stage19f.write_markdown
import_status = stage19f.import_status
load_json = stage19f.load_json
ensure_writable = stage19f.ensure_writable
expected_items = stage19f.expected_items
load_clean_eat_local = stage19f.load_clean_eat_local
freeze_all_parameters = stage19f.freeze_all_parameters
get_parent_module = stage19f.get_parent_module
inspect_lora_targets = stage19f.inspect_lora_targets
inject_lora_modules = stage19f.inject_lora_modules
build_reconstruction_head = stage19f.build_reconstruction_head
parameter_audit_rows = stage19f.parameter_audit_rows
summarize_trainable = stage19f.summarize_trainable
load_waveform_16k = stage19f.load_waveform_16k
build_batch = stage19f.build_batch
extract_frame_output = stage19f.extract_frame_output
min_clamp_np = stage19f.min_clamp_np
mean_pool_np = stage19f.mean_pool_np
l2_normalize_np = stage19f.l2_normalize_np
chord_distance_matrix = stage19f.chord_distance_matrix
baseline_knn_scores = stage19f.baseline_knn_scores
local_density_knn = stage19f.local_density_knn
evaluate = stage19f.evaluate
tail_metrics = stage19f.tail_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Stage19I fan anchor+geometry LoRA evaluation")
    p.add_argument("--mode", required=True, choices=["check", "inspect_checkpoint", "extract_adapted_embeddings", "score_adapted_fan"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--adapted_ckpt_path", default=ADAPTED_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage19h_root", default=STAGE19H_ROOT_DEFAULT)
    p.add_argument("--stage18e_root", default=STAGE18E_ROOT_DEFAULT)
    p.add_argument("--stage18f_root", default=STAGE18F_ROOT_DEFAULT)
    p.add_argument("--embedding_npz_path", default=EMBEDDING_NPZ_DEFAULT)
    return p.parse_args()


def load_stage19h_checkpoint(model_path: Path, adapted_ckpt_path: Path, device: str):
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not adapted_ckpt_path.exists():
        raise FileNotFoundError(f"Missing Stage19H adapted checkpoint: {adapted_ckpt_path}")

    model, requested = load_clean_eat_local(model_path, device)
    freeze_all_parameters(model)
    lora_cfg = LoRAConfig()
    target_rows, selected_names = inspect_lora_targets(model, lora_cfg)
    if set(selected_names) != set(EXPECTED_LORA_TARGET_NAMES):
        raise RuntimeError(
            f"Stage19H LoRA targets do not match expected targets: {selected_names} vs {EXPECTED_LORA_TARGET_NAMES}"
        )
    model = inject_lora_modules(model, selected_names, lora_cfg)

    blob = torch.load(str(adapted_ckpt_path), map_location=requested)
    if not isinstance(blob, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(blob)}")
    if "student_state_dict" not in blob:
        raise RuntimeError("Stage19H checkpoint missing student_state_dict.")
    if "recon_head_state_dict" not in blob:
        raise RuntimeError("Stage19H checkpoint missing recon_head_state_dict.")

    checkpoint_targets = blob.get("lora_target_names", [])
    if checkpoint_targets and set(checkpoint_targets) != set(EXPECTED_LORA_TARGET_NAMES):
        raise RuntimeError(
            f"Checkpoint LoRA targets do not match expected targets: {checkpoint_targets} vs {EXPECTED_LORA_TARGET_NAMES}"
        )

    checkpoint_lora_cfg = blob.get("lora_config", {})
    if checkpoint_lora_cfg:
        expected_lora_cfg = {
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.05,
        }
        for key, expected_value in expected_lora_cfg.items():
            if float(checkpoint_lora_cfg.get(key, expected_value)) != float(expected_value):
                raise RuntimeError(
                    f"Checkpoint LoRA config mismatch for {key}: {checkpoint_lora_cfg.get(key)} vs {expected_value}"
                )

    train_cfg = blob.get("train_config", {})
    if train_cfg.get("trainable_scope") and train_cfg.get("trainable_scope") != "last_block_attention_lora":
        raise RuntimeError(f"Unexpected trainable_scope in checkpoint: {train_cfg.get('trainable_scope')}")
    if train_cfg.get("lambda_anchor") is not None and float(train_cfg.get("lambda_anchor")) != 1.0:
        raise RuntimeError(f"Unexpected lambda_anchor in checkpoint: {train_cfg.get('lambda_anchor')}")
    if train_cfg.get("lambda_dist") is not None and float(train_cfg.get("lambda_dist")) != 0.1:
        raise RuntimeError(f"Unexpected lambda_dist in checkpoint: {train_cfg.get('lambda_dist')}")

    model.load_state_dict(blob["student_state_dict"], strict=True)
    recon_head = build_reconstruction_head(requested, hidden_size=CFG.hidden_size, recon_dim=CFG.recon_dim)
    recon_head.load_state_dict(blob["recon_head_state_dict"], strict=True)
    model.eval()
    recon_head.eval()
    return {
        "model": model,
        "recon_head": recon_head,
        "device": requested,
        "blob": blob,
        "selected_names": selected_names,
        "target_rows": target_rows,
        "lora_cfg": lora_cfg,
    }


def build_check_status(args, data_root: Path, out_root: Path) -> dict:
    items, train_items, test_items = expected_items(data_root, args.machine) if data_root.exists() else ([], [], [])
    return {
        "mode": "check",
        "machine": args.machine,
        "eat_model_path_exists": Path(args.eat_model_path).exists(),
        "adapted_ckpt_path_exists": Path(args.adapted_ckpt_path).exists(),
        "data_root_exists": data_root.exists(),
        "stage19h_root_exists": Path(args.stage19h_root).exists(),
        "stage18e_root_exists": Path(args.stage18e_root).exists(),
        "stage18f_root_exists": Path(args.stage18f_root).exists(),
        "output_root_writable": ensure_writable(out_root),
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "inventory_count": len(items),
        "imports": import_status(),
        "notes": "No model load, no forward, no extraction, no scoring.",
    }


def build_model_load_status(model_path: Path, device, lora_cfg, selected_names, summary, trainable_rows, recon_head_created: bool):
    return {
        "clean_eat_checkpoint": str(model_path),
        "device": str(device),
        "selected_lora_target_names": selected_names,
        "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
        "lora_target_match": set(selected_names) == set(EXPECTED_LORA_TARGET_NAMES),
        "selected_lora_target_count": len(selected_names),
        "no_mlp_lora": all("mlp" not in name.lower() for name in selected_names),
        "no_earlier_block_lora": all("blocks.11." in name for name in selected_names),
        "recon_head_created": recon_head_created,
        "no_unexpected_trainable_base_params": summary["no_unexpected_trainable_base_params"],
        "unexpected_trainable_base_parameter_names": summary["unexpected_trainable_base_parameter_names"],
        "trainable_scope": "last_block_attention_lora",
        "lora_config": {
            "rank": int(lora_cfg.rank),
            "alpha": float(lora_cfg.alpha),
            "dropout": float(lora_cfg.dropout),
        },
        "trainable_summary": summary,
    }


def build_candidate_rows(summary_rows: list[dict]) -> list[dict]:
    by_candidate = {row["candidate"]: row for row in summary_rows}
    stage18_baseline = {
        "candidate": "stage18_baseline_knn_k5",
        "hmean": STAGE18_BASELINE_HMEAN,
        "pAUC": STAGE18_BASELINE_PAUC,
        "TPR@99n": STAGE18_BASELINE_TPR99,
        "delta_hmean_vs_stage18_ldn": STAGE18_BASELINE_HMEAN - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": 0.0,
        "delta_hmean_vs_stage19f": STAGE18_BASELINE_HMEAN - STAGE19F_LDN_HMEAN,
        "pass_minimum_gate": True,
        "pass_strong_gate": True,
    }
    stage18_ldn = {
        "candidate": "stage18_ldn_diff_K1_scorek5",
        "hmean": STAGE18_LDN_HMEAN,
        "pAUC": STAGE18_LDN_PAUC,
        "TPR@99n": STAGE18_LDN_TPR99,
        "delta_hmean_vs_stage18_ldn": 0.0,
        "delta_hmean_vs_stage18_baseline": STAGE18_LDN_HMEAN - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19f": STAGE18_LDN_HMEAN - STAGE19F_LDN_HMEAN,
        "pass_minimum_gate": False,
        "pass_strong_gate": False,
    }
    stage19f_ldn = {
        "candidate": "stage19f_anchor_lora_ldn_diff_K1",
        "hmean": STAGE19F_LDN_HMEAN,
        "pAUC": STAGE19F_LDN_PAUC,
        "TPR@99n": STAGE19F_LDN_TPR99,
        "delta_hmean_vs_stage18_ldn": STAGE19F_LDN_HMEAN - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": STAGE19F_LDN_HMEAN - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19f": 0.0,
        "pass_minimum_gate": False,
        "pass_strong_gate": False,
    }
    adapted_baseline_src = by_candidate["adapted_baseline_knn_k5"]
    adapted_ldn_src = by_candidate["adapted_ldn_diff_K1_scorek5"]
    adapted_baseline = {
        "candidate": "adapted_baseline_knn_k5",
        "hmean": adapted_baseline_src["hmean"],
        "pAUC": adapted_baseline_src["pAUC"],
        "TPR@99n": adapted_baseline_src["TPR@99n"],
        "delta_hmean_vs_stage18_ldn": adapted_baseline_src["hmean"] - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": adapted_baseline_src["hmean"] - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19f": adapted_baseline_src["hmean"] - STAGE19F_LDN_HMEAN,
        "pass_minimum_gate": adapted_baseline_src["hmean"] > STAGE18_LDN_HMEAN,
        "pass_strong_gate": adapted_baseline_src["hmean"] >= STAGE18_BASELINE_HMEAN,
    }
    adapted_ldn = {
        "candidate": "adapted_ldn_diff_K1_scorek5",
        "hmean": adapted_ldn_src["hmean"],
        "pAUC": adapted_ldn_src["pAUC"],
        "TPR@99n": adapted_ldn_src["TPR@99n"],
        "delta_hmean_vs_stage18_ldn": adapted_ldn_src["hmean"] - STAGE18_LDN_HMEAN,
        "delta_hmean_vs_stage18_baseline": adapted_ldn_src["hmean"] - STAGE18_BASELINE_HMEAN,
        "delta_hmean_vs_stage19f": adapted_ldn_src["hmean"] - STAGE19F_LDN_HMEAN,
        "pass_minimum_gate": adapted_ldn_src["hmean"] > STAGE18_LDN_HMEAN,
        "pass_strong_gate": adapted_ldn_src["hmean"] >= STAGE18_BASELINE_HMEAN,
    }
    return [stage18_baseline, stage18_ldn, stage19f_ldn, adapted_baseline, adapted_ldn]


def build_audit_markdown(verdict: dict, comparison_rows: list[dict], summary_rows: list[dict]) -> list[str]:
    adapted_baseline = next(row for row in summary_rows if row["candidate"] == "adapted_baseline_knn_k5")
    adapted_ldn = next(row for row in summary_rows if row["candidate"] == "adapted_ldn_diff_K1_scorek5")
    return [
        "# CTAF Stage19I Fan Anchor Geometry LoRA Evaluation Audit",
        "",
        "## Protocol",
        "",
        "- machine: `fan`",
        "- adapted checkpoint: `Stage19H anchor+distance-geometry LoRA`",
        "- evaluation data: `fan source train-normal / test split`",
        "- preprocess: `16 kHz`, `128-bin Kaldi fbank`, `frame shift 10 ms`, `target length 1024`",
        "- normalization: `(mel - (-4.268)) / (4.569 * 2)`",
        "- frame output: `[T, 768]`",
        "- min-clamp: `abs(x) < 0.1 -> 0`",
        "- pooling: `mean`",
        "- scoring candidates: `adapted_baseline_knn_k5`, `adapted_ldn_diff_K1_scorek5`",
        "- no selector, no oracle, no fusion, no full7",
        "- no checkpoint selection by test labels",
        "",
        "## Gates",
        "",
        f"- minimum pass: `adapted_ldn fan hmean > {STAGE18_LDN_HMEAN}`",
        f"- strong pass: `adapted_ldn fan hmean >= {STAGE18_BASELINE_HMEAN}`",
        f"- hard fail: `adapted_ldn fan hmean <= {STAGE18_LDN_HMEAN}`",
        f"- representation guardrail: `adapted_baseline fan hmean >= {STAGE18_BASELINE_HMEAN - REPRESENTATION_GUARDRAIL_MAX_DROP}`",
        "",
        "## Verdict",
        "",
        f"- adapted baseline hmean: `{adapted_baseline['hmean']:.6f}`",
        f"- adapted ldn hmean: `{adapted_ldn['hmean']:.6f}`",
        f"- minimum pass: `{verdict['minimum_pass']}`",
        f"- strong pass: `{verdict['strong_pass']}`",
        f"- representation guardrail pass: `{verdict['representation_guardrail_pass']}`",
        "",
        "## Comparison Rows",
        "",
        *[
            f"- {row['candidate']}: hmean={row['hmean']:.6f}, pAUC={row['pAUC']:.6f}, TPR@99n={row['TPR@99n']:.6f}"
            for row in comparison_rows
        ],
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    check_status = build_check_status(args, data_root, out_root)
    write_json(out_root / "check_status.json", check_status)


def run_inspect_checkpoint(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        artifact = load_stage19h_checkpoint(Path(args.eat_model_path), Path(args.adapted_ckpt_path), args.device)
        model = artifact["model"]
        recon_head = artifact["recon_head"]
        blob = artifact["blob"]
        selected_names = artifact["selected_names"]
        lora_cfg = artifact["lora_cfg"]
        audit_rows = parameter_audit_rows(model, recon_head)
        summary = summarize_trainable(audit_rows, selected_names, lora_cfg)

        checkpoint_status = {
            "mode": "inspect_checkpoint",
            "checkpoint_path": str(Path(args.adapted_ckpt_path)),
            "checkpoint_exists": True,
            "checkpoint_size_bytes": int(Path(args.adapted_ckpt_path).stat().st_size),
            "checkpoint_key_count": len(blob),
            "checkpoint_top_keys": list(blob.keys())[:20],
            "contains_student_state_dict": "student_state_dict" in blob,
            "contains_recon_head_state_dict": "recon_head_state_dict" in blob,
            "contains_train_config": "train_config" in blob,
            "contains_lora_config": "lora_config" in blob,
            "contains_lora_target_names": "lora_target_names" in blob,
            "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
            "checkpoint_lora_target_names": list(blob.get("lora_target_names", [])),
            "lora_target_match": set(blob.get("lora_target_names", [])) == set(EXPECTED_LORA_TARGET_NAMES),
            "lora_config_match": blob.get("lora_config", {}) == {"rank": 4, "alpha": 8.0, "dropout": 0.05},
        }
        lora_load_status = {
            "mode": "inspect_checkpoint",
            "checkpoint_path": str(Path(args.adapted_ckpt_path)),
            "loaded_successfully": True,
            "student_state_dict_loaded": True,
            "recon_head_state_dict_loaded": True,
            "strict_load": True,
            "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
            "selected_lora_target_names": selected_names,
            "checkpoint_top_keys": list(blob.keys())[:20],
            "checkpoint_key_count": len(blob),
            "lora_config": {"rank": int(lora_cfg.rank), "alpha": float(lora_cfg.alpha), "dropout": float(lora_cfg.dropout)},
            "trainable_scope": "last_block_attention_lora",
        }
        recap = {
            "checkpoint_path": str(Path(args.adapted_ckpt_path)),
            "expected_lora_target_names": EXPECTED_LORA_TARGET_NAMES,
            "checkpoint_lora_target_names": list(blob.get("lora_target_names", [])),
            "lora_target_match": set(blob.get("lora_target_names", [])) == set(EXPECTED_LORA_TARGET_NAMES),
            "checkpoint_lora_config": blob.get("lora_config", {}),
            "checkpoint_train_config_preview": {
                "trainable_scope": blob.get("train_config", {}).get("trainable_scope"),
                "mask_ratio": blob.get("train_config", {}).get("mask_ratio"),
                "lambda_anchor": blob.get("train_config", {}).get("lambda_anchor"),
                "lambda_dist": blob.get("train_config", {}).get("lambda_dist"),
                "epochs": blob.get("train_config", {}).get("epochs"),
                "lr": blob.get("train_config", {}).get("lr"),
                "early_stop_patience": blob.get("train_config", {}).get("early_stop_patience"),
            },
            "checkpoint_trainable_summary_preview": {
                "selected_lora_target_names": blob.get("trainable_summary", {}).get("selected_lora_target_names"),
                "no_unexpected_trainable_base_params": blob.get("trainable_summary", {}).get("no_unexpected_trainable_base_params"),
            },
            "parameter_summary": summary,
            "no_unexpected_trainable_base_params": summary["no_unexpected_trainable_base_params"],
            "unexpected_trainable_base_parameter_names": summary["unexpected_trainable_base_parameter_names"],
        }
        write_json(out_root / "checkpoint_status.json", checkpoint_status)
        write_json(out_root / "lora_load_status.json", lora_load_status)
        write_csv(out_root / "adapted_parameter_audit.csv", audit_rows)
        write_json(out_root / "trainable_boundary_recap.json", recap)
    except Exception as exc:
        write_json(
            out_root / "checkpoint_status.json",
            {
                "mode": "inspect_checkpoint",
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "loaded_successfully": False,
                "error": str(exc),
            },
        )
        write_json(
            out_root / "lora_load_status.json",
            {
                "mode": "inspect_checkpoint",
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "loaded_successfully": False,
                "error": str(exc),
            },
        )
        write_json(
            out_root / "trainable_boundary_recap.json",
            {
                "mode": "inspect_checkpoint",
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "loaded_successfully": False,
                "error": str(exc),
            },
        )
        raise


def run_extract_adapted_embeddings(args):
    import torch

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    artifact = load_stage19h_checkpoint(Path(args.eat_model_path), Path(args.adapted_ckpt_path), args.device)
    model = artifact["model"]
    device = artifact["device"]

    from src.data.dataset import filter_items, scan_dataset, split_train_test

    items = filter_items(scan_dataset(data_root), args.machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)}, test={len(test_items)}")

    train_embs = []
    test_embs = []
    y_test = []
    d_test = []
    test_paths = []
    sample_frame_shape = None

    with np.errstate(all="ignore"):
        for split_name, split_items, split_embs in [
            ("train", train_items, train_embs),
            ("test", test_items, test_embs),
        ]:
            for start in range(0, len(split_items), CFG.batch_size):
                batch = split_items[start : start + CFG.batch_size]
                paths = [Path(item.path) for item in batch]
                mel_batch = build_batch(paths, device)
                with torch.no_grad():
                    frames = extract_frame_output(model, mel_batch, device)
                frames_np = frames.detach().cpu().numpy().astype(np.float32)
                for idx, item in enumerate(batch):
                    frame_np = min_clamp_np(frames_np[idx])
                    split_embs.append(mean_pool_np(frame_np))
                    sample_frame_shape = list(frame_np.shape)
                    if split_name == "test":
                        y_test.append(int(item.label))
                        d_test.append(item.domain)
                        test_paths.append(str(item.path))

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)
    d_test_arr = np.asarray(d_test, dtype="U16")
    test_paths_arr = np.asarray(test_paths, dtype=object)
    emb_path = emb_root / "fan_anchor_geometry_lora_minclamp_mean_embeddings.npz"
    np.savez_compressed(
        emb_path,
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test_arr,
        d_test=d_test_arr,
        test_paths=test_paths_arr,
        machine=args.machine,
        checkpoint_path=str(Path(args.adapted_ckpt_path)),
        preprocess="min_clamp",
        pooling="mean",
        embedding_source="stage19h_anchor_geometry_lora_eat",
    )
    write_csv(
        out_root / "embedding_summary.csv",
        [
            {
                "machine": args.machine,
                "train_count": len(train_items),
                "test_count": len(test_items),
                "embedding_dim": int(train_emb.shape[1]),
                "frame_shape": str(sample_frame_shape),
                "checkpoint_path": str(Path(args.adapted_ckpt_path)),
                "embedding_path": str(emb_path),
                "preprocess": "min_clamp",
                "pooling": "mean",
                "embedding_source": "stage19h_anchor_geometry_lora_eat",
            }
        ],
    )


def run_score_adapted_fan(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_path = Path(args.embedding_npz_path)
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing adapted embedding file: {emb_path}")

    arr = np.load(emb_path, allow_pickle=True)
    train_emb = arr["train_embeddings"].astype(np.float64)
    test_emb = arr["test_embeddings"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    test_paths = arr["test_paths"].astype(str)

    train_emb = l2_normalize_np(train_emb, axis=1)
    test_emb = l2_normalize_np(test_emb, axis=1)

    dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref = chord_distance_matrix(train_emb, train_emb)
    rho_k1 = local_density_knn(ref_ref, K=1)

    score_baseline = baseline_knn_scores(dist, score_k=CFG.score_k)
    score_ldn = baseline_knn_scores(dist - rho_k1[None, :], score_k=CFG.score_k)

    scores_root = out_root / "scores"
    (scores_root / "adapted_baseline_knn_k5").mkdir(parents=True, exist_ok=True)
    (scores_root / "adapted_ldn_diff_K1_scorek5").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    tail_rows = []
    geom_rows = []
    candidate_score_map = {
        "adapted_baseline_knn_k5": score_baseline,
        "adapted_ldn_diff_K1_scorek5": score_ldn,
    }
    for candidate, scores in candidate_score_map.items():
        summary, tail, geom = evaluate(scores, y_test, d_test)
        summary_rows.append({"candidate": candidate, **summary})
        tail_rows.append({"candidate": candidate, **tail})
        geom_rows.append({"candidate": candidate, **geom})
        np.savez(
            scores_root / candidate / "fan_scores.npz",
            test_scores=scores.astype(np.float64),
            y_test=y_test,
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            candidate=candidate,
            machine=args.machine,
        )

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)

    comparison_rows = build_candidate_rows(summary_rows)
    write_csv(out_root / "comparison_vs_stage18_fan.csv", comparison_rows)

    adapted_baseline = next(row for row in summary_rows if row["candidate"] == "adapted_baseline_knn_k5")
    adapted_ldn = next(row for row in summary_rows if row["candidate"] == "adapted_ldn_diff_K1_scorek5")
    verdict = {
        "adapted_ldn_hmean": float(adapted_ldn["hmean"]),
        "adapted_ldn_pAUC": float(adapted_ldn["pAUC"]),
        "adapted_ldn_TPR99": float(adapted_ldn["TPR@99n"]),
        "adapted_baseline_hmean": float(adapted_baseline["hmean"]),
        "stage18_ldn_hmean": STAGE18_LDN_HMEAN,
        "stage18_baseline_hmean": STAGE18_BASELINE_HMEAN,
        "stage19f_ldn_hmean": STAGE19F_LDN_HMEAN,
        "delta_vs_stage18_ldn": float(adapted_ldn["hmean"]) - STAGE18_LDN_HMEAN,
        "delta_vs_stage18_baseline": float(adapted_ldn["hmean"]) - STAGE18_BASELINE_HMEAN,
        "delta_vs_stage19f": float(adapted_ldn["hmean"]) - STAGE19F_LDN_HMEAN,
        "minimum_pass": float(adapted_ldn["hmean"]) > STAGE18_LDN_HMEAN,
        "strong_pass": float(adapted_ldn["hmean"]) >= STAGE18_BASELINE_HMEAN,
        "negative": float(adapted_ldn["hmean"]) <= STAGE18_LDN_HMEAN,
        "representation_guardrail_pass": float(adapted_baseline["hmean"]) >= (STAGE18_BASELINE_HMEAN - REPRESENTATION_GUARDRAIL_MAX_DROP),
        "notes": [
            "fan only",
            "no full7 conclusion",
            "no selector",
            "no checkpoint selection by test labels",
        ],
    }
    write_json(out_root / "stage19i_fan_anchor_geometry_lora_eval_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_audit_markdown(verdict, comparison_rows, summary_rows))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_checkpoint":
        run_inspect_checkpoint(args)
        return
    if args.mode == "extract_adapted_embeddings":
        run_extract_adapted_embeddings(args)
        return
    if args.mode == "score_adapted_fan":
        run_score_adapted_fan(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
