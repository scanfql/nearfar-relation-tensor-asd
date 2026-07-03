#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from nearfar_relation_tensor_asd.manifest import write_manifest
from nearfar_relation_tensor_asd.mpca import fit_mpca, hybridtail_score
from nearfar_relation_tensor_asd.normalization import fit_source_normal_stats
from nearfar_relation_tensor_asd.patchmax import channel_share, patchmax_score


ENERGY_CHANNELS = [0, 1]
RELATION_CHANNELS = [2, 3, 4, 5, 6, 7, 8, 11]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit source-normal MPCA and score HybridTail/PatchMax residual evidence.")
    p.add_argument("--tensor-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-fit-split", default="source_fit")
    p.add_argument("--ranks", default="32,32,8")
    p.add_argument("--patch", default="8,8")
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_tensor(row: dict) -> np.ndarray:
    return np.load(row["tensor_path"])["x"].astype(np.float32)


def main() -> None:
    args = parse_args()
    ranks = tuple(int(x) for x in args.ranks.split(","))
    patch = tuple(int(x) for x in args.patch.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.tensor_manifest)

    by_machine = defaultdict(list)
    for row in rows:
        by_machine[row["machine_type"]].append(row)

    score_rows = []
    for machine, mrows in sorted(by_machine.items()):
        fit_rows = [r for r in mrows if r.get("split") == args.source_fit_split]
        if not fit_rows:
            raise RuntimeError(f"No source-fit rows for machine={machine}")
        fit_raw = np.stack([load_tensor(r) for r in fit_rows], axis=0)
        stats = fit_source_normal_stats(fit_raw)
        fit_norm = np.stack([stats.transform(x) for x in fit_raw], axis=0)
        model = fit_mpca(fit_norm, ranks=ranks)
        source_res = model.residual(fit_norm)

        for row in mrows:
            x = stats.transform(load_tensor(row))
            residual = model.residual(x)
            pm = patchmax_score(residual, patch=patch)
            score_rows.append(
                {
                    "basename": row["basename"],
                    "machine_type": row["machine_type"],
                    "domain": row["domain"],
                    "split": row["split"],
                    "label": row.get("label", ""),
                    "score_hybridtail": hybridtail_score(residual, source_res),
                    "score_patchmax": pm.score,
                    "patch_top_time": pm.top_time,
                    "patch_top_freq": pm.top_freq,
                    "patch_relation_share": channel_share(pm.channel_energy, RELATION_CHANNELS),
                    "patch_energy_share": channel_share(pm.channel_energy, ENERGY_CHANNELS),
                }
            )
    write_manifest(out_dir / "mpca_patchmax_score_manifest.csv", score_rows)


if __name__ == "__main__":
    main()
