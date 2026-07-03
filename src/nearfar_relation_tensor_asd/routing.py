from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import group_auc_hmean


@dataclass
class RouteChoice:
    group: str
    score_id: str
    alpha: float
    hmean: float


def blend_scores(base: np.ndarray, tensor: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * np.asarray(base, dtype=np.float64) + float(alpha) * np.asarray(tensor, dtype=np.float64)


def restricted_group_routing(
    rows: list[dict],
    base_score: np.ndarray,
    score_pool: dict[str, np.ndarray],
    alpha_grid: tuple[float, ...] = (0.0, 0.075, 0.15, 0.225),
) -> tuple[np.ndarray, list[RouteChoice]]:
    """Development-protocol group routing over a fixed score pool.

    This function uses development labels to select score/alpha per
    machine-domain group and should be reported as development-protocol routing,
    not as a blind deployment rule.
    """
    out = np.asarray(base_score, dtype=np.float64).copy()
    choices: list[RouteChoice] = []
    groups = sorted({f"{r['machine_type']}/{r['domain']}" for r in rows})
    for group in groups:
        idx = np.asarray([i for i, r in enumerate(rows) if f"{r['machine_type']}/{r['domain']}" == group], dtype=int)
        best = (float("-inf"), None, 0.0, None)
        sub_rows = [rows[i] for i in idx]
        for sid, tensor_score in score_pool.items():
            for alpha in alpha_grid:
                candidate = out.copy()
                candidate[idx] = blend_scores(base_score[idx], np.asarray(tensor_score)[idx], alpha)
                _, hm = group_auc_hmean(sub_rows, candidate[idx])
                if hm > best[0]:
                    best = (hm, sid, float(alpha), candidate[idx])
        out[idx] = best[3]
        choices.append(RouteChoice(group=group, score_id=str(best[1]), alpha=best[2], hmean=float(best[0])))
    return out, choices
