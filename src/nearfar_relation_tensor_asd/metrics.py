from __future__ import annotations

from collections import defaultdict

import numpy as np


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([neg, pos])
    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=np.float64)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[order[j]] == vals[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((np.sum(ranks[len(neg) :]) - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def harmonic_mean(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=np.float64)
    if len(finite) == 0:
        return float("nan")
    return float(len(finite) / np.sum(1.0 / finite))


def group_auc_hmean(rows: list[dict], scores: np.ndarray) -> tuple[dict[str, float], float]:
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[f"{row['machine_type']}/{row['domain']}"].append(i)
    aucs = {}
    for group, idx in grouped.items():
        y = np.asarray([1 if rows[i]["label"] == "anomaly" else 0 for i in idx], dtype=int)
        aucs[group] = roc_auc(y, scores[idx])
    return aucs, harmonic_mean(list(aucs.values()))
