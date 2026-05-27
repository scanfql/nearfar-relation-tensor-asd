"""Score geometry: distribution analysis of normal scores across source/target domains."""
import numpy as np
from scipy.stats import ks_2samp


def compute_score_geometry(scores, y_true, domains):
    """Analyze score distributions for normal source vs normal target.

    Labels are used here ONLY to filter normal items — this is post-hoc analysis,
    not model selection.
    """
    y = np.array(y_true, dtype=np.int64)
    d = np.array(domains)
    s = np.array(scores)

    src_normal = s[(d == "source") & (y == 0)]
    tgt_normal = s[(d == "target") & (y == 0)]

    result = {
        "source_normal_mean": float(np.mean(src_normal)) if len(src_normal) > 0 else 0.0,
        "target_normal_mean": float(np.mean(tgt_normal)) if len(tgt_normal) > 0 else 0.0,
        "source_normal_std": float(np.std(src_normal)) if len(src_normal) > 0 else 0.0,
        "target_normal_std": float(np.std(tgt_normal)) if len(tgt_normal) > 0 else 0.0,
    }

    if len(src_normal) > 0 and len(tgt_normal) > 0:
        ks_stat, ks_pval = ks_2samp(src_normal, tgt_normal)
        result["source_target_ks"] = float(ks_stat)
        result["source_target_ks_pval"] = float(ks_pval)
        result["median_gap"] = float(np.median(tgt_normal) - np.median(src_normal))
        result["iqr_gap"] = float(
            (np.percentile(tgt_normal, 75) - np.percentile(tgt_normal, 25)) -
            (np.percentile(src_normal, 75) - np.percentile(src_normal, 25))
        )
    else:
        result["source_target_ks"] = 0.0
        result["source_target_ks_pval"] = 1.0
        result["median_gap"] = 0.0
        result["iqr_gap"] = 0.0

    # Overall score distribution stats
    all_scores = s
    result["score_mean"] = float(np.mean(all_scores))
    result["score_std"] = float(np.std(all_scores))
    result["score_skewness"] = float(
        np.mean((all_scores - result["score_mean"]) ** 3) / max(result["score_std"] ** 3, 1e-12)
    )
    result["score_kurtosis"] = float(
        np.mean((all_scores - result["score_mean"]) ** 4) / max(result["score_std"] ** 4, 1e-12)
    )
    return result
