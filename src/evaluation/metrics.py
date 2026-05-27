"""AUC, pAUC, and hmean metrics. Labels used ONLY in final evaluation, never in selection."""
import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true, scores, max_fpr=None):
    """Return 0.5 when a split has only one class."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    if max_fpr is None:
        return float(roc_auc_score(y_true, scores))
    # sklearn returns standardized partial AUC, matching the previous project.
    return float(roc_auc_score(y_true, scores, max_fpr=max_fpr))


def compute_metrics(y_true, scores, domains):
    """Compute AUC, pAUC, and hmean. domains: list of 'source'/'target'."""
    y = np.array(y_true, dtype=np.int64)
    d = np.array(domains)
    s = np.array(scores)

    src_mask = d == "source"
    tgt_mask = d == "target"

    auc_source = _safe_auc(y[src_mask], s[src_mask]) if src_mask.any() else 0.5
    auc_target = _safe_auc(y[tgt_mask], s[tgt_mask]) if tgt_mask.any() else 0.5
    pauc_all = _safe_auc(y, s, max_fpr=0.1)
    dcase_auc = (auc_source + auc_target) / 2.0

    # hmean = harmonic mean of auc_source, auc_target, pauc
    vals = [auc_source, auc_target, pauc_all]
    if any(v == 0 for v in vals):
        hmean = 0.0
    else:
        hmean = float(3.0 / (1.0 / auc_source + 1.0 / auc_target + 1.0 / pauc_all))

    return {
        "auc_source": auc_source,
        "auc_target": auc_target,
        "pauc": pauc_all,
        "dcase_auc": dcase_auc,
        "hmean": hmean,
    }
