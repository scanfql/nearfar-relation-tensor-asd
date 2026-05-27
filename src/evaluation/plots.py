"""Score histogram plotting for post-hoc visualization."""
import os
import numpy as np


def plot_score_histogram(scores, y_true, domains, machine, experiment, output_dir):
    """Plot score histogram: source normal, target normal, anomaly.

    Anomaly labels are used only for post-hoc visualization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.array(y_true, dtype=np.int64)
    d = np.array(domains)
    s = np.array(scores)

    src_norm = s[(d == "source") & (y == 0)]
    tgt_norm = s[(d == "target") & (y == 0)]
    anomaly = s[y == 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = 40
    alpha = 0.6

    if len(src_norm) > 0:
        ax.hist(src_norm, bins=bins, alpha=alpha, label="source normal", color="blue")
    if len(tgt_norm) > 0:
        ax.hist(tgt_norm, bins=bins, alpha=alpha, label="target normal", color="green")
    if len(anomaly) > 0:
        ax.hist(anomaly, bins=bins, alpha=alpha, label="anomaly", color="red")

    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Count")
    ax.set_title(f"{machine} | {experiment}")
    ax.legend()
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{machine}_{experiment}_score_hist.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
