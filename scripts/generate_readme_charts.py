"""Generate metric charts for README.md."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def overall_metrics() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["mAP@50", "Precision", "Recall"]
    vals = [36.3, 52.1, 39.7]
    colors = ["#0B3A5B", "#1F7A8C", "#BF4E30"]
    bars = ax.bar(metrics, vals, color=colors, width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("RF-DETR Medium — Overall Test Metrics (Version 1)")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(50, color="#999", ls="--", lw=0.8, label="50% reference")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "overall-metrics.png", dpi=160)
    plt.close()


def per_class() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    classes = ["solder void", "scratch", "misalignment"]
    precision = [56.9, 42.7, 0.0]
    recall = [42.9, 39.6, 0.0]
    f1 = [48.9, 41.1, 0.0]
    x = np.arange(len(classes))
    w = 0.25
    ax.bar(x - w, precision, w, label="Precision", color="#0B3A5B")
    ax.bar(x, recall, w, label="Recall", color="#1F7A8C")
    ax.bar(x + w, f1, w, label="F1", color="#E09F3E")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Per-Class Performance on Test Split")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "per-class-metrics.png", dpi=160)
    plt.close()


def confusion_matrix() -> None:
    labels = ["misalignment", "scratch", "solder void", "background"]
    m = np.array(
        [
            [0, 0, 0, 0],
            [0, 143, 0, 122],
            [0, 2, 230, 279],
            [8, 476, 269, 0],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (ground truth)")
    ax.set_title("Confusion Matrix — Test Split (conf ≥ 0.20)")
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                int(m[i, j]),
                ha="center",
                va="center",
                color="white" if m[i, j] > m.max() * 0.55 else "#111",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion-matrix.png", dpi=160)
    plt.close()


def class_balance() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["scratch\n(2,853 boxes)", "solder void\n(4,739 boxes)", "misalignment\n(53 boxes)"]
    counts = [2853, 4739, 53]
    cols = ["#1F7A8C", "#0B3A5B", "#BF4E30"]
    ax.barh(names, counts, color=cols)
    ax.set_xlabel("Annotated instances in dataset")
    ax.set_title("Class Imbalance in Training Dataset (4,531 images)")
    for i, v in enumerate(counts):
        ax.text(v + 40, i, f"{v:,}", va="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "class-balance.png", dpi=160)
    plt.close()


def map_by_size() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = ["Small", "Medium", "Large"]
    map50 = [17.3, 46.2, 57.7]
    bars = ax.bar(sizes, map50, color=["#BF4E30", "#E09F3E", "#1F7A8C"], width=0.5)
    ax.set_ylabel("mAP@50 (%)")
    ax.set_ylim(0, 70)
    ax.set_title("Test mAP@50 by Object Size — Small defects are hardest")
    for b, v in zip(bars, map50):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 1.2,
            f"{v:.1f}%",
            ha="center",
            fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "map-by-object-size.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    overall_metrics()
    per_class()
    confusion_matrix()
    class_balance()
    map_by_size()
    print("Wrote:", sorted(p.name for p in OUT.glob("*.png")))
