"""Paper figure style: vector PDF, grayscale-safe, fonts >= 6 pt (Springer LNEE single column ~ 12 cm)."""
from __future__ import annotations

from cycler import cycler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 4.7    # inches (~12 cm text width)
SEQ_CMAP = "cividis"      # perceptually uniform, grayscale-monotone
DIV_CMAP = "PuOr"         # diverging; we always add contour/labels so it survives grayscale
GRAYS = ["0.0", "0.35", "0.6", "0.8"]
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "x"]


def setup() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "axes.prop_cycle": cycler(color=["0.0", "0.3", "0.5", "0.7", "0.15", "0.45"]) +
        cycler(linestyle=["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]),
        "lines.linewidth": 1.0, "axes.linewidth": 0.6, "figure.dpi": 150,
    })


def save(fig, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
