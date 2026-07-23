#!/usr/bin/env python3
"""Prepare a labelled, white-background copy of HDP's multimodality figure.

The source is the raster included in the local HDP paper TeX package.  This
script does not alter the trajectories; it only adds panel labels and an
evidence-boundary caption so the image is self-contained in Confluence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference/papers/hyper_diffusion_planner_paper/src/assets/multimodality.png"
OUT = ROOT / "docs/t4_conference_assets/hdp_multimodality/hdp_native_multimodality_explainer.png"


def main() -> None:
    source = Image.open(SOURCE).convert("RGBA")
    white = Image.new("RGBA", source.size, "white")
    white.alpha_composite(source)
    image = white.convert("RGB")

    # The official raster contains three equal-purpose panels: scene context,
    # a low-data collapsed sample bundle, and a large-data diverse bundle.
    width, _ = image.size
    cuts = (0, int(width * 0.34), int(width * 0.675), width)
    panels = [image.crop((cuts[i], 0, cuts[i + 1], image.height)) for i in range(3)]
    labels = [
        "A · Real driving scene",
        "B · 100K-scale training: samples collapse",
        "C · 20M-scale training: samples spread into modes",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.80, bottom=0.16, wspace=0.035)
    for axis, panel, label in zip(axes, panels, labels):
        axis.imshow(panel)
        axis.set_title(label, loc="left", fontsize=12, fontweight="bold", color="#19324a")
        axis.axis("off")
    fig.suptitle(
        "HDP paper: learned semantic multimodality emerges from data scaling",
        x=0.03, y=0.965, ha="left", fontsize=20, fontweight="bold", color="#102a43",
    )
    fig.text(
        0.03, 0.875,
        "Same-scene diffusion samples (blue) versus logged trajectory (red). The paper attributes the change to scaling real-driving data, without anchor or goal conditioning.",
        fontsize=10.5, color="#52677d",
    )
    fig.text(
        0.03, 0.055,
        "Source: HDP paper multimodality figure. Trajectory Divergence = average pairwise Euclidean distance among samples for the same scene. "
        "This is paper evidence, not a result measured on the current T4 AWR checkpoint.",
        fontsize=9.5, color="#52677d",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
