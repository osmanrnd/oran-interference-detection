#!/usr/bin/env python3
"""Build the sample-spectrogram contact sheet used in the README.

Reads up to COLS images per class from data/samples/<CLASS>/ and writes a labelled
grid (one row per class) to docs/figures/sample_dataset.png. Empty slots are filled
with placeholder tiles, so the figure is valid even before real samples are added.

Usage:
    python scripts/make_sample_grid.py
"""
import glob
import os

import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLES_DIR = os.path.join(ROOT, "data", "samples")
OUT = os.path.join(ROOT, "docs", "figures", "sample_dataset.png")

CLASSES = ["SOI", "CWI", "CI"]          # presentation order: clean -> CWI -> chirped
LABELS = {"SOI": "SOI", "CWI": "SOI+CWI", "CI": "SOI+CI"}
COLS = 6
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")


def list_images(class_dir):
    files = []
    for ext in EXTS:
        files += glob.glob(os.path.join(class_dir, ext))
    return sorted(files)[:COLS]


def main():
    fig, axes = plt.subplots(len(CLASSES), COLS, figsize=(COLS * 2, len(CLASSES) * 2.1))
    n_real = 0
    for r, cls in enumerate(CLASSES):
        images = list_images(os.path.join(SAMPLES_DIR, cls))
        n_real += len(images)
        for c in range(COLS):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(images):
                ax.imshow(Image.open(images[c]).convert("RGB"))
            else:
                ax.imshow([[0.94]], cmap="gray", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "add\nsample", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="#9a9a9a")
            if c == 0:
                ax.set_ylabel(LABELS[cls], fontsize=12, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=30)

    fig.suptitle("Sample spectrograms (data/samples/)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}  ({n_real} real sample image(s) found)")


if __name__ == "__main__":
    main()
