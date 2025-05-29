# evaluation/vis/make_figure.py
# ------------------------------------------------------------------
#  Build qualitative grids identical to Figs.  S5-1 / S5-2 / S5-3.
#  • Assumes each method has already generated SR outputs in
#    <out_dir>/<method>/<img_id>.png
#  • Saves a high-resolution PDF + PNG suitable for camera-ready.
# ------------------------------------------------------------------
#  Requirements: matplotlib, pillow, tqdm
# ------------------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def build_grid(
    ids: list[str],
    lr_dir: Path,
    hr_dir: Path,
    sr_root: Path,
    methods: list[str],
    scale: int,
    save_path: Path,
) -> None:
    n_rows = len(ids)
    n_cols = 2 + len(methods)  # LR + |methods| + HR
    col_titles = ["LR"] + methods + ["HR"]

    fig_h = n_rows * 2.5
    fig_w = n_cols * 2.5
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="tight",
        dpi=300,
    )

    for r, img_id in enumerate(tqdm(ids, desc="building grid")):
        # leftmost LR
        lr_img = load_image(lr_dir / f"{img_id}.png")
        axes[r][0].imshow(lr_img, cmap="gray")
        axes[r][0].set_title(col_titles[0], fontsize=8)
        axes[r][0].axis("off")

        # SR methods
        for c, m in enumerate(methods, start=1):
            sr_path = sr_root / m / f"{img_id}.png"
            sr_img = load_image(sr_path)
            axes[r][c].imshow(sr_img)
            axes[r][c].set_title(col_titles[c], fontsize=8)
            axes[r][c].axis("off")

        # rightmost HR
        hr_img = load_image(hr_dir / f"{img_id}.png")
        axes[r][-1].imshow(hr_img)
        axes[r][-1].set_title(col_titles[-1], fontsize=8)
        axes[r][-1].axis("off")

    fig.suptitle(f"Qualitative comparison (×{scale} SR)", fontsize=12, weight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=Path, required=True, help="txt file of image IDs")
    p.add_argument("--lr_dir", type=Path, required=True, help="directory of LR crops")
    p.add_argument("--hr_dir", type=Path, required=True, help="directory of HR GT")
    p.add_argument(
        "--sr_root",
        type=Path,
        required=True,
        help="root directory containing per-method SR results",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="list of methods in display order, e.g. GFPGAN GCFSR ... EFANet",
    )
    p.add_argument("--scale", type=int, required=True, choices=[4, 8, 16])
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output filename without extension, e.g. figs/fig_s5_1",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ids_list = [l.strip() for l in args.ids.read_text().splitlines() if l.strip()]
    build_grid(
        ids=ids_list,
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        sr_root=args.sr_root,
        methods=args.methods,
        scale=args.scale,
        save_path=args.output,
    )
