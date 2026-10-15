"""Render coarse spatial hints without text labels or exact point markers."""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from vpopsd.data import validate_box


@dataclass
class PrivilegeConfig:
    grid_cells: int = 12
    margin_cells: int = 1
    sigma_scale: float = 0.65
    background_keep: float = 0.35
    red_box: bool = True
    gaussian_mask: bool = True

    def __post_init__(self):
        if self.grid_cells < 2 or self.margin_cells < 0 or self.sigma_scale <= 0:
            raise ValueError("Invalid privilege grid/margin/sigma")
        if not 0 <= self.background_keep <= 1:
            raise ValueError("background_keep must be in [0,1]")


def coarse_box(box, cfg):
    x1, y1, x2, y2 = validate_box(box)
    n, m = cfg.grid_cells, cfg.margin_cells
    return (
        max(0, math.floor(x1 * n) - m) / n,
        max(0, math.floor(y1 * n) - m) / n,
        min(n, math.ceil(x2 * n) + m) / n,
        min(n, math.ceil(y2 * n) + m) / n,
    )


def privileged_view(image, box, cfg=None):
    cfg = cfg or PrivilegeConfig()
    image = image.convert("RGB")
    w, h = image.size
    x1, y1, x2, y2 = coarse_box(box, cfg)
    pixels = np.asarray(image, dtype=np.float32).copy()
    if cfg.gaussian_mask:
        yy, xx = np.mgrid[0:h, 0:w]
        # Center and spread depend ONLY on the coarse region, not the true click.
        cx, cy = (x1 + x2) * w / 2, (y1 + y2) * h / 2
        sx, sy = max(1, (x2 - x1) * w * cfg.sigma_scale), max(1, (y2 - y1) * h * cfg.sigma_scale)
        mask = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        alpha = cfg.background_keep + (1 - cfg.background_keep) * mask
        pixels *= alpha[..., None]
    output = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if cfg.red_box:
        ImageDraw.Draw(output).rectangle(
            (round(x1 * (w - 1)), round(y1 * (h - 1)), round(x2 * (w - 1)), round(y2 * (h - 1))),
            outline=(255, 0, 0),
            width=max(2, round(min(w, h) / 250)),
        )
    return output


def prompt(instruction, teacher=False, use_hint=True):
    text = (
        "Locate the UI element described below. Return only (x, y), where x and y are "
        "integers from 0 to 1000 representing normalized screen coordinates.\n"
        f"Instruction: {instruction}"
    )
    if teacher and use_hint:
        text += (
            "\nThe image contains a coarse visual cue for the target region. "
            "Use the UI content to locate the target precisely; the cue is approximate."
        )
    return text
