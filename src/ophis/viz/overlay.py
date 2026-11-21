from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from PIL import Image

from ophis.data.ddr import LESION_COLORS


def overlay_masks(
    image: np.ndarray,
    masks: Mapping[str, np.ndarray],
    enabled: Iterable[str] | None = None,
    alpha: float = 0.6,
    colors: Mapping[str, tuple[int, int, int]] = LESION_COLORS,
) -> np.ndarray:
    enabled = set(masks) if enabled is None else set(enabled)
    out = image.astype(np.float32, copy=True)
    for lesion, mask in masks.items():
        if lesion not in enabled or not mask.any():
            continue
        color = np.asarray(colors[lesion], dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def downscale(array: np.ndarray, max_side: int, nearest: bool = False) -> np.ndarray:
    h, w = array.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return array
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if array.dtype == bool:
        im = Image.fromarray(array.astype(np.uint8) * 255)
        return np.asarray(im.resize(size, Image.Resampling.BOX)) > 0
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.asarray(Image.fromarray(array).resize(size, resample))


def lesion_pixel_counts(masks: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {lesion: int(mask.sum()) for lesion, mask in masks.items()}
