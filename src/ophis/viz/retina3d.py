from __future__ import annotations

import numpy as np
import pyvista as pv


def center_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return image[top : top + side, left : left + side]


def retina_cap(resolution: int = 200, fov_deg: float = 120.0, radius: float = 1.0) -> pv.PolyData:
    u, v = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    r = np.hypot(u, v)
    inside = r <= 1.0
    theta = np.clip(r, 0, 1) * np.deg2rad(fov_deg / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(r > 0, u / r, 0.0)
        dy = np.where(r > 0, v / r, 0.0)

    x = radius * np.sin(theta) * dx
    y = radius * np.sin(theta) * dy
    z = -radius * np.cos(theta)

    grid = pv.StructuredGrid(x, y, z)
    tcoords = np.c_[((u + 1) / 2).ravel(order="F"), ((v + 1) / 2).ravel(order="F")]
    grid.active_texture_coordinates = tcoords
    grid["inside"] = inside.ravel(order="F").astype(np.uint8)

    surface = grid.extract_surface(algorithm="dataset_surface")
    return surface.threshold(0.5, scalars="inside").extract_surface(algorithm="dataset_surface")


def fundus_texture(image: np.ndarray) -> pv.Texture:
    return pv.Texture(np.ascontiguousarray(center_square(image)))
