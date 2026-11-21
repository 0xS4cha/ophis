import numpy as np

from ophis.viz.overlay import downscale, overlay_masks
from ophis.viz.retina3d import center_square, retina_cap


def test_overlay_only_touches_enabled_masks():
    image = np.zeros((4, 4, 3), np.uint8)
    ma = np.zeros((4, 4), bool)
    ma[0, 0] = True
    he = np.zeros((4, 4), bool)
    he[1, 1] = True
    out = overlay_masks(image, {"MA": ma, "HE": he}, enabled=["MA"], alpha=1.0)
    assert tuple(out[0, 0]) == (255, 40, 40)
    assert tuple(out[1, 1]) == (0, 0, 0)


def test_downscale_keeps_tiny_lesions():
    mask = np.zeros((1000, 1000), bool)
    mask[501, 501] = True
    small = downscale(mask, 100)
    assert small.shape == (100, 100) and small.any()


def test_center_square():
    assert center_square(np.zeros((4, 10, 3))).shape == (4, 4, 3)


def test_retina_cap_has_texture_coords():
    mesh = retina_cap(resolution=30)
    assert mesh.n_points > 0
    tc = mesh.active_texture_coordinates
    assert tc is not None and tc.min() >= 0 and tc.max() <= 1
