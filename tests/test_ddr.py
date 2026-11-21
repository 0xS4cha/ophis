import numpy as np
import pytest
from PIL import Image

from ophis.data.ddr import LESIONS, DDRSegmentation


def _make_fake_ddr(root, split="test", annot_split="tet", names=("a", "b")):
    seg = root / "lesion_segmentation"
    (seg / "images" / split).mkdir(parents=True)
    for name in names:
        Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(seg / "images" / split / f"{name}.jpg")
        for lesion in LESIONS:
            d = seg / "annotations" / annot_split / lesion
            d.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((8, 12), np.uint8)
            mask[2, 3] = 255
            Image.fromarray(mask).save(d / f"{name}.tif")
    return root


def test_test_split_maps_to_tet_folder(tmp_path):
    ds = DDRSegmentation(_make_fake_ddr(tmp_path), "test")
    assert [s.name for s in ds.samples] == ["a", "b"]
    masks = ds[0].load_masks()
    assert set(masks) == set(LESIONS)
    assert masks["MA"].dtype == bool and masks["MA"].sum() == 1
    assert ds[0].load_image().shape == (8, 12, 3)


def test_accepts_lesion_segmentation_folder_as_root(tmp_path):
    _make_fake_ddr(tmp_path, split="train", annot_split="train")
    assert len(DDRSegmentation(tmp_path / "lesion_segmentation", "train")) == 2


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DDRSegmentation(tmp_path, "val")
    with pytest.raises(ValueError):
        DDRSegmentation(tmp_path, "nope")
