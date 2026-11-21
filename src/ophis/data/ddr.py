from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

LESIONS: tuple[str, ...] = ("MA", "HE", "EX", "SE")

LESION_NAMES: dict[str, str] = {
    "MA": "Microaneurysms",
    "HE": "Haemorrhages",
    "EX": "Hard exudates",
    "SE": "Soft exudates",
}

LESION_COLORS: dict[str, tuple[int, int, int]] = {
    "MA": (255, 40, 40),
    "HE": (255, 150, 0),
    "EX": (255, 240, 0),
    "SE": (0, 200, 255),
}

SPLITS: tuple[str, ...] = ("train", "val", "test")
_ANNOTATION_DIRS = ("test", "tet")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DDRSample:
    image_path: Path
    mask_paths: dict[str, Path]

    @property
    def name(self) -> str:
        return self.image_path.stem

    def load_image(self) -> np.ndarray:
        with Image.open(self.image_path) as im:
            return np.asarray(im.convert("RGB"))

    def load_masks(self) -> dict[str, np.ndarray]:
        masks = {}
        for lesion, path in self.mask_paths.items():
            with Image.open(path) as im:
                masks[lesion] = np.asarray(im.convert("L")) > 0
        return masks


def _seg_root(root: Path) -> Path:
    root = Path(root)
    return root / "lesion_segmentation" if (root / "lesion_segmentation").is_dir() else root


def _annotation_dir(seg_root: Path, split: str) -> Path:
    base = seg_root / "annotations"
    if split != "test":
        return base / split
    for name in _ANNOTATION_DIRS:
        if (base / name).is_dir():
            return base / name
    return base / "test"


class DDRSegmentation:
    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = _seg_root(Path(root))
        self.split = split

        image_dir = self.root / "images" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"DDR images not found: {image_dir}")
        annot_dir = _annotation_dir(self.root, split)

        self.samples: list[DDRSample] = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            masks = {
                lesion: annot_dir / lesion / f"{image_path.stem}.tif"
                for lesion in LESIONS
            }
            masks = {k: p for k, p in masks.items() if p.is_file()}
            self.samples.append(DDRSample(image_path, masks))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DDRSample:
        return self.samples[index]
