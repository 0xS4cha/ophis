from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QMainWindow,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ophis.data.ddr import LESION_COLORS, LESION_NAMES, LESIONS, SPLITS, DDRSegmentation
from ophis.gui.image_view import ImageView
from ophis.viz.overlay import downscale, lesion_pixel_counts, overlay_masks
from ophis.viz.retina3d import fundus_texture, retina_cap

DISPLAY_MAX_SIDE = 1600
TEXTURE_MAX_SIDE = 1024


def _color_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, dataset_root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("ophis")
        self.resize(1400, 900)

        self.dataset_root = dataset_root
        self.dataset: DDRSegmentation | None = None
        self._image: np.ndarray | None = None
        self._masks: dict[str, np.ndarray] = {}

        self._build_views()
        self._build_side_panel()
        self._build_menus()

        if dataset_root is not None:
            self.load_dataset(dataset_root)

    def _build_views(self) -> None:
        self.tabs = QTabWidget()
        self.image_view = ImageView()
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111418")
        self._retina = retina_cap()
        self.tabs.addTab(self.image_view, "2D")
        self.tabs.addTab(self.plotter.interactor, "3D")
        self.setCentralWidget(self.tabs)

    def _build_side_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.split_combo = QComboBox()
        self.split_combo.addItems(SPLITS)
        self.split_combo.currentTextChanged.connect(self._reload_split)
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._on_sample_selected)

        form = QFormLayout()
        form.addRow("Split", self.split_combo)
        layout.addLayout(form)
        layout.addWidget(self.sample_list, stretch=1)

        lesion_box = QGroupBox("Lesions")
        lesion_layout = QFormLayout(lesion_box)
        lesion_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lesion_checks: dict[str, QCheckBox] = {}
        self.lesion_counts: dict[str, QLabel] = {}
        for lesion in LESIONS:
            check = QCheckBox(f"{lesion} · {LESION_NAMES[lesion]}")
            check.setIcon(_color_icon(LESION_COLORS[lesion]))
            check.setChecked(True)
            check.toggled.connect(self._refresh_views)
            count = QLabel("–")
            count.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.lesion_checks[lesion] = check
            self.lesion_counts[lesion] = count
            lesion_layout.addRow(check, count)

        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.valueChanged.connect(self._refresh_views)
        lesion_layout.addRow("Opacity", self.opacity)
        layout.addWidget(lesion_box)

        dock = QDockWidget("Dataset", self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_ds = QAction("Open DDR folder…", self, shortcut=QKeySequence.StandardKey.Open)
        open_ds.triggered.connect(self._choose_dataset)
        open_img = QAction("Open image…", self, shortcut="Ctrl+Shift+O")
        open_img.triggered.connect(self._choose_image)
        quit_action = QAction("Quit", self, shortcut=QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addActions([open_ds, open_img])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        reset = QAction("Reset view", self, shortcut="F")
        reset.triggered.connect(self._reset_view)
        view_menu.addAction(reset)

    def load_dataset(self, root: Path) -> None:
        self.dataset_root = Path(root)
        self._reload_split(self.split_combo.currentText())

    def _reload_split(self, split: str) -> None:
        self.sample_list.clear()
        if self.dataset_root is None:
            return
        try:
            self.dataset = DDRSegmentation(self.dataset_root, split)
        except FileNotFoundError as exc:
            self.dataset = None
            self.statusBar().showMessage(str(exc))
            return
        self.sample_list.addItems([s.name for s in self.dataset.samples])
        self.statusBar().showMessage(f"{len(self.dataset)} images ({split}) — {self.dataset.root}")
        if len(self.dataset):
            self.sample_list.setCurrentRow(0)

    def _on_sample_selected(self, row: int) -> None:
        if self.dataset is None or row < 0:
            return
        sample = self.dataset[row]
        image = sample.load_image()
        masks = sample.load_masks()
        counts = lesion_pixel_counts(masks)
        for lesion in LESIONS:
            n = counts.get(lesion)
            self.lesion_counts[lesion].setText("–" if n is None else f"{n:,} px".replace(",", " "))
        self._show(image, masks, title=sample.name)

    def _choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "DDR folder", str(self.dataset_root or Path.cwd()))
        if path:
            self.load_dataset(Path(path))

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Fundus image", str(Path.cwd()), "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        if not path:
            return
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        self.sample_list.blockSignals(True)
        self.sample_list.setCurrentRow(-1)
        self.sample_list.blockSignals(False)
        for label in self.lesion_counts.values():
            label.setText("–")
        self._show(image, {}, title=Path(path).name)

    def _show(self, image: np.ndarray, masks: dict[str, np.ndarray], title: str) -> None:
        first = self._image is None
        self._image = downscale(image, DISPLAY_MAX_SIDE)
        self._masks = {k: downscale(m, DISPLAY_MAX_SIDE) for k, m in masks.items()}
        self.setWindowTitle(f"ophis — {title}")
        self._refresh_views(fit=first)
        if first:
            self.plotter.view_xy()
            self.plotter.reset_camera()

    def _refresh_views(self, *_args, fit: bool = False) -> None:
        if self._image is None:
            return
        enabled = [k for k, c in self.lesion_checks.items() if c.isChecked()]
        composite = overlay_masks(self._image, self._masks, enabled, self.opacity.value() / 100)
        self.image_view.set_image(composite, fit=fit)
        texture = fundus_texture(downscale(composite, TEXTURE_MAX_SIDE))
        self.plotter.add_mesh(self._retina, texture=texture, lighting=False, name="retina", reset_camera=False)
        self.plotter.render()

    def _reset_view(self) -> None:
        self.image_view.fit()
        self.plotter.view_xy()
        self.plotter.reset_camera()

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)
