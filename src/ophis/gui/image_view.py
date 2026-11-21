from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def numpy_to_qimage(image: np.ndarray) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    return QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QGraphicsView):
    ZOOM_STEP = 1.25

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.black)

    def set_image(self, image: np.ndarray, fit: bool = False) -> None:
        self._item.setPixmap(QPixmap.fromImage(numpy_to_qimage(image)))
        self._scene.setSceneRect(self._item.boundingRect())
        if fit:
            self.fit()

    def fit(self) -> None:
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1 / self.ZOOM_STEP
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()
