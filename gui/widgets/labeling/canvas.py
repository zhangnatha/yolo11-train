# gui/widgets/labeling/canvas.py
"""
Canvas widget for image display and shape drawing/editing.
Reference: X-AnyLabeling views/labeling/widgets/canvas.py
"""

import math
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QKeyEvent
from PyQt6.QtWidgets import QWidget, QApplication, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class Canvas(QGraphicsView):
    """Graphics view canvas for displaying images and drawing shapes."""

    # Signals
    new_shape_created = pyqtSignal(object)  # Shape
    shape_selected = pyqtSignal(object)  # Shape
    shape_moved = pyqtSignal(object, list)  # Shape, new_points
    shapes_deleted = pyqtSignal(list)  # List of shapes
    zoom_changed = pyqtSignal(float)  # zoom level

    # Drawing modes
    MODE_NONE = 0
    MODE_CREATE_RECT = 1
    MODE_CREATE_POLYGON = 2
    MODE_EDIT = 3

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene setup
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Image
        self.image_item = None
        self.image = QImage()

        # Drawing state
        self.mode = self.MODE_NONE
        self.current_shape = None
        self.current_points = []
        self.temp_line = None

        # Shapes
        self.shapes = []  # List of Shape objects
        self.selected_shape = None
        self.selected_point_index = -1

        # View state
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)

        # Style
        self.shape_color = QColor("#FF6B6B")
        self.shape_width = 2
        self.selected_color = QColor("#00BFFF")

        # Interactions
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdateMode)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Accept keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    def load_image(self, image_path):
        """Load an image from file."""
        self.image = QImage(image_path)
        if self.image.isNull():
            return False

        # Clear scene
        self.scene.clear()
        self.shapes.clear()
        self.current_points.clear()

        # Add image to scene
        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(self.image))
        self.scene.addItem(self.image_item)

        # Fit to view
        self.fit_in_view()
        return True

    def set_image_from_array(self, array):
        """Set image from numpy array (BGR format)."""
        try:
            import cv2
            import numpy as np

            # Convert BGR to RGB
            if len(array.shape) == 3 and array.shape[2] == 3:
                rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            else:
                rgb = array

            # Convert to QImage
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)

            height, width, channels = rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            self.image = q_image
            return self.load_image_from_qimage(q_image)
        except Exception as e:
            print(f"Error loading image from array: {e}")
            return False

    def load_image_from_qimage(self, q_image):
        """Load QImage directly."""
        self.image = q_image
        self.scene.clear()
        self.shapes.clear()
        self.current_points.clear()

        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(self.image))
        self.scene.addItem(self.image_item)
        self.fit_in_view()
        return True

    def fit_in_view(self):
        """Fit image in view."""
        if self.image_item:
            self.scene.setSceneRect(self.image_item.boundingRect())
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom = self.transform().m11()

    def set_draw_mode_rect(self):
        """Set drawing mode to create rectangles."""
        self.mode = self.MODE_CREATE_RECT
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.current_points.clear()

    def set_draw_mode_polygon(self):
        """Set drawing mode to create polygons."""
        self.mode = self.MODE_CREATE_POLYGON
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.current_points.clear()

    def set_edit_mode(self):
        """Set edit mode."""
        self.mode = self.MODE_EDIT
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_view_mode(self):
        """Set view mode (pan and zoom only)."""
        self.mode = self.MODE_NONE
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def clear_shapes(self):
        """Clear all shapes from canvas."""
        for shape_item in getattr(self, 'shape_items', []):
            if shape_item:
                self.scene.removeItem(shape_item)
        self.shapes.clear()
        self.shape_items = []
        self.selected_shape = None
        self.current_points.clear()
        self.current_shape = None

    def add_shape(self, shape):
        """Add a shape to the canvas."""
        from .shape import Shape

        if not isinstance(shape, Shape):
            return

        self.shapes.append(shape)
        self._draw_shape(shape)
        self.new_shape_created.emit(shape)

    def _draw_shape(self, shape):
        """Draw a shape on the canvas."""
        if not shape.points:
            return

        pen = QPen(QColor(shape.get_color()))
        pen.setWidthF(self.shape_width)

        if shape.selected:
            pen.setColor(self.selected_color)
            pen.setWidthF(self.shape_width + 1)

        if shape.shape_type == "rectangle" and len(shape.points) >= 2:
            rect = QRectF(shape.points[0], shape.points[-1])
            item = self.scene.addRect(rect, pen)
        elif shape.shape_type == "polygon" and len(shape.points) >= 3:
            from PyQt6.QtGui import QPolygonF
            polygon = QPolygonF(shape.points)
            item = self.scene.addPolygon(polygon, pen)
        elif shape.shape_type == "circle" and len(shape.points) >= 2:
            p1, p2 = shape.points[0], shape.points[-1]
            radius = math.sqrt((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)
            item = self.scene.addEllipse(
                p1.x() - radius, p1.y() - radius,
                radius * 2, radius * 2, pen
            )
        else:
            # Line or single point
            if len(shape.points) >= 2:
                item = self.scene.addLine(
                    int(shape.points[0].x()), int(shape.points[0].y()),
                    int(shape.points[-1].x()), int(shape.points[-1].y()), pen
                )
            else:
                item = self.scene.addEllipse(
                    shape.points[0].x() - 3, shape.points[0].y() - 3, 6, 6, pen
                )

        if not hasattr(self, 'shape_items'):
            self.shape_items = []
        self.shape_items.append(item)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())

            if self.mode == self.MODE_CREATE_RECT:
                self.current_points = [pos]
                self.temp_line = None
            elif self.mode == self.MODE_CREATE_POLYGON:
                self.current_points.append(pos)
                self.update()
            elif self.mode == self.MODE_EDIT:
                # Check if clicked on a shape point
                self._handle_edit_click(pos)
        elif event.button() == Qt.MouseButton.RightButton:
            if self.mode == self.MODE_CREATE_POLYGON and self.current_points:
                self._finish_polygon()
            elif self.mode == self.MODE_EDIT:
                self.set_view_mode()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())

            if self.mode == self.MODE_CREATE_RECT and self.current_points:
                self.update()
            elif self.mode == self.MODE_CREATE_POLYGON:
                self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())

            if self.mode == self.MODE_CREATE_RECT and self.current_points:
                # Finish rectangle
                from .shape import Shape
                shape = Shape(label="", shape_type="rectangle")
                shape.add_point(self.current_points[0])
                shape.add_point(pos)
                self.add_shape(shape)
                self.current_points.clear()

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom *= 1.1
            else:
                self.zoom /= 1.1
            self.zoom = max(0.1, min(10.0, self.zoom))
            self.scale(1.0 / self.transform().m11() * self.zoom, 1.0 / self.transform().m22() * self.zoom)
            self.zoom_changed.emit(self.zoom)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() == Qt.Key.Key_Escape:
            if self.mode == self.MODE_CREATE_POLYGON:
                self.current_points.clear()
                self.update()
            self.set_view_mode()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.mode == self.MODE_CREATE_POLYGON and len(self.current_points) >= 3:
                self._finish_polygon()
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.selected_shape:
                self._delete_selected_shape()

        super().keyPressEvent(event)

    def _finish_polygon(self):
        """Finish drawing polygon."""
        if len(self.current_points) < 3:
            return

        from .shape import Shape
        shape = Shape(label="", shape_type="polygon")
        for p in self.current_points:
            shape.add_point(p)
        self.add_shape(shape)
        self.current_points.clear()
        self.update()

    def _handle_edit_click(self, pos):
        """Handle click in edit mode."""
        # Find if clicked near any shape point
        threshold = 10 / self.zoom

        for shape in self.shapes:
            for i, point in enumerate(shape.points):
                if (pos - point).manhattanLength() < threshold:
                    self.selected_shape = shape
                    self.selected_point_index = i
                    self.shape_selected.emit(shape)
                    return

        # Check if clicked inside a shape
        for shape in self.shapes:
            if shape.bounding_rect().contains(pos):
                self.selected_shape = shape
                self.selected_point_index = -1
                self.shape_selected.emit(shape)
                return

        self.selected_shape = None
        self.selected_point_index = -1

    def _delete_selected_shape(self):
        """Delete the selected shape."""
        if self.selected_shape:
            self.shapes.remove(self.selected_shape)
            self.shapes_deleted.emit([self.selected_shape])
            self.selected_shape = None

    def get_shapes(self):
        """Get all shapes."""
        return self.shapes

    def to_dict(self):
        """Export shapes to dictionary."""
        return [s.to_dict() for s in self.shapes]

    def from_dict(self, data):
        """Import shapes from dictionary."""
        from .shape import Shape
        self.clear_shapes()
        for item in data:
            shape = Shape.from_dict(item)
            self.add_shape(shape)
