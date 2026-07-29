# gui/widgets/labeling/shape.py
"""
Shape definitions for labeling.
Reference: X-AnyLabeling views/labeling/shape.py
"""

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem
import numpy as np


# Default colors for different labels
DEFAULT_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
]


class Shape:
    """Represents a labeled shape with points and metadata."""

    def __init__(self, label="", shape_type="rectangle", points=None):
        self.label = label
        self.shape_type = shape_type  # rectangle, polygon, circle, line, point
        self.points = points if points else []  # List of QPointF
        self.color = None
        self.selected = False
        self.locked = False
        self.visible = True
        self.group_id = None
        self.score = None
        self.attributes = {}
        self.description = ""

    def get_color(self):
        """Get color for this shape based on label."""
        if self.color:
            return self.color

        # Generate consistent color from label
        if self.label:
            hash_val = sum(ord(c) for c in self.label)
            return DEFAULT_COLORS[hash_val % len(DEFAULT_COLORS)]
        return DEFAULT_COLORS[0]

    def set_color(self, color):
        """Set custom color for this shape."""
        self.color = color

    def add_point(self, point):
        """Add a point to the shape."""
        if isinstance(point, (list, tuple)):
            point = QPointF(point[0], point[1])
        self.points.append(point)

    def update_point(self, index, point):
        """Update a specific point."""
        if 0 <= index < len(self.points):
            if isinstance(point, (list, tuple)):
                point = QPointF(point[0], point[1])
            self.points[index] = point

    def get_mins(self):
        """Get minimum x and y coordinates."""
        if not self.points:
            return 0, 0
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        return min(xs), min(ys)

    def get_maxs(self):
        """Get maximum x and y coordinates."""
        if not self.points:
            return 0, 0
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        return max(xs), max(ys)

    def bounding_rect(self):
        """Get bounding rectangle."""
        x1, y1 = self.get_mins()
        x2, y2 = self.get_maxs()
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def to_dict(self):
        """Convert shape to dictionary."""
        return {
            "label": self.label,
            "shape_type": self.shape_type,
            "points": [(p.x(), p.y()) for p in self.points],
            "color": self.get_color(),
            "group_id": self.group_id,
            "score": self.score,
            "attributes": self.attributes,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data):
        """Create shape from dictionary."""
        shape = cls(
            label=data.get("label", ""),
            shape_type=data.get("shape_type", "rectangle"),
            points=[QPointF(p[0], p[1]) for p in data.get("points", [])],
        )
        shape.color = data.get("color")
        shape.group_id = data.get("group_id")
        shape.score = data.get("score")
        shape.attributes = data.get("attributes", {})
        shape.description = data.get("description", "")
        return shape

    def __repr__(self):
        return f"Shape({self.shape_type}, {self.label}, {len(self.points)} points)"


class ShapeCollection:
    """Collection of shapes with management methods."""

    def __init__(self):
        self.shapes = []
        self._selected_shape = None

    def add_shape(self, shape):
        """Add a shape to the collection."""
        self.shapes.append(shape)

    def remove_shape(self, shape):
        """Remove a shape from the collection."""
        if shape in self.shapes:
            self.shapes.remove(shape)
            if self._selected_shape == shape:
                self._selected_shape = None

    def clear(self):
        """Clear all shapes."""
        self.shapes.clear()
        self._selected_shape = None

    def select_shape(self, shape):
        """Select a shape."""
        if shape in self.shapes:
            if self._selected_shape:
                self._selected_shape.selected = False
            self._selected_shape = shape
            shape.selected = True

    def get_selected_shape(self):
        """Get currently selected shape."""
        return self._selected_shape

    def get_shapes_by_label(self, label):
        """Get all shapes with a specific label."""
        return [s for s in self.shapes if s.label == label]

    def get_unique_labels(self):
        """Get list of unique labels."""
        return list(set(s.label for s in self.shapes if s.label))

    def __len__(self):
        return len(self.shapes)

    def __iter__(self):
        return iter(self.shapes)

    def __getitem__(self, index):
        return self.shapes[index]
