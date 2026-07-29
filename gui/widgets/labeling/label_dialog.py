# gui/widgets/labeling/label_dialog.py
"""
Label dialog for shape labeling.
Reference: X-AnyLabeling views/labeling/widgets/label_dialog.py
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox
)


class LabelDialog(QDialog):
    """Dialog for selecting or creating labels."""

    def __init__(self, parent=None, labels=None, sort_labels=True):
        super().__init__(parent)
        self.setWindowTitle("Select Label")
        self.resize(300, 400)

        self.labels = labels if labels else []
        self.sort_labels = sort_labels
        self.current_label = ""

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Label list
        self.list_widget = QListWidget()
        for label in self.labels:
            item = QListWidgetItem(label)
            self.list_widget.addItem(item)

        if self.sort_labels:
            self.list_widget.sortItems()

        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)

        # Label edit
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label:"))
        self.label_edit = QLineEdit()
        label_layout.addWidget(self.label_edit)
        layout.addLayout(label_layout)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def on_item_clicked(self, item):
        """Handle item click."""
        self.label_edit.setText(item.text())

    def accept(self):
        """Accept dialog."""
        self.current_label = self.label_edit.text().strip()
        super().accept()

    def reject(self):
        """Reject dialog."""
        self.current_label = ""
        super().reject()

    @staticmethod
    def get_label(parent=None, labels=None, default=""):
        """Static method to show dialog and get label."""
        dialog = LabelDialog(parent, labels)
        dialog.label_edit.setText(default)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            return dialog.current_label
        return None


class LabelListWidget(QListWidget):
    """List widget for displaying unique labels with color indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_colors = {}

    def add_label(self, label, color=None):
        """Add a label to the list."""
        item = QListWidgetItem(label)
        if color:
            item.setBackground(QColor(color))
        self.addItem(item)
        self.label_colors[label] = color

    def update_labels(self, labels, colors=None):
        """Update the label list."""
        self.clear()
        for i, label in enumerate(labels):
            color = colors[i] if colors and i < len(colors) else None
            self.add_label(label, color)

    def get_selected_label(self):
        """Get currently selected label."""
        item = self.currentItem()
        if item:
            return item.text()
        return None
