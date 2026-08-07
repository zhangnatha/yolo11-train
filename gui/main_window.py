import os
import sys
from pathlib import Path

from gui.qt_compat import (
    Qt,
    QtCore,
    QtGui,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QApplication,
)

from gui.theme import init_theme, get_theme, get_mode, get_app_stylesheet, get_dark_palette
from gui.ultralytics_widget import UltralyticsWidget
from services.config import get_project_root


class CarvedLabel(QLabel):
    """Gradient engraved credit label shared with the segmentation GUI style."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("background: transparent; padding: 0 4px;")
        font = self.font()
        font.setFamily("DejaVu Sans, Arial, Helvetica, sans-serif")
        font.setPointSize(10)
        font.setBold(True)
        self.setFont(font)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        render_hint = (QtGui.QPainter.RenderHint
                       if hasattr(QtGui.QPainter, "RenderHint") else QtGui.QPainter)
        painter.setRenderHint(render_hint.Antialiasing)
        painter.setRenderHint(render_hint.TextAntialiasing)

        rect = QtCore.QRectF(self.rect())
        alignment = self.alignment() | Qt.AlignVCenter
        painter.setPen(QtGui.QColor(255, 255, 255, 220))
        painter.drawText(rect.translated(1.0, 1.0), alignment, self.text())
        painter.setPen(QtGui.QColor(15, 15, 20, 210))
        painter.drawText(rect.translated(-1.0, -1.0), alignment, self.text())
        painter.setPen(QtGui.QColor(0, 0, 0, 140))
        painter.drawText(rect.translated(0.0, -1.0), alignment, self.text())

        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0.0, QtGui.QColor(0, 210, 255))
        gradient.setColorAt(0.45, QtGui.QColor(0, 114, 255))
        gradient.setColorAt(0.8, QtGui.QColor(155, 44, 243))
        gradient.setColorAt(1.0, QtGui.QColor(247, 37, 133))
        painter.setPen(QtGui.QPen(QtGui.QBrush(gradient), 0))
        painter.drawText(rect, alignment, self.text())
        painter.setPen(QtGui.QColor(0, 0, 0, 45))
        painter.drawText(rect.translated(0.0, -0.5), alignment, self.text())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Training & Inference Platform")
        self.resize(1150, 880)

        self.current_dataset = ""
        self.init_central_widget()
        self.apply_theme()

    def closeEvent(self, event):
        if hasattr(self, "workflow_widget") and hasattr(self.workflow_widget, "save_config"):
            self.workflow_widget.save_config()
        super().closeEvent(event)

    def init_central_widget(self):
        # Single Unified Interface with Multi-Tab Guided Workflow (No Menu Bar)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 4, 0)
        top_bar.setSpacing(0)
        top_bar.addStretch()
        self.lbl_carved_credits = CarvedLabel(
            "powered by zhangjianan/zhangnatha@qq.com", central)
        self.lbl_carved_credits.setFixedHeight(20)
        top_bar.addWidget(self.lbl_carved_credits)
        layout.addLayout(top_bar)

        self.workflow_widget = UltralyticsWidget(self, dataset_dir=self.current_dataset)
        layout.addWidget(self.workflow_widget)
        self.setCentralWidget(central)

    def apply_theme(self):
        t = get_theme()
        mode = get_mode()

        # Apply dark palette if needed
        palette = get_dark_palette()
        if palette:
            self.setPalette(palette)

        # Apply additional stylesheet
        base_style = f"""
            QMainWindow {{
                background-color: {t["background"]};
                color: {t["text"]};
            }}
        """
        app_style = get_app_stylesheet()
        self.setStyleSheet(base_style + app_style)
