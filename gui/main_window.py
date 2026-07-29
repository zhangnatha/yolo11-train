import os
import sys
from pathlib import Path

from gui.qt_compat import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QApplication,
)

from gui.theme import init_theme, get_theme, get_mode, get_app_stylesheet, get_dark_palette
from gui.ultralytics_widget import UltralyticsWidget
from services.config import get_project_root


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
        self.workflow_widget = UltralyticsWidget(self, dataset_dir=self.current_dataset)
        self.setCentralWidget(self.workflow_widget)

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
