from gui.qt_compat import QDialog, QVBoxLayout
from gui.ultralytics_widget import UltralyticsWidget
from services.config import DEFAULT_WINDOW_TITLE

class UltralyticsDialog(QDialog):
    """Legacy dialog wrapper for UltralyticsWidget"""
    def __init__(self, parent=None, dataset_dir=None):
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(1100, 840)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget = UltralyticsWidget(self, dataset_dir=dataset_dir)
        layout.addWidget(self.widget)
