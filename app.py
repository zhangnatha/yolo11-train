import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from services.config import init_pretrained_model_env
from gui.qt_compat import QApplication, exec_app
from gui.theme import init_theme, get_dark_palette
from gui.main_window import MainWindow

def main():
    # Initialize pretrained model environment
    init_pretrained_model_env()

    # Initialize theme (auto-detect system preference)
    init_theme("auto")

    # Create application
    app = QApplication(sys.argv)

    # Apply dark palette if needed
    palette = get_dark_palette()
    if palette:
        app.setPalette(palette)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(exec_app(app))

if __name__ == "__main__":
    main()
