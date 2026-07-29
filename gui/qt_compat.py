import sys

# Cross-platform Qt compatibility wrapper (PyQt5 & PyQt6)
# Supports Ubuntu 18.04 (PyQt5) and Windows/Ubuntu 20+ (PyQt5/PyQt6)

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPoint, QRect, QSize
    from PyQt5.QtGui import QIcon, QPixmap, QBrush, QColor, QFont, QCursor, QPalette
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QMessageBox, QScrollArea, QGroupBox,
        QFileDialog, QFormLayout, QGridLayout, QProgressBar, QTextEdit, QLineEdit,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QTableWidget,
        QTableWidgetItem, QHeaderView, QAbstractItemView, QFrame, QAction, QMenuBar, QMenu,
        QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QListWidget, QListWidgetItem
    )
    QT_VERSION = 5
except ImportError:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPoint, QRect, QSize
    from PyQt6.QtGui import QIcon, QPixmap, QBrush, QColor, QFont, QCursor, QAction, QPalette
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QMessageBox, QScrollArea, QGroupBox,
        QFileDialog, QFormLayout, QGridLayout, QProgressBar, QTextEdit, QLineEdit,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QTableWidget,
        QTableWidgetItem, QHeaderView, QAbstractItemView, QFrame, QMenuBar, QMenu,
        QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QListWidget, QListWidgetItem
    )
    QT_VERSION = 6

# PyQt6 compatibility patch: map scoped enums to top-level Qt attribute names if missing
if QT_VERSION == 6:
    for enum_name, targets in [
        ("TextElideMode", ["ElideNone", "ElideLeft", "ElideRight", "ElideMiddle"]),
        ("AlignmentFlag", ["AlignCenter", "AlignLeft", "AlignRight", "AlignTop", "AlignBottom", "AlignHCenter", "AlignVCenter"]),
        ("AspectRatioMode", ["KeepAspectRatio", "KeepAspectRatioByExpanding", "IgnoreAspectRatio"]),
        ("TransformationMode", ["SmoothTransformation", "FastTransformation"]),
        ("Orientation", ["Horizontal", "Vertical"]),
        ("FocusPolicy", ["NoFocus", "TabFocus", "ClickFocus", "StrongFocus", "WheelFocus"]),
        ("CursorShape", ["PointingHandCursor", "ArrowCursor", "WaitCursor", "IBeamCursor"]),
        ("ScrollBarPolicy", ["ScrollBarAsNeeded", "ScrollBarAlwaysOff", "ScrollBarAlwaysOn"]),
        ("MouseButton", ["LeftButton", "RightButton", "MiddleButton"]),
        ("KeyboardModifier", ["ControlModifier", "ShiftModifier", "AltModifier"]),
    ]:
        if hasattr(Qt, enum_name):
            enum_cls = getattr(Qt, enum_name)
            for val in targets:
                if hasattr(enum_cls, val) and not hasattr(Qt, val):
                    try:
                        setattr(Qt, val, getattr(enum_cls, val))
                    except Exception:
                        pass

def exec_dialog(dialog_instance):
    """Executes dialog compatible with both Qt5 (exec_()) and Qt6 (exec())"""
    if hasattr(dialog_instance, "exec"):
        return dialog_instance.exec()
    return dialog_instance.exec_()

def exec_app(app_instance):
    """Executes application loop compatible with both Qt5 and Qt6"""
    if hasattr(app_instance, "exec"):
        return app_instance.exec()
    return app_instance.exec_()
