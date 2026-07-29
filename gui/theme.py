# gui/theme.py
"""
Theme system for yolo11-train UI.
Reference: X-AnyLabeling theme system with light/dark mode support.
"""

import os
import subprocess
from typing import Dict, Optional

from gui.qt_compat import QColor, QPalette

try:
    import darkdetect as _darkdetect
except ImportError:
    _darkdetect = None

# Light theme colors
LIGHT_THEME: Dict[str, str] = {
    "primary": "#0071e3",
    "primary_hover": "#0077ED",
    "primary_active": "#0068D0",
    "primary_pressed": "#0068D0",
    "background": "#ffffff",
    "background_secondary": "#F9F9F9",
    "background_hover": "#DBDBDB",
    "surface": "#f5f5f7",
    "surface_hover": "#e5e5e5",
    "surface_pressed": "#d5d5d5",
    "border": "#E5E5E5",
    "border_light": "#d2d2d7",
    "text": "#1d1d1f",
    "text_secondary": "#86868b",
    "text_placeholder": "#718096",
    "text_muted": "#94a3b8",
    "highlight": "#60A5FA",
    "highlight_text": "#2196F3",
    "success": "#30D158",
    "warning": "#FF9F0A",
    "error": "#FF453A",
    "scrollbar": "#c1c1c1",
    "scrollbar_hover": "#a8a8a8",
    "selection": "#0071e3",
    "selection_text": "#ffffff",
    "tooltip_bg": "#1d1d1f",
    "tooltip_text": "#f5f5f7",
    "spinbox_button": "#f0f0f0",
    "spinbox_button_hover": "#e0e0e0",
    "card_bg": "#ffffff",
    "card_selected_bg": "#e6f7ff",
    "card_selected_border": "#1890ff",
}

# Dark theme colors
DARK_THEME: Dict[str, str] = {
    "primary": "#0A84FF",
    "primary_hover": "#409CFF",
    "primary_active": "#0071e3",
    "primary_pressed": "#0071e3",
    "background": "#1c1c1e",
    "background_secondary": "#2c2c2e",
    "background_hover": "#3a3a3c",
    "surface": "#2c2c2e",
    "surface_hover": "#3a3a3c",
    "surface_pressed": "#48484a",
    "border": "#3a3a3c",
    "border_light": "#48484a",
    "text": "#f5f5f7",
    "text_secondary": "#aeaeb2",
    "text_placeholder": "#8e8e93",
    "text_muted": "#636366",
    "highlight": "#409CFF",
    "highlight_text": "#409CFF",
    "success": "#30D158",
    "warning": "#FF9F0A",
    "error": "#FF453A",
    "scrollbar": "#48484a",
    "scrollbar_hover": "#636366",
    "selection": "#0A84FF",
    "selection_text": "#ffffff",
    "tooltip_bg": "#2c2c2e",
    "tooltip_text": "#f5f5f7",
    "spinbox_button": "#3a3a3c",
    "spinbox_button_hover": "#48484a",
    "card_bg": "#2c2c2e",
    "card_selected_bg": "#1a3a5c",
    "card_selected_border": "#0A84FF",
}

# Default fallback theme (light)
_DEFAULT_THEME: Dict[str, str] = {
    "background": "#ffffff",
    "background_secondary": "#f8fafc",
    "surface": "#ffffff",
    "surface_hover": "#f1f5f9",
    "surface_pressed": "#e2e8f0",
    "primary": "#1890ff",
    "primary_hover": "#40a9ff",
    "primary_active": "#096dd9",
    "secondary": "#f3f4f6",
    "secondary_hover": "#e5e7eb",
    "text": "#1e293b",
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "border": "#cbd5e1",
    "border_light": "#e2e8f0",
    "spinbox_button": "#f1f5f9",
    "spinbox_button_hover": "#cbd5e1",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "card_bg": "#ffffff",
    "card_selected_bg": "#e6f7ff",
    "card_selected_border": "#1890ff",
}

_active_mode: str = "light"
_active_theme: Dict[str, str] = LIGHT_THEME


def _is_wsl() -> bool:
    """Returns True when the process is running inside WSL (1 or 2)."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _wsl_windows_theme() -> str:
    """Reads the Windows host dark/light preference from inside WSL."""
    try:
        result = subprocess.run(
            [
                "reg.exe", "query",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                "/v", "AppsUseLightTheme",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "AppsUseLightTheme" in line:
                    return "light" if "0x1" in line else "dark"
    except Exception:
        pass
    return "light"


def _detect_system_theme() -> str:
    """Returns the OS dark/light preference as 'dark' or 'light'."""
    if _is_wsl():
        return _wsl_windows_theme()

    if _darkdetect is not None:
        try:
            result = _darkdetect.theme()
            if isinstance(result, str):
                return "dark" if result.lower() == "dark" else "light"
        except Exception:
            pass

    return "light"


def init_theme(mode: str = "auto") -> None:
    """
    Initializes the global theme state at application startup.

    Args:
        mode: 'auto' (default), 'light', or 'dark'.
              'auto' resolves the OS preference via _detect_system_theme().
    """
    global _active_mode, _active_theme
    if mode == "auto":
        resolved = _detect_system_theme()
    elif mode in ("light", "dark"):
        resolved = mode
    else:
        resolved = "light"
    _active_mode = resolved
    _active_theme = DARK_THEME if resolved == "dark" else LIGHT_THEME


def get_theme() -> Dict[str, str]:
    """Returns the currently active theme color dictionary."""
    return _active_theme


def get_mode() -> str:
    """Returns the currently active theme mode ('light' or 'dark')."""
    return _active_mode


def _checkbox_indicator_qss() -> str:
    """Returns QSS for QCheckBox indicator sub-controls."""
    t = _active_theme
    if _active_mode == "dark":
        checked_bg = t["primary"]
        checked_border = t["primary"]
    else:
        checked_bg = "#ffffff"
        checked_border = t["border_light"]
    return f"""
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {t["border_light"]};
            background-color: {t["background_secondary"]};
        }}
        QCheckBox::indicator:hover {{
            border-color: {t["primary"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {checked_bg};
            border: 1px solid {checked_border};
        }}
    """


def get_app_stylesheet() -> str:
    """
    Returns a comprehensive QSS stylesheet for the active theme.
    Returns empty string for light mode to preserve original behavior.
    """
    if _active_mode == "light":
        return ""
    t = _active_theme
    return f"""
        QMainWindow, QDialog {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
        QScrollBar:vertical {{
            background-color: {t["background_secondary"]};
            width: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background-color: {t["scrollbar"]};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
        QScrollBar:horizontal {{
            background-color: {t["background_secondary"]};
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {t["scrollbar"]};
            border-radius: 4px;
            min-width: 20px;
        }}
    """


def get_dark_palette() -> Optional[QPalette]:
    """
    Returns a QPalette configured for the active dark theme, or None in light mode.
    """
    if _active_mode != "dark":
        return None
    t = _active_theme
    palette = QPalette()
    bg = QColor(t["background"])
    bg2 = QColor(t["background_secondary"])
    text = QColor(t["text"])
    text2 = QColor(t["text_secondary"])
    surface = QColor(t["surface"])
    highlight = QColor(t["selection"])
    highlight_text = QColor(t["selection_text"])

    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        palette.setColor(group, QPalette.ColorRole.Window, bg)
        palette.setColor(group, QPalette.ColorRole.WindowText, text)
        palette.setColor(group, QPalette.ColorRole.Base, bg2)
        palette.setColor(group, QPalette.ColorRole.AlternateBase, bg)
        palette.setColor(group, QPalette.ColorRole.Text, text)
        palette.setColor(group, QPalette.ColorRole.Button, surface)
        palette.setColor(group, QPalette.ColorRole.ButtonText, text)
        palette.setColor(group, QPalette.ColorRole.Highlight, highlight)
        palette.setColor(group, QPalette.ColorRole.HighlightedText, highlight_text)

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, text2)

    return palette


# Legacy compatibility - kept for existing code references
_THEME = LIGHT_THEME


def get_ultralytics_dialog_style() -> str:
    """Returns the main dialog stylesheet."""
    t = get_theme()
    return f"""
        QWidget {{
            background-color: {t["background"]};
            color: {t["text"]};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        QTabWidget::pane {{
            border: 1px solid {t["border"]};
            border-radius: 8px;
            background-color: {t["background"]};
        }}
        QTabBar::tab {{
            background: {t["surface"]};
            color: {t["text_secondary"]};
            border: 1px solid {t["border"]};
            border-bottom: none;
            padding: 10px 28px;
            min-width: 100px;
            font-weight: 600;
            font-size: 14px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 6px;
        }}
        QTabBar::tab:selected {{
            background: {t["primary"]};
            color: #ffffff;
            border-color: {t["primary"]};
            font-weight: 700;
        }}
        QTabBar::tab:hover:!selected {{
            background: {t["surface_hover"]};
            color: {t["text"]};
        }}
        QGroupBox {{
            font-weight: bold;
            font-size: 14px;
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 14px;
            background-color: {t["card_bg"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
            background-color: {t["background"]};
            border-radius: 4px;
            color: {t["primary"]};
        }}
        QScrollBar:vertical {{
            background-color: {t["background_secondary"]};
            width: 8px;
            margin: 0;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {t["scrollbar"]};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
        QScrollBar:vertical:hover {{
            background-color: {t["surface_hover"]};
        }}
        QScrollBar:horizontal {{
            background-color: {t["background_secondary"]};
            height: 8px;
            margin: 0;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {t["scrollbar"]};
            border-radius: 4px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
    """


def get_checkbox_indicator_style() -> str:
    """Returns QSS for QCheckBox indicator."""
    return _checkbox_indicator_qss()


def get_ok_btn_style() -> str:
    """Returns primary button stylesheet."""
    t = get_theme()
    return f"""
    QPushButton {{
        background: {t["primary"]};
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 8px 22px;
        min-height: 36px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: {t["primary_hover"]};
    }}
    QPushButton:pressed {{
        background: {t["primary_pressed"]};
    }}
    QPushButton:disabled {{
        background: {t["surface"]};
        color: {t["text_muted"]};
        border: 1px solid {t["border"]};
    }}
    """


def get_cancel_btn_style() -> str:
    """Returns secondary button stylesheet."""
    t = get_theme()
    return f"""
    QPushButton {{
        background: {t["surface"]};
        color: {t["text"]};
        border: 1px solid {t["border"]};
        border-radius: 6px;
        padding: 8px 18px;
        min-height: 36px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: {t["surface_hover"]};
        border-color: {t["primary"]};
    }}
    QPushButton:disabled {{
        background: {t["background_secondary"]};
        color: {t["text_muted"]};
        border-color: {t["border"]};
    }}
    """


def get_danger_btn_style() -> str:
    """Returns danger button stylesheet."""
    t = get_theme()
    return f"""
    QPushButton {{
        background: {t["error"]};
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 8px 18px;
        min-height: 36px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: #ff6b6b;
    }}
    QPushButton:pressed {{
        background: #cc3a3a;
    }}
    """


def get_custom_table_style() -> str:
    """Returns table widget stylesheet."""
    t = get_theme()
    return f"""
        QTableWidget {{
            border: 1px solid {t["border"]};
            border-radius: 8px;
            background-color: {t["background"]};
            gridline-color: {t["border"]};
            outline: none;
        }}
        QTableWidget::item {{
            padding: 10px 14px;
            border: none;
            border-bottom: 1px solid {t["border"]};
            color: {t["text"]};
            font-size: 13px;
        }}
        QTableWidget::item:hover {{
            background-color: {t["surface_hover"]};
        }}
        QTableWidget::item:selected {{
            background-color: {t["selection"]};
            color: {t["selection_text"]};
        }}
        QHeaderView::section {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 12px 14px;
            border: none;
            border-bottom: 2px solid {t["border"]};
        }}
        QTableCornerButton::section {{
            background-color: {t["surface"]};
            border: none;
        }}
    """


def get_progress_bar_style() -> str:
    """Returns progress bar stylesheet."""
    t = get_theme()
    return f"""
        QProgressBar {{
            border: 1px solid {t["border"]};
            border-radius: 6px;
            text-align: center;
            height: 24px;
            background-color: {t["surface"]};
            color: {t["text"]};
        }}
        QProgressBar::chunk {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {t["primary"]},
                stop:0.5 {t["primary_hover"]},
                stop:1 {t["primary"]}
            );
            border-radius: 5px;
        }}
    """


def get_log_display_style() -> str:
    """Returns log text display stylesheet."""
    t = get_theme()
    return f"""
        QTextEdit {{
            background-color: {t["surface"]};
            color: {t["text"]};
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 8px;
        }}
    """


def get_status_label_style(color: str = None) -> str:
    """Returns status label stylesheet."""
    if color is None:
        color = get_theme()["text_secondary"]
    return f"""
        font-size: 14px;
        font-weight: bold;
        color: {color};
    """
