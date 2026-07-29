# gui/widgets/custom_widgets.py
"""
Custom widgets with X-AnyLabeling-inspired styling.
Reference: X-AnyLabeling views/training/widgets/ultralytics_widgets/custom_widgets.py
"""

import os

from gui.qt_compat import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
    QProgressBar,
    QGroupBox,
    QAbstractItemView,
    QScrollArea,
    Qt,
    QBrush,
    QColor,
    QPixmap,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
    exec_dialog,
)

from gui.theme import (
    get_theme,
    get_mode,
    get_ok_btn_style,
    get_cancel_btn_style,
    get_checkbox_indicator_style,
    get_custom_table_style,
    get_progress_bar_style,
)


class CustomCheckBox(QCheckBox):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        t = get_theme()
        self.setStyleSheet(f"""
            QCheckBox {{
                spacing: 8px;
                color: {t["text"]};
                font-size: 13px;
            }}
            {get_checkbox_indicator_style()}
        """)


class CustomComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = get_theme()
        mode = get_mode()
        if hasattr(self.view(), "setFocusPolicy"):
            self.view().setFocusPolicy(Qt.FocusPolicy.NoFocus if hasattr(Qt, "FocusPolicy") else Qt.NoFocus)

        # Dark mode uses different background for dropdown
        dropdown_bg = t["background_secondary"] if mode == "dark" else "#ffffff"
        selection_bg = t["selection"] if mode == "dark" else t["primary"]

        self.setStyleSheet(f"""
            QComboBox {{
                padding: 6px 12px;
                background: {t["background_secondary"]};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                min-height: 26px;
                color: {t["text"]};
                font-size: 13px;
            }}
            QComboBox:hover {{
                border-color: {t["primary"]};
                background: {t["surface_hover"]};
            }}
            QComboBox:focus {{
                border-color: {t["primary"]};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {t["text_secondary"]};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {dropdown_bg};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                padding: 4px;
                selection-background-color: {selection_bg};
                selection-color: #ffffff;
                color: {t["text"]};
                outline: none;
            }}
        """)

    def wheelEvent(self, event):
        event.ignore()


class CustomSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = get_theme()
        self.setStyleSheet(f"""
            QSpinBox {{
                padding: 6px 10px;
                background: {t["background_secondary"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                min-height: 26px;
                font-size: 13px;
            }}
            QSpinBox:hover {{
                border-color: {t["primary"]};
                background: {t["surface_hover"]};
            }}
            QSpinBox:focus {{
                border-color: {t["primary"]};
            }}
        """)

    def wheelEvent(self, event):
        event.ignore()


class CustomDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = get_theme()
        self.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 6px 10px;
                background: {t["background_secondary"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                min-height: 26px;
                font-size: 13px;
            }}
            QDoubleSpinBox:hover {{
                border-color: {t["primary"]};
                background: {t["surface_hover"]};
            }}
            QDoubleSpinBox:focus {{
                border-color: {t["primary"]};
            }}
        """)

    def wheelEvent(self, event):
        event.ignore()


class CustomLineEdit(QLineEdit):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        t = get_theme()
        mode = get_mode()
        # Dark mode uses different focus background
        focus_bg = t["surface"] if mode == "dark" else "#ffffff"
        self.setStyleSheet(f"""
            QLineEdit {{
                padding: 6px 10px;
                background: {t["background_secondary"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                min-height: 26px;
                font-size: 13px;
            }}
            QLineEdit:hover {{
                border-color: {t["primary"]};
                background: {t["surface_hover"]};
            }}
            QLineEdit:focus {{
                border-color: {t["primary"]};
                background: {focus_bg};
            }}
            QLineEdit:disabled {{
                background: {t["surface"]};
                color: {t["text_muted"]};
            }}
        """)


class CustomSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal if hasattr(Qt, "Horizontal") else Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        t = get_theme()
        mode = get_mode()
        handle_bg = "#ffffff" if mode == "dark" else "#ffffff"
        self.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {t["border_light"]};
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {t["primary"]};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {t["primary"]};
                border: 2px solid {handle_bg};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {t["primary_hover"]};
            }}
        """)

    def wheelEvent(self, event):
        event.ignore()


class CustomQPushButton(QPushButton):
    """Selectable Card Button matching X-AnyLabeling styled selection cards"""
    def __init__(self, text="", parent=None, subtitle=""):
        super().__init__(text, parent)
        self.selected = False
        self.subtitle = subtitle
        self.setFixedHeight(40)
        self.setMinimumWidth(90)
        if hasattr(Qt, "NoFocus"):
            self.setFocusPolicy(Qt.NoFocus)
        elif hasattr(Qt, "FocusPolicy"):
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.update_style()

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update_style()

    def update_style(self):
        t = get_theme()
        if self.selected:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {t["primary"]};
                    color: #ffffff;
                    border: 2px solid {t["primary_active"]};
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-weight: bold;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: {t["primary_hover"]};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {t["card_bg"]};
                    color: {t["text"]};
                    border: 1px solid {t["border"]};
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: {t["surface_hover"]};
                    border-color: {t["primary"]};
                    color: {t["primary"]};
                }}
            """)


class PrimaryButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(get_ok_btn_style())
        if hasattr(Qt, "PointingHandCursor"):
            self.setCursor(Qt.PointingHandCursor)


class SecondaryButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(get_cancel_btn_style())
        if hasattr(Qt, "PointingHandCursor"):
            self.setCursor(Qt.PointingHandCursor)


class DangerButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        t = get_theme()
        self.setStyleSheet(f"""
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
                background: #dc2626;
            }}
            QPushButton:disabled {{
                background: {t["border_light"]};
                color: {t["text_muted"]};
            }}
        """)
        if hasattr(Qt, "PointingHandCursor"):
            self.setCursor(Qt.PointingHandCursor)


class CustomTable(QTableWidget):
    """Table Widget with X-AnyLabeling custom section headers & row hover effects"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_table()

    def setup_table(self):
        no_edit = QAbstractItemView.NoEditTriggers if hasattr(QAbstractItemView, "NoEditTriggers") else QAbstractItemView.EditTrigger.NoEditTriggers
        no_select = QAbstractItemView.NoSelection if hasattr(QAbstractItemView, "NoSelection") else QAbstractItemView.SelectionMode.NoSelection
        stretch_mode = QHeaderView.Stretch if hasattr(QHeaderView, "Stretch") else QHeaderView.ResizeMode.Stretch

        self.setEditTriggers(no_edit)
        self.setSelectionMode(no_select)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(stretch_mode)
        self.setStyleSheet(get_custom_table_style())

    def load_data(self, data):
        if not data:
            self.clear()
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        headers = data[0]
        rows = data[1:]

        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setRowCount(len(rows))

        t = get_theme()
        center_align = Qt.AlignCenter if hasattr(Qt, "AlignCenter") else Qt.AlignmentFlag.AlignCenter
        for r_idx, row in enumerate(rows):
            is_total_row = (r_idx == len(rows) - 1 and str(row[0]).lower() in ["total", "合计", "总计"])
            for c_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(center_align)
                if is_total_row:
                    item.setForeground(QBrush(QColor(t["primary"])))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.setItem(r_idx, c_idx, item)


class ClickableImageLabel(QLabel):
    """Clickable thumbnail image label for displaying training metrics & plots"""
    def __init__(self, text="No Plot Available", parent=None):
        super().__init__(text, parent)
        self.image_path = None
        self.setAlignment(Qt.AlignCenter if hasattr(Qt, "AlignCenter") else Qt.AlignmentFlag.AlignCenter)
        if hasattr(Qt, "PointingHandCursor"):
            self.setCursor(Qt.PointingHandCursor)
        elif hasattr(Qt, "CursorShape"):
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if self.image_path and os.path.exists(self.image_path):
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Plot Preview - {os.path.basename(self.image_path)}")
            dlg_layout = QVBoxLayout(dlg)
            lbl = QLabel()
            pix = QPixmap(self.image_path)
            if not pix.isNull():
                keep_aspect = Qt.KeepAspectRatio if hasattr(Qt, "KeepAspectRatio") else Qt.AspectRatioMode.KeepAspectRatio
                smooth_trans = Qt.SmoothTransformation if hasattr(Qt, "SmoothTransformation") else Qt.TransformationMode.SmoothTransformation
                scaled = pix.scaled(980, 750, keep_aspect, smooth_trans)
                lbl.setPixmap(scaled)
                dlg_layout.addWidget(lbl)
                exec_dialog(dlg)


from gui.theme import get_theme


class ZoomableImageGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        anchor_under_mouse = getattr(QGraphicsView.ViewportAnchor, "AnchorUnderMouse", None) if hasattr(QGraphicsView, "ViewportAnchor") else getattr(QGraphicsView, "AnchorUnderMouse", None)
        drag_mode = getattr(QGraphicsView.DragMode, "ScrollHandDrag", None) if hasattr(QGraphicsView, "DragMode") else getattr(QGraphicsView, "ScrollHandDrag", None)
        viewport_update = getattr(QGraphicsView.ViewportUpdateMode, "FullViewportUpdate", None) if hasattr(QGraphicsView, "ViewportUpdateMode") else getattr(QGraphicsView, "FullViewportUpdate", None)
        scrollbar_as_needed = getattr(Qt.ScrollBarPolicy, "ScrollBarAsNeeded", None) if hasattr(Qt, "ScrollBarPolicy") else getattr(Qt, "ScrollBarAsNeeded", None)

        if anchor_under_mouse is not None:
            self.setTransformationAnchor(anchor_under_mouse)
            self.setResizeAnchor(anchor_under_mouse)
        if drag_mode is not None:
            self.setDragMode(drag_mode)
        if viewport_update is not None:
            self.setViewportUpdateMode(viewport_update)
        if scrollbar_as_needed is not None:
            self.setHorizontalScrollBarPolicy(scrollbar_as_needed)
            self.setVerticalScrollBarPolicy(scrollbar_as_needed)

        t = get_theme()
        self.setStyleSheet(f"""
            QGraphicsView {{
                border: 1px solid {t['border']};
                border-radius: 6px;
                background-color: {t['background_secondary']};
            }}
        """)
        self._image_path = None
        self._current_scale = 1.0

    def set_image(self, image_path):
        self._image_path = image_path
        t = get_theme()
        if not image_path or not os.path.exists(image_path):
            self.scene.clear()
            self.pixmap_item = QGraphicsPixmapItem()
            self.scene.addItem(self.pixmap_item)
            text_item = self.scene.addText("No inference result image")
            text_item.setDefaultTextColor(QColor(t.get('text_muted', '#64748b')))
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.scene.clear()
            self.pixmap_item = QGraphicsPixmapItem()
            self.scene.addItem(self.pixmap_item)
            text_item = self.scene.addText("Failed to load inference result image")
            text_item.setDefaultTextColor(QColor(t.get('error', '#ff4d4f')))
            return

        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        self.resetTransform()
        self._current_scale = 1.0
        keep_aspect = Qt.KeepAspectRatio if hasattr(Qt, "KeepAspectRatio") else Qt.AspectRatioMode.KeepAspectRatio
        self.fitInView(self.pixmap_item, keep_aspect)

    def wheelEvent(self, event):
        if not self._image_path or not self.pixmap_item or self.pixmap_item.pixmap().isNull():
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        new_scale = self._current_scale * zoom_factor
        if 0.05 <= new_scale <= 50.0:
            self._current_scale = new_scale
            self.scale(zoom_factor, zoom_factor)

    def mouseDoubleClickEvent(self, event):
        if self.pixmap_item and not self.pixmap_item.pixmap().isNull():
            self.resetTransform()
            self._current_scale = 1.0
            keep_aspect = Qt.KeepAspectRatio if hasattr(Qt, "KeepAspectRatio") else Qt.AspectRatioMode.KeepAspectRatio
            self.fitInView(self.pixmap_item, keep_aspect)
        super().mouseDoubleClickEvent(event)


class ZoomableImageWidget(QWidget):
    """Interactive Zoomable Image Viewer Component with Mouse Position Zoom & Dragging (No Toolbar Buttons)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view = ZoomableImageGraphicsView(self)
        layout.addWidget(self.view, 1)

    def set_image(self, image_path):
        self.image_path = image_path
        self.view.set_image(image_path)

    def update_display(self):
        self.view.set_image(self.image_path)


class TrainingConfirmDialog(QDialog):
    """Command & Config Preview Confirmation Dialog matching X-AnyLabeling"""
    def __init__(self, parent=None, config=None, cmd_str=""):
        super().__init__(parent)
        self.setWindowTitle("Confirm Training Task Configuration")
        self.setFixedSize(720, 420)
        t = get_theme()
        self.setStyleSheet(f"background-color: {t['background']}; color: {t['text']};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title_label = QLabel("Ready to Start YOLO Training Task")
        title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {t['primary']};")
        layout.addWidget(title_label)

        desc_label = QLabel("The following command and parameters will be used to start the YOLO training engine:")
        desc_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 13px;")
        layout.addWidget(desc_label)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {t["background_secondary"]};
                color: {t["text"]};
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid {t["border"]};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        layout.addWidget(self.text_edit, 1)

        display_lines = []
        if cmd_str:
            display_lines.append(f"[Command]:\n{cmd_str}\n")

        if config:
            display_lines.append("[Configuration Summary]:")
            for cat, sub_cfg in config.items():
                display_lines.append(f"  - {cat.upper()}:")
                if isinstance(sub_cfg, dict):
                    for k, v in sub_cfg.items():
                        display_lines.append(f"      - {k}: {v}")
                else:
                    display_lines.append(f"      - {sub_cfg}")
        self.text_edit.setText("\n".join(display_lines))

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = SecondaryButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.confirm_btn = PrimaryButton("Confirm Start Training")
        self.confirm_btn.clicked.connect(self.accept)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.confirm_btn)
        layout.addLayout(btn_layout)


class AdvancedToggleButton(QPushButton):
    """Toggle button for advanced settings collapse/expand"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.expanded = False
        self.setFixedSize(24, 24)
        t = get_theme()
        self.setStyleSheet(f"""
            QPushButton {{
                border: none;
                text-align: center;
                font-weight: bold;
                font-size: 10px;
                background-color: transparent;
                color: {t["text"]};
            }}
            QPushButton:hover {{
                background-color: {t["surface_hover"]};
                border-radius: 4px;
            }}
        """)
        self.update_icon()

    def set_expanded(self, expanded: bool):
        self.expanded = expanded
        self.update_icon()

    def update_icon(self):
        icon = ">" if not self.expanded else "v"
        self.setText(icon)

    def mousePressEvent(self, event):
        self.expanded = not self.expanded
        self.update_icon()
        super().mousePressEvent(event)


class CustomProgressBar(QProgressBar):
    """Custom styled progress bar matching X-AnyLabeling design"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(get_progress_bar_style())
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignCenter if hasattr(Qt, "AlignCenter") else Qt.AlignmentFlag.AlignCenter)


class CollapsibleGroupBox(QGroupBox):
    """Collapsible group box for advanced settings"""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.expanded = False
        t = get_theme()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header with toggle button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(6, 6, 6, 6)

        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setStyleSheet(f"color: {t['text']};")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.toggle_btn = AdvancedToggleButton()
        self.toggle_btn.clicked.connect(self.toggle)
        header_layout.addWidget(self.toggle_btn)

        main_layout.addWidget(header_widget)

        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 5, 10, 10)
        self.content_widget.setVisible(False)
        main_layout.addWidget(self.content_widget)

        # Style
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {t["border"]};
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: {t["card_bg"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color: {t["text"]};
            }}
        """)

    def toggle(self):
        self.expanded = not self.expanded
        self.content_widget.setVisible(self.expanded)
        self.toggle_btn.set_expanded(self.expanded)

    def content_layout(self):
        return self.content_layout
