import os
from pathlib import Path
from gui.qt_compat import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFormLayout,
    QMessageBox,
    QFileDialog,
)
from gui.theme import get_theme
from gui.widgets.custom_widgets import (
    CustomComboBox,
    CustomSpinBox,
    CustomCheckBox,
    CustomLineEdit,
    PrimaryButton,
    SecondaryButton,
)
from services.config import get_project_root

class ExportFormatDialog(QDialog):
    def __init__(self, parent=None, default_model_name="best"):
        super().__init__(parent)
        self.setWindowTitle("Export Model Configuration")
        self.resize(480, 320)
        t = get_theme()
        self.setStyleSheet(f"background-color: #ffffff; color: {t['text']};")

        project_root = get_project_root()
        default_dir = os.path.join(project_root, "models")
        os.makedirs(default_dir, exist_ok=True)
        
        clean_name = default_model_name
        if clean_name.endswith(".pt"):
            clean_name = clean_name[:-3]

        default_save_path = os.path.join(default_dir, f"{clean_name}.onnx")

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.format_combo = CustomComboBox()
        self.format_combo.addItems(["onnx", "engine", "torchscript"])
        self.format_combo.currentTextChanged.connect(self.update_default_extension)
        form_layout.addRow("Export Format:", self.format_combo)

        # Save Path selection with Browse button
        save_path_layout = QHBoxLayout()
        self.save_path_edit = CustomLineEdit(default_save_path)
        self.browse_btn = SecondaryButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_save_path)
        save_path_layout.addWidget(self.save_path_edit, 1)
        save_path_layout.addWidget(self.browse_btn)
        form_layout.addRow("Save Path:", save_path_layout)

        self.opset_spin = CustomSpinBox()
        self.opset_spin.setRange(11, 20)
        self.opset_spin.setValue(17)
        form_layout.addRow("Opset Version:", self.opset_spin)

        self.dynamic_cb = CustomCheckBox("Enable Dynamic Shape")
        form_layout.addRow("", self.dynamic_cb)

        self.simplify_cb = CustomCheckBox("Simplify ONNX Model")
        form_layout.addRow("", self.simplify_cb)

        layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        self.cancel_btn = SecondaryButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.export_btn = PrimaryButton("Export")
        self.export_btn.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)

    def update_default_extension(self, fmt):
        cur_path = self.save_path_edit.text().strip()
        if cur_path:
            p = Path(cur_path)
            ext_map = {"onnx": ".onnx", "engine": ".engine", "torchscript": ".torchscript"}
            new_ext = ext_map.get(fmt, f".{fmt}")
            
            name = p.name
            for old_ext in [".pt.onnx", ".pt.engine", ".pt.torchscript", ".onnx", ".engine", ".torchscript", ".pt"]:
                if name.endswith(old_ext):
                    name = name[:-len(old_ext)]
                    break
            
            new_path = str(p.parent / f"{name}{new_ext}")
            self.save_path_edit.setText(new_path)

    def browse_save_path(self):
        fmt = self.format_combo.currentText()
        ext_map = {"onnx": "ONNX Model (*.onnx)", "engine": "TensorRT Engine (*.engine)", "torchscript": "TorchScript (*.torchscript)"}
        filter_str = f"{ext_map.get(fmt, 'All Files (*)')};;All Files (*)"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Export Model Save Path",
            self.save_path_edit.text(),
            filter_str
        )
        if file_path:
            self.save_path_edit.setText(file_path)

    def get_export_config(self):
        return {
            "format": self.format_combo.currentText(),
            "save_path": self.save_path_edit.text().strip(),
            "opset": self.opset_spin.value(),
            "dynamic": self.dynamic_cb.isChecked(),
            "simplify": self.simplify_cb.isChecked(),
        }
