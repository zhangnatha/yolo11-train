# gui/ultralytics_widget.py

import os
import sys
import time
import json
import shutil
import glob
import re
import cv2
from pathlib import Path

from gui.qt_compat import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QLabel,
    QDialog,
    QMessageBox,
    QScrollArea,
    QGroupBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QProgressBar,
    QTextEdit,
    QApplication,
    QSizePolicy,
    QTimer,
    QThread,
    pyqtSignal,
    Qt,
    QPixmap,
    exec_dialog,
    QListWidget,
    QListWidgetItem,
    QTableWidgetItem,
    QHeaderView,
    QColor,
    QBrush,
)

from gui.theme import get_theme, get_ultralytics_dialog_style
from gui.widgets.custom_widgets import (
    CustomCheckBox,
    CustomComboBox,
    CustomSpinBox,
    CustomDoubleSpinBox,
    CustomLineEdit,
    CustomSlider,
    CustomQPushButton,
    PrimaryButton,
    SecondaryButton,
    DangerButton,
    CustomTable,
    ClickableImageLabel,
    ZoomableImageWidget,
    TrainingConfirmDialog,
)
from services.config import get_project_root, get_pretrained_model_dir, init_pretrained_model_env
from services.dataset_service import prepare_origin_dataset, create_train_dataset_split
from services.training_service import get_training_manager
from services.compare_service import CompareService
from utils.data_utils import DataPreparer


class ExportWorkerThread(QThread):
    """Background thread worker for model export with robust error handling and custom save path support"""
    success_signal = pyqtSignal(str, str) # format, saved_path
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, model_target, cfg, parent=None):
        super().__init__(parent)
        self.model_target = model_target
        self.cfg = cfg

    def run(self):
        try:
            from utils.train_utils import YOLOTrainer
            project_root = get_project_root()
            model_path = os.path.join(project_root, self.model_target)
            if not os.path.exists(model_path):
                pretrained_path = os.path.join(get_pretrained_model_dir(), os.path.basename(self.model_target))
                if os.path.exists(pretrained_path):
                    model_path = pretrained_path
                else:
                    exp_dirs = []
                    for task_dir in ["train", "detect", "segment", "classify", "pose", "obb"]:
                        exp_dirs.extend(glob.glob(os.path.join(project_root, "runs", task_dir, "*")))
                    for d in sorted(exp_dirs, key=os.path.getmtime, reverse=True):
                        weights = os.path.join(d, "weights", "best.pt")
                        if os.path.exists(weights):
                            model_path = weights
                            break

            self.log_signal.emit(f"[Export] Exporting model weights: {model_path}")
            model_stem = Path(model_path).stem
            if model_stem.endswith(".pt"):
                model_stem = model_stem[:-3]

            inferred_task = "segment"
            lower_name = model_path.lower()
            if "-cls" in lower_name:
                inferred_task = "classify"
            elif "-obb" in lower_name:
                inferred_task = "obb"
            elif "-pose" in lower_name:
                inferred_task = "pose"
            elif "-seg" in lower_name:
                inferred_task = "segment"
            elif "yolo" in lower_name:
                inferred_task = "detect"

            trainer = YOLOTrainer(model_type=model_stem, task=inferred_task, iteration_path=model_path)

            format_type = self.cfg.get("format", "onnx")
            opset = self.cfg.get("opset", 17)
            dynamic = self.cfg.get("dynamic", False)
            simplify = self.cfg.get("simplify", False)
            custom_save_path = self.cfg.get("save_path", "").strip()

            try:
                res = trainer.export(
                    format=format_type,
                    opset=opset,
                    dynamic=dynamic,
                    simplify=simplify
                )
            except BaseException as exp_err:
                err_str = str(exp_err)
                if format_type == "engine" and ("Unsupported SM" in err_str or "nullptr" in err_str or "createInferBuilder" in err_str or "pybind11" in err_str):
                    raise RuntimeError(
                        f"TensorRT export failed: Current GPU architecture does not support TensorRT 10+.\n\n"
                        f"💡 Recommendations:\n"
                        f"1. Switch export format to ONNX (*.onnx) or TorchScript (*.torchscript).\n"
                        f"2. To generate .engine format, run on an RTX 20/30/40 series GPU.\n\n"
                        f"Underlying Exception: {err_str}"
                    )
                raise exp_err

            if isinstance(res, (list, tuple)) and len(res) > 0:
                out_file = str(res[0])
            elif res:
                out_file = str(res)
            else:
                ext = ".torchscript" if format_type == "torchscript" else f".{format_type}"
                out_file = os.path.join(os.path.dirname(model_path), f"{model_stem}{ext}")

            if custom_save_path and os.path.exists(out_file):
                custom_save_path_abs = os.path.abspath(custom_save_path)
                out_file_abs = os.path.abspath(out_file)
                if out_file_abs != custom_save_path_abs:
                    os.makedirs(os.path.dirname(custom_save_path_abs), exist_ok=True)
                    if os.path.exists(custom_save_path_abs):
                        try:
                            os.remove(custom_save_path_abs)
                        except Exception:
                            pass
                    shutil.move(out_file_abs, custom_save_path_abs)
                    out_file = custom_save_path_abs

            self.success_signal.emit(format_type, out_file)
        except BaseException as e:
            err_msg = str(e) or repr(e)
            self.error_signal.emit(err_msg)


class ModelLoaderThread(QThread):
    """Background thread worker for model loading with explicit task definition & real session warm-up"""
    success_signal = pyqtSignal(object, int) # loaded_model, load_cost_ms
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, model_path, task_key="segment", device="cpu", parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.task_key = task_key
        self.device = device

    def run(self):
        try:
            import numpy as np
            from ultralytics import YOLO
            init_pretrained_model_env()

            target_path = self.model_path
            if not os.path.exists(target_path):
                p_path = os.path.join(get_pretrained_model_dir(), os.path.basename(self.model_path))
                if os.path.exists(p_path):
                    target_path = p_path

            self.log_signal.emit(f"[Model Init] Loading model architecture and warmup: {target_path} (Task={self.task_key}, Device={self.device})...")
            start_t = time.time()
            
            target_device = self.device
            # Intelligent fallback to CPU if ONNX Runtime lacks CUDA provider
            if target_path.lower().endswith(".onnx") and target_device != "cpu":
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if "CUDAExecutionProvider" not in providers and "TensorrtExecutionProvider" not in providers:
                        self.log_signal.emit(f"[Environment Notice] ONNX Runtime available providers: {providers}, auto switching to CPU mode for stability.")
                        target_device = "cpu"
                except Exception:
                    pass

            try:
                model = YOLO(target_path, task=self.task_key)
            except Exception:
                model = YOLO(target_path)

            # Triggering backend initialization and warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                model.predict(source=dummy_img, task=self.task_key, device=target_device, verbose=False)
            except BaseException as dev_err:
                err_str = str(dev_err)
                if target_device != "cpu" and ("libcudart" in err_str or "CUDA" in err_str or "onnxruntime" in err_str or "libcublas" in err_str or "FAIL" in err_str):
                    self.log_signal.emit(f"[Init Notice] Dependency issue detected, falling back to CPU mode...")
                    target_device = "cpu"
                    model.predict(source=dummy_img, task=self.task_key, device="cpu", verbose=False)
                else:
                    raise dev_err

            load_cost_ms = max(1, int((time.time() - start_t) * 1000))
            self.log_signal.emit(f"[Model Init] Completed memory residency & warmup, cost: {load_cost_ms} ms (Device: {target_device})")
            self.success_signal.emit(model, load_cost_ms)
        except BaseException as e:
            err_msg = str(e) or repr(e)
            self.error_signal.emit(err_msg)


class InferenceRunnerThread(QThread):
    """Background thread worker for executing inference on pre-loaded YOLO model with CUDA fallback & custom classes mapping"""
    success_signal = pyqtSignal(str, int, int, int) # result_image_path, preview_count, total_count, infer_cost_ms
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, model, source_path, conf=0.25, iou=0.45, imgsz=640, device="cpu", classes_file="", parent=None):
        super().__init__(parent)
        self.model = model
        self.source_path = source_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.classes_file = classes_file

    def run(self):
        try:
            import cv2
            self.log_signal.emit(f"[Inference] Executing prediction: Source={self.source_path}, Conf={self.conf}, IoU={self.iou}, ImgSz={self.imgsz}, Device={self.device}")
            
            custom_names = {}
            if self.classes_file and os.path.exists(self.classes_file):
                try:
                    with open(self.classes_file, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        custom_names = {i: name for i, name in enumerate(lines)}
                        self.log_signal.emit(f"[Classes File] Loaded custom classes ({len(custom_names)} classes): {list(custom_names.values())}")
                except Exception as ex:
                    self.log_signal.emit(f"[Classes File Warning] Failed to read classes file: {ex}")

            target_device = self.device
            model_path_str = str(getattr(self.model, "model_name", "")) or str(getattr(self.model, "ckpt_path", ""))
            if model_path_str.lower().endswith(".onnx") and target_device != "cpu":
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if "CUDAExecutionProvider" not in providers and "TensorrtExecutionProvider" not in providers:
                        target_device = "cpu"
                except Exception:
                    pass

            start_t = time.time()

            try:
                results = self.model.predict(
                    source=self.source_path,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=target_device,
                    verbose=False
                )
            except BaseException as gpu_err:
                err_str = str(gpu_err)
                if target_device != "cpu" and ("libcudart" in err_str or "CUDA" in err_str or "onnxruntime" in err_str or "libcublas" in err_str or "FAIL" in err_str):
                    self.log_signal.emit(f"[Auto Fallback Warning] Environment mismatch detected, switched to CPU mode for inference...")
                    start_t = time.time()
                    results = self.model.predict(
                        source=self.source_path,
                        conf=self.conf,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        device="cpu",
                        verbose=False
                    )
                else:
                    raise gpu_err

            infer_cost_ms = int((time.time() - start_t) * 1000)

            if not results:
                self.error_signal.emit("Inference returned no valid prediction results!")
                return

            def _count_result_items(result):
                if getattr(result, "boxes", None) is not None:
                    return len(result.boxes)
                if getattr(result, "masks", None) is not None:
                    return len(result.masks)
                if getattr(result, "probs", None) is not None:
                    return 1
                if getattr(result, "keypoints", None) is not None:
                    return len(result.keypoints)
                if getattr(result, "obb", None) is not None:
                    return len(result.obb)
                return 0

            project_root = get_project_root()
            results_dir = os.path.join(project_root, "results")
            os.makedirs(results_dir, exist_ok=True)

            preview_out_path = None
            for idx, result in enumerate(results):
                if custom_names:
                    result.names = custom_names

                plotted_bgr = result.plot()
                orig_file = os.path.basename(getattr(result, "path", ""))
                if not orig_file:
                    orig_file = f"result_{idx+1}.jpg"

                save_dst = os.path.join(results_dir, orig_file)
                cv2.imwrite(save_dst, plotted_bgr)

                # Export prediction result JSON file (Labelme format) for Mask Comparison tab
                img_stem = Path(orig_file).stem
                json_path = os.path.join(results_dir, f"{img_stem}.json")
                names_map = getattr(result, "names", custom_names or {})

                shapes = []
                if hasattr(result, "masks") and result.masks is not None and len(result.masks) > 0:
                    xy_list = result.masks.xy
                    classes_arr = result.boxes.cls.cpu().numpy()
                    confs_arr = result.boxes.conf.cpu().numpy()
                    for m_idx, pts in enumerate(xy_list):
                        if len(pts) < 3:
                            continue
                        cid = int(classes_arr[m_idx])
                        conf_val = float(confs_arr[m_idx])
                        cname = names_map.get(cid, str(cid)) if isinstance(names_map, dict) else str(cid)
                        shapes.append({
                            "label": cname,
                            "score": round(conf_val, 4),
                            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in pts],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        })
                elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                    xyxy_arr = result.boxes.xyxy.cpu().numpy()
                    classes_arr = result.boxes.cls.cpu().numpy()
                    confs_arr = result.boxes.conf.cpu().numpy()
                    for b_idx, box in enumerate(xyxy_arr):
                        x1, y1, x2, y2 = [float(v) for v in box]
                        cid = int(classes_arr[b_idx])
                        conf_val = float(confs_arr[b_idx])
                        cname = names_map.get(cid, str(cid)) if isinstance(names_map, dict) else str(cid)
                        shapes.append({
                            "label": cname,
                            "score": round(conf_val, 4),
                            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        })

                orig_shape = getattr(result, "orig_shape", (480, 640))
                json_data = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": orig_file,
                    "imageData": None,
                    "imageHeight": int(orig_shape[0]),
                    "imageWidth": int(orig_shape[1])
                }
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(json_data, jf, indent=2, ensure_ascii=False)

                if idx == 0:
                    preview_out_path = save_dst

            if not preview_out_path or not os.path.exists(preview_out_path):
                preview_out_path = os.path.join(results_dir, "result.jpg")
                cv2.imwrite(preview_out_path, results[0].plot())

            preview_count = _count_result_items(results[0])
            total_count = sum(_count_result_items(result) for result in results)
            input_kind = "Directory" if os.path.isdir(self.source_path) else "Single Image"

            self.log_signal.emit(
                f"[Inference Success] {input_kind}Inference Complete, total {len(results)} images. Results saved to results/ folder. First preview contains {preview_count} targets | Cost: {infer_cost_ms} ms"
            )

            self.success_signal.emit(preview_out_path, preview_count, total_count, infer_cost_ms)
        except BaseException as e:
            err_msg = str(e) or repr(e)
            self.error_signal.emit(err_msg)


class CompareThread(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict, str, list)
    error_signal = pyqtSignal(str)

    def __init__(self, gt_dir, infer_source, images_dir="", classes_file="", conf_thresh=0.25, parent=None):
        super().__init__(parent)
        self.gt_dir = gt_dir
        self.infer_source = infer_source
        self.images_dir = images_dir
        self.classes_file = classes_file
        self.conf_thresh = conf_thresh

    def run(self):
        try:
            def on_progress(current, total, status_text):
                self.progress_signal.emit(current, total, status_text)

            metrics, compare_dir, output_image_paths = CompareService.run_mask_comparison(
                gt_dir=self.gt_dir,
                infer_source=self.infer_source,
                images_dir=self.images_dir,
                classes_file=self.classes_file,
                conf_thresh=self.conf_thresh,
                progress_callback=on_progress,
            )
            self.finished_signal.emit(metrics, compare_dir, output_image_paths)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Comparison process encountered error: {str(e)}")


class UltralyticsWidget(QWidget):
    """Refactored Ultralytics Training & Inference Platform UI referencing X-AnyLabeling design system"""
    def __init__(self, parent=None, dataset_dir=None):
        super().__init__(parent)

        self.dataset_dir = dataset_dir or ""

        self.selected_task_type = "Segment"
        self.selected_yolo_version = "yolo11"
        self.selected_model_size = "s"

        self.class_names = ['leg', 'milkcup', 'nipple', 'tail']
        root_cls_file = Path(get_project_root()) / "classes.txt"
        if root_cls_file.exists():
            try:
                loaded = [line.strip() for line in root_cls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                if loaded:
                    self.class_names = loaded
            except Exception:
                pass
        self.task_type_buttons = {}
        self.version_buttons = {}
        self.size_buttons = {}

        self.max_unlocked_step = 1

        self.start_time = None
        self.elapsed_seconds = 0
        self.total_epochs = 1000
        self.current_epoch = 0
        self.export_thread = None

        # Model Inference state attributes
        self.cached_loaded_model = None
        self.cached_model_path = ""
        self.model_load_time_ms = 0
        self.loader_thread = None
        self.runner_thread = None

        self.training_manager = get_training_manager()
        self.training_manager.signals.log_signal.connect(self.append_training_log)
        self.training_manager.signals.event_signal.connect(self.on_training_event)

        self.image_timer = QTimer(self)
        self.image_timer.setInterval(2000)
        self.image_timer.timeout.connect(self.update_training_images)

        self.time_timer = QTimer(self)
        self.time_timer.setInterval(1000)
        self.time_timer.timeout.connect(self.update_time_display)

        self.init_ui()
        self.apply_styles()
        loaded = self.load_config()
        if not loaded and self.dataset_dir and os.path.exists(self.dataset_dir):
            self.load_dataset_info(self.dataset_dir)
        self.enforce_step_locks()

    def save_config(self):
        try:
            config_data = {
                "dataset_dir": self.dataset_path_edit.text() if hasattr(self, "dataset_path_edit") else self.dataset_dir,
                "selected_task_type": getattr(self, "selected_task_type", "Segment"),
                "selected_yolo_version": getattr(self, "selected_yolo_version", "yolo11"),
                "selected_model_size": getattr(self, "selected_model_size", "s"),
                "model_path": self.model_path_edit.text() if hasattr(self, "model_path_edit") else "",
                "exp_name": self.exp_name_edit.text() if hasattr(self, "exp_name_edit") else "exp",
                "data_yaml": self.data_yaml_edit.text() if hasattr(self, "data_yaml_edit") else "",
                "train_ratio": self.train_ratio_spin.value() if hasattr(self, "train_ratio_spin") else 0.70,
                "val_ratio": self.val_ratio_spin.value() if hasattr(self, "val_ratio_spin") else 0.20,
                "epochs": self.epochs_spin.value() if hasattr(self, "epochs_spin") else 1000,
                "batch": self.batch_spin.value() if hasattr(self, "batch_spin") else 8,
                "imgsz": self.imgsz_spin.value() if hasattr(self, "imgsz_spin") else 640,
                "workers": self.workers_spin.value() if hasattr(self, "workers_spin") else 8,
                "classes": self.classes_edit.text() if hasattr(self, "classes_edit") else "",
                "single_cls": self.single_cls_cb.isChecked() if hasattr(self, "single_cls_cb") else False,
                "patience": self.patience_spin.value() if hasattr(self, "patience_spin") else 100,
                "close_mosaic": self.close_mosaic_spin.value() if hasattr(self, "close_mosaic_spin") else 10,
                "optimizer": self.optimizer_combo.currentText() if hasattr(self, "optimizer_combo") else "auto",
                "cos_lr": self.cos_lr_cb.isChecked() if hasattr(self, "cos_lr_cb") else False,
                "amp": self.amp_cb.isChecked() if hasattr(self, "amp_cb") else True,
                "multi_scale": self.multi_scale_cb.isChecked() if hasattr(self, "multi_scale_cb") else False,
                "lr0": self.lr0_spin.value() if hasattr(self, "lr0_spin") else 0.004,
                "lrf": self.lrf_spin.value() if hasattr(self, "lrf_spin") else 0.01,
                "momentum": self.momentum_spin.value() if hasattr(self, "momentum_spin") else 0.937,
                "weight_decay": self.weight_decay_spin.value() if hasattr(self, "weight_decay_spin") else 0.0005,
                "warmup_epochs": self.warmup_epochs_spin.value() if hasattr(self, "warmup_epochs_spin") else 3.0,
                "warmup_momentum": self.warmup_mom_spin.value() if hasattr(self, "warmup_mom_spin") else 0.8,
                "warmup_bias_lr": self.warmup_bias_spin.value() if hasattr(self, "warmup_bias_spin") else 0.1,
                "hsv_h": self.hsv_h_spin.value() if hasattr(self, "hsv_h_spin") else 0.015,
                "hsv_s": self.hsv_s_spin.value() if hasattr(self, "hsv_s_spin") else 0.7,
                "hsv_v": self.hsv_v_spin.value() if hasattr(self, "hsv_v_spin") else 0.4,
                "degrees": self.degrees_spin.value() if hasattr(self, "degrees_spin") else 0.0,
                "translate": self.translate_spin.value() if hasattr(self, "translate_spin") else 0.1,
                "scale": self.scale_spin.value() if hasattr(self, "scale_spin") else 0.5,
                "shear": self.shear_spin.value() if hasattr(self, "shear_spin") else 0.0,
                "perspective": self.perspective_spin.value() if hasattr(self, "perspective_spin") else 0.0,
                "mosaic": self.mosaic_spin.value() if hasattr(self, "mosaic_spin") else 0.0,
                "copy_paste": self.copy_paste_spin.value() if hasattr(self, "copy_paste_spin") else 0.3,
                "erasing": self.erasing_spin.value() if hasattr(self, "erasing_spin") else 0.4,
                "flipud": self.flipud_spin.value() if hasattr(self, "flipud_spin") else 0.2,
                "fliplr": self.fliplr_spin.value() if hasattr(self, "fliplr_spin") else 0.5,
                "dropout": self.dropout_spin.value() if hasattr(self, "dropout_spin") else 0.0,
                "fraction": self.fraction_spin.value() if hasattr(self, "fraction_spin") else 1.0,
                "rect": self.rect_cb.isChecked() if hasattr(self, "rect_cb") else False,
                "box": self.box_spin.value() if hasattr(self, "box_spin") else 7.5,
                "cls": self.cls_spin.value() if hasattr(self, "cls_spin") else 0.5,
                "dfl": self.dfl_spin.value() if hasattr(self, "dfl_spin") else 1.5,
                "pose": self.pose_spin.value() if hasattr(self, "pose_spin") else 12.0,
                "kobj": self.kobj_spin.value() if hasattr(self, "kobj_spin") else 1.0,
                "export_model_path": self.export_model_path_edit.text() if hasattr(self, "export_model_path_edit") else "",
                "export_format": self.export_format_combo.currentText() if hasattr(self, "export_format_combo") else "onnx",
                "export_opset": self.export_opset_spin.value() if hasattr(self, "export_opset_spin") else 17,
                "export_dynamic": self.export_dynamic_cb.isChecked() if hasattr(self, "export_dynamic_cb") else False,
                "export_simplify": self.export_simplify_cb.isChecked() if hasattr(self, "export_simplify_cb") else False,
                "infer_model_path": self.infer_model_path_edit.text() if hasattr(self, "infer_model_path_edit") else "",
                "infer_conf": self.infer_conf_spin.value() if hasattr(self, "infer_conf_spin") else 0.25,
                "infer_iou": self.infer_iou_spin.value() if hasattr(self, "infer_iou_spin") else 0.45,
                "infer_imgsz": self.infer_imgsz_spin.value() if hasattr(self, "infer_imgsz_spin") else 640,
            }
            cfg_path = os.path.join(get_project_root(), "gui_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Config Save Warning] Failed to save gui_config.json: {e}")

    def load_config(self):
        cfg_path = os.path.join(get_project_root(), "gui_config.json")
        if not os.path.exists(cfg_path):
            self.dataset_dir = ""
            if hasattr(self, "dataset_path_edit"):
                self.dataset_path_edit.setText("")
            return False

        def resolve_path(p):
            if not p:
                return ""
            p_str = str(p).strip()
            if os.path.exists(p_str):
                return p_str
            abs_p = os.path.join(get_project_root(), p_str)
            if os.path.exists(abs_p):
                return abs_p
            parts = Path(p_str).parts
            for key_folder in ("data_sets", "models", "runs", "pre_trained_model"):
                if key_folder in parts:
                    idx = parts.index(key_folder)
                    cand = os.path.join(get_project_root(), *parts[idx:])
                    if os.path.exists(cand):
                        return cand
                    return cand
            return abs_p

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            saved_ds = resolve_path(cfg.get("dataset_dir", ""))
            self.dataset_dir = saved_ds
            if hasattr(self, "dataset_path_edit"):
                self.dataset_path_edit.setText(saved_ds)
                if saved_ds and os.path.exists(saved_ds):
                    self.load_dataset_info(saved_ds)

            if "selected_task_type" in cfg:
                self.on_task_type_selected(cfg["selected_task_type"])
            if "selected_yolo_version" in cfg:
                self.on_yolo_version_selected(cfg["selected_yolo_version"])
            if "selected_model_size" in cfg:
                self.on_model_size_selected(cfg["selected_model_size"])

            if "model_path" in cfg and hasattr(self, "model_path_edit") and cfg["model_path"]:
                self.model_path_edit.setText(cfg["model_path"])
            if "exp_name" in cfg and hasattr(self, "exp_name_edit"):
                self.exp_name_edit.setText(cfg["exp_name"])
            if "data_yaml" in cfg and hasattr(self, "data_yaml_edit") and cfg["data_yaml"]:
                self.data_yaml_edit.setText(resolve_path(cfg["data_yaml"]))

            if "train_ratio" in cfg and hasattr(self, "train_ratio_spin"):
                self.train_ratio_spin.setValue(float(cfg["train_ratio"]))
            if "val_ratio" in cfg and hasattr(self, "val_ratio_spin"):
                self.val_ratio_spin.setValue(float(cfg["val_ratio"]))
                self.update_test_ratio_label()

            if "epochs" in cfg and hasattr(self, "epochs_spin"):
                self.epochs_spin.setValue(int(cfg["epochs"]))
            if "batch" in cfg and hasattr(self, "batch_spin"):
                self.batch_spin.setValue(int(cfg["batch"]))
            if "imgsz" in cfg and hasattr(self, "imgsz_spin"):
                self.imgsz_spin.setValue(int(cfg["imgsz"]))
            if "workers" in cfg and hasattr(self, "workers_spin"):
                self.workers_spin.setValue(int(cfg["workers"]))
            if "classes" in cfg and hasattr(self, "classes_edit"):
                self.classes_edit.setText(str(cfg["classes"]))
            if "single_cls" in cfg and hasattr(self, "single_cls_cb"):
                self.single_cls_cb.setChecked(bool(cfg["single_cls"]))

            if "patience" in cfg and hasattr(self, "patience_spin"):
                self.patience_spin.setValue(int(cfg["patience"]))
            if "close_mosaic" in cfg and hasattr(self, "close_mosaic_spin"):
                self.close_mosaic_spin.setValue(int(cfg["close_mosaic"]))
            if "optimizer" in cfg and hasattr(self, "optimizer_combo"):
                idx = self.optimizer_combo.findText(str(cfg["optimizer"]))
                if idx >= 0:
                    self.optimizer_combo.setCurrentIndex(idx)
            if "cos_lr" in cfg and hasattr(self, "cos_lr_cb"):
                self.cos_lr_cb.setChecked(bool(cfg["cos_lr"]))
            if "amp" in cfg and hasattr(self, "amp_cb"):
                self.amp_cb.setChecked(bool(cfg["amp"]))
            if "multi_scale" in cfg and hasattr(self, "multi_scale_cb"):
                self.multi_scale_cb.setChecked(bool(cfg["multi_scale"]))

            if "lr0" in cfg and hasattr(self, "lr0_spin"):
                self.lr0_spin.setValue(float(cfg["lr0"]))
            if "lrf" in cfg and hasattr(self, "lrf_spin"):
                self.lrf_spin.setValue(float(cfg["lrf"]))
            if "momentum" in cfg and hasattr(self, "momentum_spin"):
                self.momentum_spin.setValue(float(cfg["momentum"]))
            if "weight_decay" in cfg and hasattr(self, "weight_decay_spin"):
                self.weight_decay_spin.setValue(float(cfg["weight_decay"]))
            if "warmup_epochs" in cfg and hasattr(self, "warmup_epochs_spin"):
                self.warmup_epochs_spin.setValue(float(cfg["warmup_epochs"]))
            if "warmup_momentum" in cfg and hasattr(self, "warmup_mom_spin"):
                self.warmup_mom_spin.setValue(float(cfg["warmup_momentum"]))
            if "warmup_bias_lr" in cfg and hasattr(self, "warmup_bias_spin"):
                self.warmup_bias_spin.setValue(float(cfg["warmup_bias_lr"]))

            if "hsv_h" in cfg and hasattr(self, "hsv_h_spin"):
                self.hsv_h_spin.setValue(float(cfg["hsv_h"]))
            if "hsv_s" in cfg and hasattr(self, "hsv_s_spin"):
                self.hsv_s_spin.setValue(float(cfg["hsv_s"]))
            if "hsv_v" in cfg and hasattr(self, "hsv_v_spin"):
                self.hsv_v_spin.setValue(float(cfg["hsv_v"]))
            if "degrees" in cfg and hasattr(self, "degrees_spin"):
                self.degrees_spin.setValue(float(cfg["degrees"]))
            if "translate" in cfg and hasattr(self, "translate_spin"):
                self.translate_spin.setValue(float(cfg["translate"]))
            if "scale" in cfg and hasattr(self, "scale_spin"):
                self.scale_spin.setValue(float(cfg["scale"]))
            if "shear" in cfg and hasattr(self, "shear_spin"):
                self.shear_spin.setValue(float(cfg["shear"]))
            if "perspective" in cfg and hasattr(self, "perspective_spin"):
                self.perspective_spin.setValue(float(cfg["perspective"]))
            if "mosaic" in cfg and hasattr(self, "mosaic_spin"):
                self.mosaic_spin.setValue(float(cfg["mosaic"]))
            if "copy_paste" in cfg and hasattr(self, "copy_paste_spin"):
                self.copy_paste_spin.setValue(float(cfg["copy_paste"]))
            if "erasing" in cfg and hasattr(self, "erasing_spin"):
                self.erasing_spin.setValue(float(cfg["erasing"]))
            if "flipud" in cfg and hasattr(self, "flipud_spin"):
                self.flipud_spin.setValue(float(cfg["flipud"]))
            if "fliplr" in cfg and hasattr(self, "fliplr_spin"):
                self.fliplr_spin.setValue(float(cfg["fliplr"]))
            if "dropout" in cfg and hasattr(self, "dropout_spin"):
                self.dropout_spin.setValue(float(cfg["dropout"]))
            if "fraction" in cfg and hasattr(self, "fraction_spin"):
                self.fraction_spin.setValue(float(cfg["fraction"]))
            if "rect" in cfg and hasattr(self, "rect_cb"):
                self.rect_cb.setChecked(bool(cfg["rect"]))

            if "box" in cfg and hasattr(self, "box_spin"):
                self.box_spin.setValue(float(cfg["box"]))
            if "cls" in cfg and hasattr(self, "cls_spin"):
                self.cls_spin.setValue(float(cfg["cls"]))
            if "dfl" in cfg and hasattr(self, "dfl_spin"):
                self.dfl_spin.setValue(float(cfg["dfl"]))
            if "pose" in cfg and hasattr(self, "pose_spin"):
                self.pose_spin.setValue(float(cfg["pose"]))
            if "kobj" in cfg and hasattr(self, "kobj_spin"):
                self.kobj_spin.setValue(float(cfg["kobj"]))

            if "export_model_path" in cfg and hasattr(self, "export_model_path_edit") and cfg["export_model_path"]:
                self.export_model_path_edit.setText(resolve_path(cfg["export_model_path"]))
            if "export_format" in cfg and hasattr(self, "export_format_combo"):
                idx = self.export_format_combo.findText(str(cfg["export_format"]))
                if idx >= 0:
                    self.export_format_combo.setCurrentIndex(idx)
            if "export_opset" in cfg and hasattr(self, "export_opset_spin"):
                self.export_opset_spin.setValue(int(cfg["export_opset"]))
            if "export_dynamic" in cfg and hasattr(self, "export_dynamic_cb"):
                self.export_dynamic_cb.setChecked(bool(cfg["export_dynamic"]))
            if "export_simplify" in cfg and hasattr(self, "export_simplify_cb"):
                self.export_simplify_cb.setChecked(bool(cfg["export_simplify"]))

            if "infer_model_path" in cfg and hasattr(self, "infer_model_path_edit") and cfg["infer_model_path"]:
                self.infer_model_path_edit.setText(resolve_path(cfg["infer_model_path"]))
            if "infer_conf" in cfg and hasattr(self, "infer_conf_spin"):
                self.infer_conf_spin.setValue(float(cfg["infer_conf"]))
            if "infer_iou" in cfg and hasattr(self, "infer_iou_spin"):
                self.infer_iou_spin.setValue(float(cfg["infer_iou"]))
            return True
        except Exception as e:
            print(f"[Config Load Error] {e}")
            return False

    def apply_styles(self):
        self.setStyleSheet(get_ultralytics_dialog_style())

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        t = get_theme()

        # Header Title Banner - Professional Design
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 8)

        title_lbl = QLabel("Ultralytics Training & Inference Platform")
        title_lbl.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 800;
            color: {t['primary']};
        """)

        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addStretch()

        main_layout.addWidget(header_widget)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.tabBar().setElideMode(Qt.ElideNone if hasattr(Qt, "ElideNone") else Qt.TextElideMode.ElideNone)
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {t['border']};
                border-radius: 8px;
                background-color: {t['background']};
            }}
            QTabBar::tab {{
                background: {t['surface']};
                color: {t['text_secondary']};
                border: 1px solid {t['border']};
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
                background: {t['primary']};
                color: #ffffff;
                border-color: {t['primary']};
                font-weight: 700;
            }}
            QTabBar::tab:hover:!selected {{
                background: {t['surface_hover']};
                color: {t['text']};
            }}
        """)

        self.data_tab = QWidget()
        self.config_tab = QWidget()
        self.train_tab = QWidget()
        self.export_tab = QWidget()
        self.infer_tab = QWidget()
        self.compare_tab = QWidget()

        self.tab_widget.addTab(self.data_tab, "Data")
        self.tab_widget.addTab(self.config_tab, "Config")
        self.tab_widget.addTab(self.train_tab, "Train")
        self.tab_widget.addTab(self.export_tab, "Export")
        self.tab_widget.addTab(self.infer_tab, "Inference")
        self.tab_widget.addTab(self.compare_tab, "Mask Compare")
        main_layout.addWidget(self.tab_widget, 1)

        self.init_data_tab()
        self.init_config_tab()
        self.init_train_tab()
        self.init_export_tab()
        self.init_inference_tab()
        self.init_compare_tab()
        self.enforce_step_locks()

    def enforce_step_locks(self):
        for idx in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(idx, idx < self.max_unlocked_step)

    def unlock_step(self, step_number):
        if step_number > self.max_unlocked_step:
            self.max_unlocked_step = step_number
            self.enforce_step_locks()

    # ----------------------------------------------------
    # STEP 1: DATA PREPARATION TAB
    # ----------------------------------------------------
    def init_data_tab(self):
        layout = QVBoxLayout(self.data_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        t = get_theme()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(14)

        # 1. Task Selection Box
        task_box = QGroupBox("Select Training Task Type")
        task_layout = QVBoxLayout(task_box)
        task_btn_row = QHBoxLayout()

        tasks = [
            ("Segment", "Segment"),
            ("Detect", "Detect"),
            ("OBB", "OBB"),
            ("Pose", "Pose"),
            ("Classify", "Classify"),
        ]
        for key, label in tasks:
            btn = CustomQPushButton(label)
            btn.clicked.connect(lambda _, k=key: self.on_task_type_selected(k))
            self.task_type_buttons[key] = btn
            task_btn_row.addWidget(btn)

        task_layout.addLayout(task_btn_row)

        # Pose Configuration Fields (Referencing X-AnyLabeling)
        self.pose_config_box = QWidget()
        pose_layout = QHBoxLayout(self.pose_config_box)
        pose_layout.setContentsMargins(0, 4, 0, 0)
        pose_layout.addWidget(QLabel("Pose Keypoints Shape (kpt_shape):"))
        self.kpt_shape_edit = CustomLineEdit("[17, 3]")
        self.kpt_shape_edit.setToolTip("Format: [num_keypoints, dim], e.g., human pose is [17, 3]")
        pose_layout.addWidget(self.kpt_shape_edit, 1)
        self.pose_config_box.setVisible(False)
        task_layout.addWidget(self.pose_config_box)

        scroll_layout.addWidget(task_box)

        # 2. Dataset Path Selection Box
        ds_box = QGroupBox("Dataset Path")
        ds_layout = QVBoxLayout(ds_box)
        path_row = QHBoxLayout()

        self.dataset_path_edit = CustomLineEdit("")
        self.dataset_path_edit.setPlaceholderText("Select or enter path to source images and labels directory...")
        self.browse_ds_btn = SecondaryButton("Browse...")
        self.browse_ds_btn.clicked.connect(self.browse_dataset_dir)
        self.prepare_origin_btn = SecondaryButton("Generate Origin_dataset")
        self.prepare_origin_btn.clicked.connect(self.prepare_origin_dataset_from_ui)

        path_row.addWidget(self.dataset_path_edit, 1)
        path_row.addWidget(self.browse_ds_btn)
        path_row.addWidget(self.prepare_origin_btn)
        ds_layout.addLayout(path_row)

        scroll_layout.addWidget(ds_box)

        # 3. Dataset Summary Table (Referencing X-AnyLabeling Table View)
        summary_box = QGroupBox("Dataset Overview")
        summary_layout = QVBoxLayout(summary_box)

        self.dataset_table = CustomTable()
        summary_layout.addWidget(self.dataset_table)
        scroll_layout.addWidget(summary_box, 1)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Bottom Action Bar
        action_row = QHBoxLayout()
        action_row.addStretch()
        self.next_config_btn = PrimaryButton("Next: Confirm Data -> Config")
        self.next_config_btn.clicked.connect(self.proceed_to_config)
        action_row.addWidget(self.next_config_btn)
        layout.addLayout(action_row)

        self.on_task_type_selected(self.selected_task_type)

    def on_task_type_selected(self, task_key):
        self.selected_task_type = task_key
        for k, btn in self.task_type_buttons.items():
            btn.set_selected(k == task_key)

        if hasattr(self, "pose_config_box"):
            self.pose_config_box.setVisible(task_key == "Pose")

        if hasattr(self, "dataset_path_edit") and self.dataset_path_edit.text().strip():
            self.load_dataset_info(self.dataset_path_edit.text().strip())

    def browse_dataset_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Original Dataset Directory", self.dataset_path_edit.text() or get_project_root())
        if d:
            self.dataset_path_edit.setText(d)
            self.load_dataset_info(d)

    def scan_dataset_dir(self):
        d = self.dataset_path_edit.text().strip()
        if d and os.path.exists(d):
            self.load_dataset_info(d)
        else:
            QMessageBox.warning(self, "Warning", "Specified physical path not found!")

    def load_dataset_info(self, directory):
        if not os.path.exists(directory):
            return

        self.dataset_dir = directory
        dataset_path = Path(directory)
        task_key = self.selected_task_type.lower()
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

        # Resolve classes file in dataset_path or project root
        candidates = [
            dataset_path / "classes.txt",
            dataset_path / "classes.names",
            Path(get_project_root()) / "classes.txt",
            Path(get_project_root()) / "classes.names",
            Path(get_project_root()) / "data_sets" / "Origin_dataset" / "classes.txt",
            Path(get_project_root()) / "data_sets" / "Origin_dataset" / "classes.names",
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    loaded = [line.strip() for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if loaded:
                        self.class_names = loaded
                        break
                except Exception:
                    pass

        class_names = self.class_names or ['leg', 'milkcup', 'nipple', 'tail']
        categories = {name: 0 for name in class_names}

        if task_key == "classify":
            images_dir = dataset_path / "images"
            images = []
            class_dirs = [p for p in images_dir.iterdir()] if images_dir.exists() else []
            for class_dir in class_dirs:
                if not class_dir.is_dir():
                    continue
                class_images = [
                    p for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in image_extensions
                ]
                categories[class_dir.name] = len(class_images)
                images.extend(class_images)
        else:
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"
            images = []
            if images_dir.exists():
                for ext in image_extensions:
                    images.extend(images_dir.glob(f"*{ext}"))
                    images.extend(images_dir.glob(f"*{ext.upper()}"))

            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    try:
                        with open(label_file, "r", encoding="utf-8") as fh:
                            for line in fh:
                                parts = line.strip().split()
                                if not parts:
                                    continue
                                cls_idx = int(float(parts[0]))
                                class_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"class_{cls_idx}"
                                categories[class_name] = categories.get(class_name, 0) + 1
                    except (OSError, ValueError):
                        continue

        table_data = [
            ["Class Name (Class)", "Bboxes Count", "Ratio"],
        ]
        total_count = sum(categories.values()) if categories else 0
        for name, count in categories.items():
            ratio_str = f"{count / total_count * 100:.1f}%" if total_count > 0 else "0.0%"
            table_data.append([name, count, ratio_str])

        total_ratio = "100.0%" if total_count > 0 else "0.0%"
        table_data.append(["Total", f"Images: {len(images)} | Bboxes: {total_count}", total_ratio])
        self.dataset_table.load_data(table_data)

    def prepare_origin_dataset_from_ui(self):
        source_dir = self.dataset_path_edit.text().strip()
        if not source_dir or not os.path.exists(source_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid source dataset directory first!")
            return

        try:
            origin_dir, class_names, image_count = prepare_origin_dataset(
                source_dir=source_dir,
                task_type=self.selected_task_type,
                project_root=get_project_root(),
                force=True,
            )
            self.class_names = class_names or self.class_names
            self.dataset_path_edit.setText(origin_dir)
            self.load_dataset_info(origin_dir)
            self.append_training_log(f"[Data Prep] Origin_dataset generated: {origin_dir} | Images: {image_count}")
            QMessageBox.information(self, "Completed", f"Origin_dataset generated:\n{origin_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Dataset Preparation Failed", str(e))

    def split_dataset_from_ui(self):
        project_root = get_project_root()
        origin_dir = os.path.join(project_root, "data_sets", "Origin_dataset")
        train_dir = os.path.join(project_root, "data_sets", "Train_dataset")
        if not os.path.exists(origin_dir):
            QMessageBox.warning(self, "Warning", "Please generate Origin_dataset first!")
            return

        try:
            yaml_path = create_train_dataset_split(
                origin_dataset_dir=origin_dir,
                train_dataset_dir=train_dir,
                task_type=self.selected_task_type.lower(),
                train_ratio=self.train_ratio_spin.value(),
                val_ratio=self.val_ratio_spin.value(),
                class_names=self.class_names,
            )
            self.data_yaml_edit.setText(yaml_path)
            self.append_training_log(f"[Dataset Split] Train_dataset generated: {train_dir}")
            self.append_training_log(f"[Dataset Split] Data YAML: {yaml_path}")
            QMessageBox.information(self, "Completed", f"Train_dataset generated:\n{train_dir}\n\nYAML:\n{yaml_path}")
        except Exception as e:
            QMessageBox.warning(self, "Dataset Split Failed", str(e))

    def proceed_to_config(self):
        if not self.dataset_path_edit.text().strip() or not os.path.exists(self.dataset_path_edit.text().strip()):
            QMessageBox.warning(self, "Warning", "Please select a valid original dataset directory first!")
            return
        self.unlock_step(2)
        self.tab_widget.setCurrentIndex(1)
        self.save_config()

    # ----------------------------------------------------
    # STEP 2: MODEL CONFIGURATION TAB
    # ----------------------------------------------------
    def init_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(14)

        # 1. Version & Model Scale Cards
        arch_box = QGroupBox("YOLO Architecture & Model Scale")
        arch_layout = QVBoxLayout(arch_box)

        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("YOLO Version:"))
        for ver in ["yolo11", "yolov8", "yolo26"]:
            btn = CustomQPushButton(ver.upper())
            btn.clicked.connect(lambda _, v=ver: self.on_yolo_version_selected(v))
            self.version_buttons[ver] = btn
            v_row.addWidget(btn)
        v_row.addStretch()
        arch_layout.addLayout(v_row)

        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("Model Size:"))
        sizes = [("n", "Nano (n)"), ("s", "Small (s)"), ("m", "Medium (m)"), ("l", "Large (l)"), ("x", "XLarge (x)")]
        for sz, label in sizes:
            btn = CustomQPushButton(label)
            btn.clicked.connect(lambda _, s=sz: self.on_model_size_selected(s))
            self.size_buttons[sz] = btn
            s_row.addWidget(btn)
        s_row.addStretch()
        arch_layout.addLayout(s_row)

        scroll_layout.addWidget(arch_box)

        # 2. Dataset Split Ratios
        split_box = QGroupBox("Dataset Auto Split Ratio")
        split_layout = QVBoxLayout(split_box)

        form_split = QFormLayout()
        self.train_ratio_spin = CustomDoubleSpinBox()
        self.train_ratio_spin.setRange(0.10, 0.90)
        self.train_ratio_spin.setValue(0.70)
        self.train_ratio_spin.setSingleStep(0.05)

        self.val_ratio_spin = CustomDoubleSpinBox()
        self.val_ratio_spin.setRange(0.05, 0.50)
        self.val_ratio_spin.setValue(0.20)
        self.val_ratio_spin.setSingleStep(0.05)

        self.test_ratio_label = QLabel("Test (Val/Test): 0.10 (Auto calculated: 1.0 - Train - Val)")
        self.test_ratio_label.setStyleSheet("color: #1890ff; font-weight: bold;")

        self.train_ratio_spin.valueChanged.connect(self.update_test_ratio_label)
        self.val_ratio_spin.valueChanged.connect(self.update_test_ratio_label)

        form_split.addRow("Train Split Ratio:", self.train_ratio_spin)
        form_split.addRow("Val Split Ratio:", self.val_ratio_spin)
        form_split.addRow("", self.test_ratio_label)
        split_layout.addLayout(form_split)

        split_action_row = QHBoxLayout()
        split_action_row.addStretch()
        self.split_dataset_btn = PrimaryButton("Split & Generate Train_dataset")
        self.split_dataset_btn.clicked.connect(self.split_dataset_from_ui)
        split_action_row.addWidget(self.split_dataset_btn)
        split_layout.addLayout(split_action_row)

        scroll_layout.addWidget(split_box)

        # 3. Basic Hyperparameters
        param_box = QGroupBox("Basic Training Hyperparameters")
        param_layout = QVBoxLayout(param_box)
        form_param = QFormLayout()

        self.model_path_edit = CustomLineEdit()
        self.model_path_edit.setPlaceholderText("Leave blank to auto-use weights in pre_trained_model/")
        form_param.addRow("Base Weights:", self.model_path_edit)

        self.exp_name_edit = CustomLineEdit("exp")
        form_param.addRow("Experiment Name (Exp Name):", self.exp_name_edit)

        self.data_yaml_edit = CustomLineEdit()
        self.data_yaml_edit.setPlaceholderText("Leave blank to auto-generate in data_sets/Train_dataset")
        form_param.addRow("Data YAML Path:", self.data_yaml_edit)

        self.epochs_spin = CustomSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(1000)
        form_param.addRow("Training Epochs:", self.epochs_spin)

        self.batch_spin = CustomSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)
        form_param.addRow("Batch Size:", self.batch_spin)

        self.imgsz_spin = CustomSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        form_param.addRow("Image Size (ImgSz):", self.imgsz_spin)

        self.device_combo = CustomComboBox()
        self.device_combo.addItems(["0", "cpu"])
        form_param.addRow("Hardware Device:", self.device_combo)

        self.optimizer_combo = CustomComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        form_param.addRow("Optimizer:", self.optimizer_combo)

        self.workers_spin = CustomSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(8)
        form_param.addRow("Data Load Workers:", self.workers_spin)

        self.classes_edit = CustomLineEdit("")
        self.classes_edit.setPlaceholderText("Leave blank for all classes; or enter comma-separated class IDs")
        form_param.addRow("Train Specific Classes:", self.classes_edit)

        self.single_cls_cb = CustomCheckBox("Enable Single Class Mode")
        form_param.addRow("", self.single_cls_cb)

        self.patience_spin = CustomSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(100)
        form_param.addRow("Early Stopping Patience:", self.patience_spin)

        self.close_mosaic_spin = CustomSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(10)
        form_param.addRow("Close Mosaic Last N Epochs:", self.close_mosaic_spin)

        self.amp_cb = CustomCheckBox("Enable Mixed Precision (AMP)")
        self.amp_cb.setChecked(True)
        form_param.addRow("", self.amp_cb)

        self.multi_scale_cb = CustomCheckBox("Enable Multi-Scale Training")
        form_param.addRow("", self.multi_scale_cb)

        self.cos_lr_cb = CustomCheckBox("Enable Cosine LR Scheduler")
        form_param.addRow("", self.cos_lr_cb)

        param_layout.addLayout(form_param)

        # 4. Collapsible Advanced Hyperparameters Section (Referencing X-AnyLabeling)
        self.adv_toggle_btn = SecondaryButton("➕ Expand Advanced Hyperparameters")
        self.adv_toggle_btn.clicked.connect(self.toggle_advanced_params)
        param_layout.addWidget(self.adv_toggle_btn)

        self.adv_box = QWidget()
        adv_layout = QFormLayout(self.adv_box)

        self.lr0_spin = CustomDoubleSpinBox()
        self.lr0_spin.setRange(0.0001, 1.0)
        self.lr0_spin.setValue(0.004)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setDecimals(4)
        adv_layout.addRow("Initial Learning Rate (lr0):", self.lr0_spin)

        self.lrf_spin = CustomDoubleSpinBox()
        self.lrf_spin.setRange(0.0001, 1.0)
        self.lrf_spin.setValue(0.01)
        self.lrf_spin.setSingleStep(0.001)
        self.lrf_spin.setDecimals(4)
        adv_layout.addRow("Final LR Ratio (lrf):", self.lrf_spin)

        self.momentum_spin = CustomDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setValue(0.937)
        adv_layout.addRow("Momentum:", self.momentum_spin)

        self.weight_decay_spin = CustomDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setDecimals(4)
        adv_layout.addRow("Weight Decay:", self.weight_decay_spin)

        self.warmup_epochs_spin = CustomDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0.0, 20.0)
        self.warmup_epochs_spin.setValue(3.0)
        adv_layout.addRow("Warmup Epochs:", self.warmup_epochs_spin)

        self.warmup_mom_spin = CustomDoubleSpinBox()
        self.warmup_mom_spin.setRange(0.0, 1.0)
        self.warmup_mom_spin.setValue(0.8)
        adv_layout.addRow("Warmup Initial Momentum:", self.warmup_mom_spin)

        self.warmup_bias_spin = CustomDoubleSpinBox()
        self.warmup_bias_spin.setRange(0.0, 1.0)
        self.warmup_bias_spin.setValue(0.1)
        adv_layout.addRow("Warmup Bias LR:", self.warmup_bias_spin)

        self.hsv_h_spin = CustomDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setValue(0.015)
        adv_layout.addRow("HSV-H Augmentation:", self.hsv_h_spin)

        self.hsv_s_spin = CustomDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setValue(0.7)
        adv_layout.addRow("HSV-S Augmentation:", self.hsv_s_spin)

        self.hsv_v_spin = CustomDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setValue(0.4)
        adv_layout.addRow("HSV-V Augmentation:", self.hsv_v_spin)

        self.degrees_spin = CustomDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setValue(0.0)
        adv_layout.addRow("Rotation Degrees:", self.degrees_spin)

        self.translate_spin = CustomDoubleSpinBox()
        self.translate_spin.setRange(0.0, 1.0)
        self.translate_spin.setValue(0.1)
        adv_layout.addRow("Translate Aug:", self.translate_spin)

        self.scale_spin = CustomDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1.0)
        self.scale_spin.setValue(0.5)
        adv_layout.addRow("Scale Aug:", self.scale_spin)

        self.shear_spin = CustomDoubleSpinBox()
        self.shear_spin.setRange(0.0, 180.0)
        self.shear_spin.setValue(0.0)
        adv_layout.addRow("Shear Aug:", self.shear_spin)

        self.perspective_spin = CustomDoubleSpinBox()
        self.perspective_spin.setRange(0.0, 0.001)
        self.perspective_spin.setValue(0.0)
        self.perspective_spin.setDecimals(4)
        adv_layout.addRow("Perspective Aug:", self.perspective_spin)

        self.mosaic_spin = CustomDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setValue(0.0)
        adv_layout.addRow("Mosaic Aug Ratio:", self.mosaic_spin)

        self.copy_paste_spin = CustomDoubleSpinBox()
        self.copy_paste_spin.setRange(0.0, 1.0)
        self.copy_paste_spin.setValue(0.3)
        adv_layout.addRow("Copy-Paste Aug Ratio:", self.copy_paste_spin)

        self.erasing_spin = CustomDoubleSpinBox()
        self.erasing_spin.setRange(0.0, 1.0)
        self.erasing_spin.setValue(0.4)
        adv_layout.addRow("Random Erasing Ratio:", self.erasing_spin)

        self.flipud_spin = CustomDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setValue(0.2)
        adv_layout.addRow("Flip Up-Down (FlipUD):", self.flipud_spin)

        self.fliplr_spin = CustomDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setValue(0.5)
        adv_layout.addRow("Flip Left-Right (FlipLR):", self.fliplr_spin)

        self.dropout_spin = CustomDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.0)
        adv_layout.addRow("Dropout Probability:", self.dropout_spin)

        self.fraction_spin = CustomDoubleSpinBox()
        self.fraction_spin.setRange(0.01, 1.0)
        self.fraction_spin.setValue(1.0)
        adv_layout.addRow("Dataset Fraction:", self.fraction_spin)

        self.rect_cb = CustomCheckBox("Enable Rectangular Training (Rect)")
        adv_layout.addRow("", self.rect_cb)

        self.box_spin = CustomDoubleSpinBox()
        self.box_spin.setRange(0.0, 20.0)
        self.box_spin.setValue(7.5)
        adv_layout.addRow("Box Loss Gain:", self.box_spin)

        self.cls_spin = CustomDoubleSpinBox()
        self.cls_spin.setRange(0.0, 20.0)
        self.cls_spin.setValue(0.5)
        adv_layout.addRow("Cls Loss Gain:", self.cls_spin)

        self.dfl_spin = CustomDoubleSpinBox()
        self.dfl_spin.setRange(0.0, 20.0)
        self.dfl_spin.setValue(1.5)
        adv_layout.addRow("DFL Loss Gain:", self.dfl_spin)

        self.pose_spin = CustomDoubleSpinBox()
        self.pose_spin.setRange(0.0, 30.0)
        self.pose_spin.setValue(12.0)
        adv_layout.addRow("Pose Loss Gain:", self.pose_spin)

        self.kobj_spin = CustomDoubleSpinBox()
        self.kobj_spin.setRange(0.0, 10.0)
        self.kobj_spin.setValue(1.0)
        adv_layout.addRow("Kobj Loss Gain:", self.kobj_spin)

        self.adv_box.setVisible(False)
        param_layout.addWidget(self.adv_box)

        scroll_layout.addWidget(param_box)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Bottom Action Bar
        action_row = QHBoxLayout()
        self.back_to_data_btn = SecondaryButton("<- Back: Data Preparation")
        self.back_to_data_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        self.next_train_btn = PrimaryButton("Next: Confirm Config -> Train")
        self.next_train_btn.clicked.connect(self.proceed_to_train)

        action_row.addWidget(self.back_to_data_btn)
        action_row.addStretch()
        action_row.addWidget(self.next_train_btn)
        layout.addLayout(action_row)

        self.on_yolo_version_selected(self.selected_yolo_version)
        self.on_model_size_selected(self.selected_model_size)

    def on_yolo_version_selected(self, version):
        self.selected_yolo_version = version
        for k, btn in self.version_buttons.items():
            btn.set_selected(k == version)
        self.update_default_weights_display()

    def on_model_size_selected(self, size):
        self.selected_model_size = size
        for k, btn in self.size_buttons.items():
            btn.set_selected(k == size)
        self.update_default_weights_display()

    def update_default_weights_display(self):
        if hasattr(self, "model_path_edit"):
            task_suffix = {
                "Segment": "-seg",
                "Classify": "-cls",
                "OBB": "-obb",
                "Pose": "-pose",
                "Detect": ""
            }.get(self.selected_task_type, "")
            model_name = f"{self.selected_yolo_version}{self.selected_model_size}{task_suffix}.pt"
            
            cur_text = self.model_path_edit.text().strip()
            if cur_text and not os.path.exists(cur_text):
                if re.match(r"^yolo\d*[n|s|m|l|x]?(-[a-z]+)?\.pt$", cur_text, re.IGNORECASE):
                    self.model_path_edit.setText("")

            self.model_path_edit.setPlaceholderText(f"Default path: pre_trained_model/{model_name}")

    def update_test_ratio_label(self):
        tr = self.train_ratio_spin.value()
        vr = self.val_ratio_spin.value()
        te = max(0.0, round(1.0 - tr - vr, 2))
        self.test_ratio_label.setText(f"Test Ratio: {te:.2f} (Auto calculated: 1.0 - Train - Val)")

    def toggle_advanced_params(self):
        is_visible = self.adv_box.isVisible()
        self.adv_box.setVisible(not is_visible)
        if not is_visible:
            self.adv_toggle_btn.setText("➖ Collapse Advanced Hyperparameters")
        else:
            self.adv_toggle_btn.setText("➕ Expand Advanced Hyperparameters")

    def proceed_to_train(self):
        self.unlock_step(3)
        self.tab_widget.setCurrentIndex(2)
        self.save_config()

    # ----------------------------------------------------
    # STEP 3: TRAINING & MONITORING TAB
    # ----------------------------------------------------
    def init_train_tab(self):
        layout = QVBoxLayout(self.train_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 1. Status Dashboard Header
        status_box = QGroupBox("Training Monitor & Status")
        status_layout = QVBoxLayout(status_box)

        top_info_row = QHBoxLayout()
        self.status_badge = QLabel("Status: Ready")
        self.status_badge.setStyleSheet("font-weight: bold; font-size: 14px; color: #1890ff;")

        self.epoch_info_label = QLabel("Epochs: 0 / 1000")
        self.epoch_info_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #595959;")

        self.time_info_label = QLabel("⏱ Elapsed Time: 00:00:00")
        self.time_info_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #595959;")

        top_info_row.addWidget(self.status_badge)
        top_info_row.addStretch()
        top_info_row.addWidget(self.epoch_info_label)
        top_info_row.addWidget(self.time_info_label)
        status_layout.addLayout(top_info_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                text-align: center;
                height: 22px;
                background-color: #f8fafc;
            }
            QProgressBar::chunk {
                background-color: #1890ff;
                border-radius: 5px;
            }
        """)
        status_layout.addWidget(self.progress_bar)

        # Action Buttons Row
        ctrl_row = QHBoxLayout()
        self.start_train_btn = PrimaryButton("🚀 Start Training")
        self.start_train_btn.clicked.connect(self.start_training_process)

        self.stop_train_btn = DangerButton("⏹ Force Stop")
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.clicked.connect(self.stop_training_process)

        self.export_cmd_btn = SecondaryButton("📋 Copy Command")
        self.export_cmd_btn.clicked.connect(self.export_command_line)

        ctrl_row.addWidget(self.start_train_btn)
        ctrl_row.addWidget(self.stop_train_btn)
        ctrl_row.addWidget(self.export_cmd_btn)
        ctrl_row.addStretch()
        status_layout.addLayout(ctrl_row)

        layout.addWidget(status_box)

        # 2. Main Dual Panel Split View: Logs (Left) & Plot Gallery (Right)
        main_split = QHBoxLayout()

        # Left: Training Log Console
        log_box = QGroupBox("Console Logs")
        log_layout = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #fafafa; color: #262626; font-family: monospace; font-size: 12px; border: 1px solid #d9d9d9;")
        log_layout.addWidget(self.log_text)

        log_btn_row = QHBoxLayout()
        log_btn_row.addStretch()
        self.clear_log_btn = SecondaryButton("Clear")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        self.copy_log_btn = SecondaryButton("Copy")
        self.copy_log_btn.clicked.connect(lambda: self.copy_to_clipboard(self.log_text.toPlainText()))
        self.save_log_btn = SecondaryButton("Save Log File")
        self.save_log_btn.clicked.connect(self.save_log_file)

        log_btn_row.addWidget(self.clear_log_btn)
        log_btn_row.addWidget(self.copy_log_btn)
        log_btn_row.addWidget(self.save_log_btn)
        log_layout.addLayout(log_btn_row)

        main_split.addWidget(log_box, 1)

        # Right: Real-time Plot Gallery
        gallery_box = QGroupBox("Metrics & Charts Preview (Click to Enlarge)")
        gallery_layout = QGridLayout(gallery_box)

        self.plot_labels = {}
        plot_names = [
            (["results.png"], "Results Loss & Metrics Curves"),
            (["labels.jpg", "labels.png"], "Labels Distribution Plot"),
            (["confusion_matrix.png"], "Confusion Matrix"),
            (["MaskF1_curve.png", "BoxF1_curve.png", "F1_curve.png"], "F1 Confidence Curve"),
        ]
        for idx, (filenames, desc) in enumerate(plot_names):
            lbl_box = QWidget()
            box_layout = QVBoxLayout(lbl_box)
            box_layout.setContentsMargins(2, 2, 2, 2)

            title = QLabel(desc)
            title.setStyleSheet("font-size: 11px; font-weight: bold; color: #595959;")

            img_lbl = ClickableImageLabel()
            img_lbl.setFixedSize(280, 180)
            img_lbl.setAlignment(Qt.AlignCenter if hasattr(Qt, "AlignCenter") else Qt.AlignmentFlag.AlignCenter)
            img_lbl.setStyleSheet("border:1px dashed #cbd5e1; background:#f8fafc; color:#64748b;")
            img_lbl.setText(f"Waiting for training metrics\n{desc}")

            box_layout.addWidget(title)
            box_layout.addWidget(img_lbl, 1)

            row, col = divmod(idx, 2)
            gallery_layout.addWidget(lbl_box, row, col)
            self.plot_labels[tuple(filenames)] = img_lbl

        main_split.addWidget(gallery_box, 1)
        layout.addLayout(main_split, 1)

        # Bottom Action Bar
        action_row = QHBoxLayout()
        self.back_to_config_btn = SecondaryButton("<- Back: Config")
        self.back_to_config_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))

        self.next_export_btn = PrimaryButton("Next: Go to Export")
        self.next_export_btn.clicked.connect(self.proceed_to_export)

        action_row.addWidget(self.back_to_config_btn)
        action_row.addStretch()
        action_row.addWidget(self.next_export_btn)
        layout.addLayout(action_row)

    def start_training_process(self):
        if getattr(self, "_training_starting", False) or self.training_manager.is_training:
            return

        self._training_starting = True
        self.start_train_btn.setEnabled(False)

        try:
            project_root = get_project_root()
            dataset_dir = self.dataset_path_edit.text().strip()
            output_dir = os.path.join(project_root, "data_sets", "Train_dataset")

            if not dataset_dir or not os.path.exists(dataset_dir):
                QMessageBox.warning(self, "Warning", "Specified physical dataset path not found!")
                return

            task_key = self.selected_task_type.lower()

            preparer = DataPreparer(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                tasks=[task_key],
                class_names=self.class_names
            )
            self.append_training_log(f"[Data Prep] Splitting dataset to: {output_dir}")
            preparer.split_dataset(
                train_ratio=self.train_ratio_spin.value(),
                val_ratio=self.val_ratio_spin.value(),
                force=True
            )

            yaml_path = self.data_yaml_edit.text().strip()
            fallback_yaml = os.path.join(output_dir, f"data_{task_key}.yaml")
            if not yaml_path or not os.path.exists(yaml_path):
                preparer.generate_yaml()
                yaml_path = fallback_yaml
                self.data_yaml_edit.setText(yaml_path)

            task_suffix = {
                "Segment": "-seg",
                "Classify": "-cls",
                "OBB": "-obb",
                "Pose": "-pose",
                "Detect": ""
            }.get(self.selected_task_type, "")

            raw_input = self.model_path_edit.text().strip()
            if not raw_input:
                model_name = f"{self.selected_yolo_version}{self.selected_model_size}{task_suffix}.pt"
            elif os.path.exists(raw_input):
                model_name = raw_input
            else:
                m = re.match(r"^(yolo\d+)(-[a-z]+)?(\.pt)?$", raw_input, re.IGNORECASE)
                if m:
                    ver = m.group(1)
                    suffix = m.group(2) or ""
                    ext = m.group(3) or ".pt"
                    model_name = f"{ver}{self.selected_model_size}{suffix}{ext}"
                else:
                    model_name = raw_input if raw_input.endswith(".pt") else f"{raw_input}.pt"

            cfg_summary = {
                "task": task_key,
                "yolo_version": self.selected_yolo_version,
                "model_scale": self.selected_model_size,
                "epochs": self.epochs_spin.value(),
                "batch": self.batch_spin.value(),
                "imgsz": self.imgsz_spin.value(),
                "device": self.device_combo.currentText(),
                "optimizer": self.optimizer_combo.currentText(),
                "data_yaml": yaml_path,
            }

            cmd_preview = (
                f"python train.py \\\n"
                f"  --task {task_key} \\\n"
                f"  --model {model_name} \\\n"
                f"  --epochs {self.epochs_spin.value()} \\\n"
                f"  --batch {self.batch_spin.value()} \\\n"
                f"  --imgsz {self.imgsz_spin.value()} \\\n"
                f"  --device {self.device_combo.currentText()}"
            )

            dlg = TrainingConfirmDialog(self, config=cfg_summary, cmd_str=cmd_preview)
            accepted_code = QDialog.Accepted if hasattr(QDialog, "Accepted") else QDialog.DialogCode.Accepted
            if exec_dialog(dlg) != accepted_code:
                return

            self.stop_train_btn.setEnabled(True)
            self.status_badge.setText("Status: Training Running 🚀")
            self.status_badge.setStyleSheet("font-weight: bold; font-size: 14px; color: #52c41a;")
            self.log_text.clear()
            self.append_training_log("[Training] Initializing background training process...")

            self.current_training_task = task_key
            self.current_training_start_time = time.time()
            self.current_training_run_dir = None

            # Clear and reset previous training metric plots
            for filenames, img_lbl in self.plot_labels.items():
                img_lbl.clear()
                img_lbl.image_path = None
                img_lbl.setText("Waiting for training metrics...")

            self.total_epochs = self.epochs_spin.value()
            self.progress_bar.setRange(0, self.total_epochs)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"%p% (0/{self.total_epochs})")
            self.current_epoch = 0

            self.start_time = time.time()
            self.elapsed_seconds = 0
            self.time_timer.start()
            self.image_timer.start()

            kwargs = {
                "model_type": model_name,
                "task": task_key,
                "data_yaml": yaml_path,
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "imgsz": self.imgsz_spin.value(),
                "device": self.device_combo.currentText(),
                "optimizer": self.optimizer_combo.currentText(),
                "workers": self.workers_spin.value(),
                "patience": self.patience_spin.value(),
                "close_mosaic": self.close_mosaic_spin.value(),
                "amp": self.amp_cb.isChecked(),
                "multi_scale": self.multi_scale_cb.isChecked(),
                "cos_lr": self.cos_lr_cb.isChecked(),
                "single_cls": self.single_cls_cb.isChecked(),
                "lr0": self.lr0_spin.value(),
                "lrf": self.lrf_spin.value(),
                "momentum": self.momentum_spin.value(),
                "weight_decay": self.weight_decay_spin.value(),
                "warmup_epochs": self.warmup_epochs_spin.value(),
                "warmup_momentum": self.warmup_mom_spin.value(),
                "warmup_bias_lr": self.warmup_bias_spin.value(),
                "hsv_h": self.hsv_h_spin.value(),
                "hsv_s": self.hsv_s_spin.value(),
                "hsv_v": self.hsv_v_spin.value(),
                "degrees": self.degrees_spin.value(),
                "translate": self.translate_spin.value(),
                "scale": self.scale_spin.value(),
                "shear": self.shear_spin.value(),
                "perspective": self.perspective_spin.value(),
                "dropout": self.dropout_spin.value(),
                "fraction": self.fraction_spin.value(),
                "rect": self.rect_cb.isChecked(),
                "box": self.box_spin.value(),
                "cls": self.cls_spin.value(),
                "dfl": self.dfl_spin.value(),
                "pose": self.pose_spin.value(),
                "kobj": self.kobj_spin.value(),
                "flipud": self.flipud_spin.value(),
                "fliplr": self.fliplr_spin.value(),
                "mosaic": self.mosaic_spin.value(),
                "copy_paste": self.copy_paste_spin.value(),
                "erasing": self.erasing_spin.value(),
                "model": model_name,
                "data": yaml_path,
                "name": self.exp_name_edit.text().strip() or "exp",
                "yolo_version": self.selected_yolo_version,
            }

            if self.classes_edit.text().strip():
                try:
                    classes_list = [int(c.strip()) for c in self.classes_edit.text().strip().split(",") if c.strip().isdigit()]
                    if classes_list:
                        kwargs["classes"] = classes_list
                except Exception:
                    pass

            started, message = self.training_manager.start_training(kwargs)
            if not started:
                self.append_training_log(f"[Training Start Failed] {message}")
                self.status_badge.setText("Status: Training Not Started")
                self.status_badge.setStyleSheet("font-weight: bold; font-size: 14px; color: #ff4d4f;")
                self.stop_train_btn.setEnabled(False)
                self.start_train_btn.setEnabled(True)
        finally:
            self._training_starting = False
            if not self.training_manager.is_training:
                self.start_train_btn.setEnabled(True)

    def stop_training_process(self):
        mb_yes = QMessageBox.Yes if hasattr(QMessageBox, "Yes") else QMessageBox.StandardButton.Yes
        mb_no = QMessageBox.No if hasattr(QMessageBox, "No") else QMessageBox.StandardButton.No
        if QMessageBox.question(self, "Confirm", "Are you sure you want to force terminate the current training process?", mb_yes | mb_no) == mb_yes:
            self.append_training_log("[Termination] Sending SIGKILL to stop background training process...")
            self.training_manager.stop_training()
            self.on_training_finished("terminated")

    def on_training_event(self, event_type, data):
        if event_type == "epoch_end":
            epoch = data.get("epoch", 0)
            tot_epochs = data.get("total_epochs", self.total_epochs)
            self.current_epoch = epoch
            self.total_epochs = tot_epochs
            self.progress_bar.setRange(0, self.total_epochs)
            self.progress_bar.setValue(epoch)
            self.progress_bar.setFormat(f"%p% ({epoch}/{self.total_epochs})")
            self.epoch_info_label.setText(f"Epochs: {epoch} / {self.total_epochs}")
            self.update_training_images()
        elif event_type == "training_completed":
            self.on_training_finished("completed")
        elif event_type == "training_stopped":
            self.on_training_finished("terminated")
        elif event_type == "training_error":
            err = data.get("error", "Unknown training error")
            self.append_training_log(f"[Training Error] {err}")
            self.on_training_finished("error")

    def on_training_finished(self, status):
        self.time_timer.stop()
        self.image_timer.stop()
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

        if status == "completed":
            self.status_badge.setText("Status: Training Completed")
            self.status_badge.setStyleSheet("font-weight: bold; font-size: 14px; color: #52c41a;")
            self.progress_bar.setValue(self.total_epochs)
            self.progress_bar.setFormat(f"100% ({self.total_epochs}/{self.total_epochs})")
            self.unlock_step(4)
            QMessageBox.information(self, "Complete", "Training completed! Export tab is now unlocked.")
        elif status == "terminated":
            self.status_badge.setText("Status: Training Stopped")
            self.status_badge.setStyleSheet("font-weight: bold; font-size: 14px; color: #faad14;")
        else:
            self.status_badge.setText("Status: Training Error")
            self.status_badge.setStyleSheet("font-weight: bold; font-size: 14px; color: #ff4d4f;")

        self.update_training_images()

    def update_time_display(self):
        if self.start_time:
            self.elapsed_seconds = int(time.time() - self.start_time)
            m, s = divmod(self.elapsed_seconds, 60)
            h, m = divmod(m, 60)
            self.time_info_label.setText(f"⏱ Elapsed Time: {h:02d}:{m:02d}:{s:02d}")

    def update_training_images(self):
        project_root = get_project_root()
        task_key = getattr(self, "current_training_task", None) or self.selected_task_type.lower()
        task_runs_dir = os.path.join(project_root, "runs", task_key)
        start_t = getattr(self, "current_training_start_time", 0)
        is_training = getattr(self, "training_manager", None) and self.training_manager.is_training

        latest_exp = getattr(self, "current_training_run_dir", None)
        if not latest_exp or not os.path.exists(latest_exp):
            exp_dirs = []
            if os.path.exists(task_runs_dir):
                exp_dirs = [os.path.join(task_runs_dir, d) for d in os.listdir(task_runs_dir) if os.path.isdir(os.path.join(task_runs_dir, d))]

            if not exp_dirs:
                for t in ["segment", "detect", "classify", "pose", "obb"]:
                    p = os.path.join(project_root, "runs", t)
                    if os.path.exists(p):
                        exp_dirs.extend([os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])

            if is_training and start_t > 0:
                # Filter out runs created after current training start
                new_dirs = [d for d in exp_dirs if os.path.getmtime(d) >= (start_t - 5.0)]
                if new_dirs:
                    latest_exp = max(new_dirs, key=os.path.getmtime)
                    self.current_training_run_dir = latest_exp
                else:
                    # Maintain waiting state until current run folder is created
                    return
            elif exp_dirs:
                latest_exp = max(exp_dirs, key=os.path.getmtime)

        if not latest_exp or not os.path.exists(latest_exp):
            return

        for filenames, img_lbl in self.plot_labels.items():
            matched_path = None
            for filename in filenames:
                candidate = os.path.join(latest_exp, filename)
                if os.path.exists(candidate):
                    matched_path = candidate
                    break
            if matched_path:
                img_lbl.image_path = matched_path
                pix = QPixmap(matched_path)
                if not pix.isNull():
                    keep_aspect = Qt.KeepAspectRatio if hasattr(Qt, "KeepAspectRatio") else Qt.AspectRatioMode.KeepAspectRatio
                    smooth_trans = Qt.SmoothTransformation if hasattr(Qt, "SmoothTransformation") else Qt.TransformationMode.SmoothTransformation
                    scaled = pix.scaled(280, 180, keep_aspect, smooth_trans)
                    img_lbl.setPixmap(scaled)
                    keep_aspect = Qt.KeepAspectRatio if hasattr(Qt, "KeepAspectRatio") else Qt.AspectRatioMode.KeepAspectRatio
                    smooth_trans = Qt.SmoothTransformation if hasattr(Qt, "SmoothTransformation") else Qt.TransformationMode.SmoothTransformation
                    scaled = pix.scaled(280, 180, keep_aspect, smooth_trans)
                    img_lbl.setPixmap(scaled)

    def export_command_line(self):
        task_key = self.selected_task_type.lower()
        project_root = get_project_root()
        origin_dir = os.path.join(project_root, "data_sets", "Origin_dataset")
        train_dir = os.path.join(project_root, "data_sets", "Train_dataset")
        yaml_path = self.data_yaml_edit.text().strip() or os.path.join(train_dir, f"data_{task_key}.yaml")
        task_suffix = {
            "Segment": "-seg",
            "Classify": "-cls",
            "OBB": "-obb",
            "Pose": "-pose",
            "Detect": ""
        }.get(self.selected_task_type, "")
        raw_input = self.model_path_edit.text().strip()
        if not raw_input:
            model_name = f"{self.selected_yolo_version}{self.selected_model_size}{task_suffix}.pt"
        elif os.path.exists(raw_input):
            model_name = raw_input
        else:
            m = re.match(r"^(yolo\d+)(-[a-z]+)?(\.pt)?$", raw_input, re.IGNORECASE)
            if m:
                ver = m.group(1)
                suffix = m.group(2) or ""
                ext = m.group(3) or ".pt"
                model_name = f"{ver}{self.selected_model_size}{suffix}{ext}"
            else:
                model_name = raw_input if raw_input.endswith(".pt") else f"{raw_input}.pt"
        cmd = (
            f"python prepare_origin.py --source-dir {self.dataset_path_edit.text().strip()} --task {task_key} && \\\n"
            f"python split_dataset.py --origin-dir {origin_dir} --train-dir {train_dir} --task {task_key} --train-ratio {self.train_ratio_spin.value():.2f} --val-ratio {self.val_ratio_spin.value():.2f} && \\\n"
            f"python train.py --task {task_key} --data {yaml_path} --model {model_name} --epochs {self.epochs_spin.value()} --batch {self.batch_spin.value()} --imgsz {self.imgsz_spin.value()} --device {self.device_combo.currentText()}"
        )
        self.copy_to_clipboard(cmd)
        QMessageBox.information(self, "Command Export", f"Training command copied to clipboard:\n\n{cmd}")

    def save_log_file(self):
        text = self.log_text.toPlainText()
        if not text.strip():
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Training Log", os.path.join(get_project_root(), "training_log.txt"), "Text Files (*.txt);;All Files (*)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Notice", f"Log file saved successfully to: {p}")

    def copy_to_clipboard(self, text):
        cb = QApplication.clipboard()
        if cb:
            cb.setText(text)

    def append_training_log(self, text):
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text).strip()
        if clean_text:
            self.log_text.append(clean_text)

    def proceed_to_export(self):
        self.unlock_step(4)
        self.tab_widget.setCurrentIndex(3)
        self.save_config()

    # ----------------------------------------------------
    # STEP 4: MODEL EXPORT TAB
    # ----------------------------------------------------
    def init_export_tab(self):
        layout = QVBoxLayout(self.export_tab)
        layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(14)

        export_box = QGroupBox("Model Export Console")
        form_layout = QFormLayout(export_box)

        # 1. Target Weights
        model_path_row = QHBoxLayout()
        self.export_model_path_edit = CustomLineEdit("yolo11s-seg.pt")
        self.export_model_browse_btn = SecondaryButton("Browse Weights...")
        self.export_model_browse_btn.clicked.connect(self.browse_export_model_file)
        model_path_row.addWidget(self.export_model_path_edit, 1)
        model_path_row.addWidget(self.export_model_browse_btn)
        form_layout.addRow("Weights to Export:", model_path_row)

        # 2. Target Format
        self.export_format_combo = CustomComboBox()
        self.export_format_combo.addItems(["onnx", "engine", "torchscript"])
        self.export_format_combo.currentTextChanged.connect(self.on_export_format_changed)
        form_layout.addRow("Export Format:", self.export_format_combo)

        # 3. Custom Save Path
        save_path_row = QHBoxLayout()
        default_models_dir = os.path.join(get_project_root(), "models")
        os.makedirs(default_models_dir, exist_ok=True)

        self.export_save_path_edit = CustomLineEdit(os.path.join(default_models_dir, "yolo11s-seg.onnx"))
        self.export_save_browse_btn = SecondaryButton("Browse Save Path...")
        self.export_save_browse_btn.clicked.connect(self.browse_export_save_file)
        save_path_row.addWidget(self.export_save_path_edit, 1)
        save_path_row.addWidget(self.export_save_browse_btn)
        form_layout.addRow("Save Path:", save_path_row)

        # 4. Advanced Export Configurations
        self.export_opset_spin = CustomSpinBox()
        self.export_opset_spin.setRange(11, 20)
        self.export_opset_spin.setValue(17)
        form_layout.addRow("Opset Version:", self.export_opset_spin)

        self.export_dynamic_cb = CustomCheckBox("Enable Dynamic Shape")
        form_layout.addRow("", self.export_dynamic_cb)

        self.export_simplify_cb = CustomCheckBox("Simplify ONNX Model")
        form_layout.addRow("", self.export_simplify_cb)

        scroll_layout.addWidget(export_box)

        # Action Execution Button
        btn_row = QHBoxLayout()
        self.start_export_btn = PrimaryButton("Start Model Export")
        self.start_export_btn.clicked.connect(self.start_export_process)
        btn_row.addWidget(self.start_export_btn)
        btn_row.addStretch()
        scroll_layout.addLayout(btn_row)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Bottom Action Bar
        action_row = QHBoxLayout()
        self.back_to_train_btn = SecondaryButton("<- Back: Training")
        self.back_to_train_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))

        self.next_infer_btn = PrimaryButton("Next: Go to Inference")
        self.next_infer_btn.clicked.connect(self.proceed_to_inference)

        action_row.addWidget(self.back_to_train_btn)
        action_row.addStretch()
        action_row.addWidget(self.next_infer_btn)
        layout.addLayout(action_row)

    def browse_export_model_file(self):
        project_root = get_project_root()
        task_key = getattr(self, "current_training_task", None) or self.selected_task_type.lower()
        init_dir = None

        latest_run = getattr(self, "current_training_run_dir", None)
        if latest_run and os.path.exists(os.path.join(latest_run, "weights")):
            init_dir = os.path.join(latest_run, "weights")
        else:
            task_runs_dir = os.path.join(project_root, "runs", task_key)
            exp_dirs = []
            if os.path.exists(task_runs_dir):
                exp_dirs = [os.path.join(task_runs_dir, d) for d in os.listdir(task_runs_dir) if os.path.isdir(os.path.join(task_runs_dir, d))]
            if not exp_dirs:
                for t in ["segment", "detect", "classify", "pose", "obb"]:
                    p = os.path.join(project_root, "runs", t)
                    if os.path.exists(p):
                        exp_dirs.extend([os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])
            if exp_dirs:
                latest_exp = max(exp_dirs, key=os.path.getmtime)
                weights_dir = os.path.join(latest_exp, "weights")
                if os.path.exists(weights_dir):
                    init_dir = weights_dir
                elif os.path.exists(latest_exp):
                    init_dir = latest_exp

        if not init_dir or not os.path.exists(init_dir):
            models_dir = os.path.join(project_root, "models")
            init_dir = models_dir if os.path.exists(models_dir) else get_pretrained_model_dir()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Weights (*.pt)",
            init_dir,
            "YOLO Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.export_model_path_edit.setText(file_path)
            self.on_export_format_changed(self.export_format_combo.currentText())

    def browse_export_save_file(self):
        fmt = self.export_format_combo.currentText()
        ext_map = {"onnx": "ONNX Model (*.onnx)", "engine": "TensorRT Engine (*.engine)", "torchscript": "TorchScript (*.torchscript)"}
        filter_str = f"{ext_map.get(fmt, 'All Files (*)')};;All Files (*)"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Export Model Save Path",
            self.export_save_path_edit.text(),
            filter_str
        )
        if file_path:
            self.export_save_path_edit.setText(file_path)

    def on_export_format_changed(self, fmt):
        cur_model = self.export_model_path_edit.text().strip()
        stem = Path(cur_model).stem if cur_model else "yolo11s-seg"
        if stem.endswith(".pt"):
            stem = stem[:-3]

        ext_map = {"onnx": ".onnx", "engine": ".engine", "torchscript": ".torchscript"}
        ext = ext_map.get(fmt, f".{fmt}")

        default_dir = os.path.join(get_project_root(), "models")
        os.makedirs(default_dir, exist_ok=True)
        self.export_save_path_edit.setText(os.path.join(default_dir, f"{stem}{ext}"))

    def start_export_process(self):
        model_target = self.export_model_path_edit.text().strip()
        format_type = self.export_format_combo.currentText()
        save_path = self.export_save_path_edit.text().strip()

        cfg = {
            "format": format_type,
            "opset": self.export_opset_spin.value(),
            "dynamic": self.export_dynamic_cb.isChecked(),
            "simplify": self.export_simplify_cb.isChecked(),
            "save_path": save_path,
        }

        self.start_export_btn.setEnabled(False)
        self.start_export_btn.setText("⌛ Exporting...")

        self.export_thread = ExportWorkerThread(model_target=model_target, cfg=cfg, parent=self)
        self.export_thread.log_signal.connect(self.append_training_log)
        self.export_thread.success_signal.connect(self.on_export_success)
        self.export_thread.error_signal.connect(self.on_export_error)
        self.export_thread.start()

    def on_export_success(self, fmt, out_path):
        self.start_export_btn.setEnabled(True)
        self.start_export_btn.setText("Start Model Export")
        self.unlock_step(5)
        if hasattr(self, "infer_model_path_edit"):
            self.infer_model_path_edit.setText(out_path)

        QMessageBox.information(self, "Export Success", f"Model exported successfully to {fmt.upper()} format!\n\nSave path:\n{out_path}")

    def on_export_error(self, err_msg):
        self.start_export_btn.setEnabled(True)
        self.start_export_btn.setText("Start Model Export")
        QMessageBox.warning(self, "Export Failed", f"Model export error:\n\n{err_msg}")

    def proceed_to_inference(self):
        self.unlock_step(5)
        self.tab_widget.setCurrentIndex(4)
        self.save_config()

    def proceed_to_compare(self):
        self.unlock_step(6)
        self.tab_widget.setCurrentIndex(5)
        self.save_config()

    # ----------------------------------------------------
    # STEP 5: MODEL INFERENCE TAB
    # ----------------------------------------------------
    def init_inference_tab(self):
        main_layout = QHBoxLayout(self.infer_tab)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Control Panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        param_box = QGroupBox("Inference Controls & Parameters")
        param_form = QFormLayout(param_box)

        # 1. Model Weights Path (.pt, .onnx, .engine, .torchscript)
        model_row = QHBoxLayout()
        project_root = get_project_root()
        default_models_dir = os.path.join(project_root, "models")
        default_onnx = os.path.join(default_models_dir, "yolo11s-seg.onnx")
        default_pt = os.path.join(get_pretrained_model_dir(), "yolo11s-seg.pt")

        init_infer_path = default_onnx if os.path.exists(default_onnx) else default_pt

        self.infer_model_path_edit = CustomLineEdit(init_infer_path)
        self.infer_model_path_edit.textChanged.connect(self.on_infer_model_path_changed)

        self.infer_model_browse_btn = SecondaryButton("Browse Model...")
        self.infer_model_browse_btn.clicked.connect(self.browse_infer_model_file)
        model_row.addWidget(self.infer_model_path_edit, 1)
        model_row.addWidget(self.infer_model_browse_btn)
        param_form.addRow("Model Weights:", model_row)

        # 2. Custom Classes Mapping File
        classes_row = QHBoxLayout()
        default_followed = os.path.join(os.path.dirname(project_root), "followed.txt")
        if not os.path.exists(default_followed):
            default_followed = os.path.join(project_root, "followed.txt")
        if not os.path.exists(default_followed):
            default_followed = os.path.join(project_root, "data_sets", "Origin_dataset", "classes.names")

        self.infer_classes_file_edit = CustomLineEdit(default_followed if os.path.exists(default_followed) else "")
        self.infer_classes_browse_btn = SecondaryButton("Browse Classes...")
        self.infer_classes_browse_btn.clicked.connect(self.browse_infer_classes_file)
        classes_row.addWidget(self.infer_classes_file_edit, 1)
        classes_row.addWidget(self.infer_classes_browse_btn)
        param_form.addRow("Classes File:", classes_row)

        # 3. Source Image / Directory
        source_row = QHBoxLayout()
        default_sr0007 = os.path.join(os.path.dirname(project_root), "SR0007")
        default_src = default_sr0007 if os.path.exists(default_sr0007) else project_root
        
        self.infer_source_edit = CustomLineEdit(default_src)
        self.infer_source_file_btn = SecondaryButton("Browse File...")
        self.infer_source_file_btn.clicked.connect(self.browse_infer_source_image)
        self.infer_source_dir_btn = SecondaryButton("Browse Dir...")
        self.infer_source_dir_btn.clicked.connect(self.browse_infer_source_dir)
        source_row.addWidget(self.infer_source_edit, 1)
        source_row.addWidget(self.infer_source_file_btn)
        source_row.addWidget(self.infer_source_dir_btn)
        param_form.addRow("Inference Source:", source_row)

        # 4. Confidence Threshold
        self.infer_conf_spin = CustomDoubleSpinBox()
        self.infer_conf_spin.setRange(0.01, 1.00)
        self.infer_conf_spin.setSingleStep(0.05)
        self.infer_conf_spin.setValue(0.25)
        param_form.addRow("Confidence Threshold:", self.infer_conf_spin)

        # 5. IoU / NMS Threshold
        self.infer_iou_spin = CustomDoubleSpinBox()
        self.infer_iou_spin.setRange(0.01, 1.00)
        self.infer_iou_spin.setSingleStep(0.05)
        self.infer_iou_spin.setValue(0.45)
        param_form.addRow("NMS IoU Threshold:", self.infer_iou_spin)

        # 6. Image Size
        self.infer_imgsz_spin = CustomSpinBox()
        self.infer_imgsz_spin.setRange(32, 2048)
        self.infer_imgsz_spin.setSingleStep(32)
        self.infer_imgsz_spin.setValue(640)
        param_form.addRow("Image Size (px):", self.infer_imgsz_spin)

        # 7. Computing Device
        self.infer_device_combo = CustomComboBox()
        self.infer_device_combo.addItems(["0", "cuda", "cpu"])
        self.infer_device_combo.setCurrentText("0")
        param_form.addRow("Inference Device:", self.infer_device_combo)

        left_layout.addWidget(param_box)

        # Action Button Row
        action_row = QHBoxLayout()
        self.back_to_export_btn = SecondaryButton("<- Back: Export")
        self.back_to_export_btn.clicked.connect(self.go_back_to_export)

        self.infer_load_btn = PrimaryButton("1. Load Model")
        self.infer_load_btn.setToolTip("Load model architecture and weights into memory (execute once)")
        self.infer_load_btn.clicked.connect(self.start_model_loading_process)

        self.infer_run_btn = PrimaryButton("2. Run Inference")
        self.infer_run_btn.setToolTip("Execute inference with the loaded model (can run multiple times)")
        self.infer_run_btn.setEnabled(False)
        self.infer_run_btn.clicked.connect(self.start_inference_runner_process)

        self.next_to_compare_btn = PrimaryButton("Next: Mask Compare ➔")
        self.next_to_compare_btn.clicked.connect(self.proceed_to_compare)

        action_row.addWidget(self.back_to_export_btn)
        action_row.addStretch()
        action_row.addWidget(self.infer_load_btn)
        action_row.addWidget(self.infer_run_btn)
        action_row.addWidget(self.next_to_compare_btn)
        left_layout.addLayout(action_row)

        # Log box
        log_box = QGroupBox("Inference Console Logs")
        log_layout = QVBoxLayout(log_box)
        self.infer_log_text = QTextEdit()
        self.infer_log_text.setReadOnly(True)
        self.infer_log_text.setStyleSheet("background-color: #fafafa; color: #262626; font-family: monospace; font-size: 12px; border: 1px solid #d9d9d9;")
        log_layout.addWidget(self.infer_log_text)

        infer_btn_row = QHBoxLayout()
        infer_btn_row.addStretch()
        self.clear_infer_log_btn = SecondaryButton("Clear")
        self.clear_infer_log_btn.clicked.connect(self.infer_log_text.clear)
        self.copy_infer_log_btn = SecondaryButton("Copy")
        self.copy_infer_log_btn.clicked.connect(lambda: self.copy_to_clipboard(self.infer_log_text.toPlainText()))
        self.copy_infer_cmd_btn = SecondaryButton("CopyInference Command")
        self.copy_infer_cmd_btn.clicked.connect(self.copy_inference_command)
        infer_btn_row.addWidget(self.clear_infer_log_btn)
        infer_btn_row.addWidget(self.copy_infer_log_btn)
        infer_btn_row.addWidget(self.copy_infer_cmd_btn)
        log_layout.addLayout(infer_btn_row)

        left_layout.addWidget(log_box, 1)

        # Separated 2-Step Action Buttons
        action_row = QHBoxLayout()
        self.back_to_export_btn = SecondaryButton("<- Back: Export")
        self.back_to_export_btn.clicked.connect(self.go_back_to_export)

        self.infer_load_btn = PrimaryButton("1. Load Model")
        self.infer_load_btn.setToolTip("Load model architecture and weights into memory (execute once)")
        self.infer_load_btn.clicked.connect(self.start_model_loading_process)

        self.infer_run_btn = PrimaryButton("2. Run Inference")
        self.infer_run_btn.setToolTip("Execute inference with the loaded model (can run multiple times)")
        self.infer_run_btn.setEnabled(False)
        self.infer_run_btn.clicked.connect(self.start_inference_runner_process)

        action_row.addWidget(self.back_to_export_btn)
        action_row.addStretch()
        action_row.addWidget(self.infer_load_btn)
        action_row.addWidget(self.infer_run_btn)
        left_layout.addLayout(action_row)

        main_layout.addWidget(left_widget, 1)

        # Right Panel: Zoomable Result Image Display
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        res_box = QGroupBox("Inference Result Preview")
        res_layout = QVBoxLayout(res_box)

        # Status label
        self.infer_summary_label = QLabel("Status: Waiting for model initialization (click [1. Load Model] first)")
        self.infer_summary_label.setStyleSheet("font-weight: bold; color: #1890ff; font-size: 13px; padding: 4px;")
        res_layout.addWidget(self.infer_summary_label)

        self.zoom_image_area = ZoomableImageWidget()
        res_layout.addWidget(self.zoom_image_area, 1)

        right_layout.addWidget(res_box)
        main_layout.addWidget(right_widget, 2)

    def on_infer_model_path_changed(self, text):
        self.cached_loaded_model = None
        self.infer_run_btn.setEnabled(False)
        self.infer_summary_label.setText("Status: Model configuration changed (click [1. Load Model] again)")

    def browse_infer_model_file(self):
        models_dir = os.path.join(get_project_root(), "models")
        os.makedirs(models_dir, exist_ok=True)
        init_dir = models_dir
        cur_text = self.infer_model_path_edit.text().strip()
        if cur_text and os.path.exists(cur_text):
            init_dir = os.path.dirname(cur_text) if os.path.isfile(cur_text) else cur_text

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Inference Weights (.pt, .onnx, .engine, .torchscript)",
            init_dir,
            "YOLO Models (*.pt *.onnx *.engine *.torchscript);;All Files (*)"
        )
        if file_path:
            self.infer_model_path_edit.setText(file_path)

    def browse_infer_classes_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Classes File (.txt, .names)",
            self.infer_classes_file_edit.text() or get_project_root(),
            "Classes TXT Files (*.txt *.names);;All Files (*)"
        )
        if file_path:
            self.infer_classes_file_edit.setText(file_path)

    def browse_infer_source_image(self):
        cur_text = self.infer_source_edit.text().strip()
        init_dir = cur_text if cur_text and os.path.exists(cur_text) else get_project_root()
        if os.path.isfile(init_dir):
            init_dir = os.path.dirname(init_dir)
        origin_img_dir = os.path.join(get_project_root(), "data_sets", "Origin_dataset", "images")
        if os.path.abspath(init_dir) == os.path.abspath(origin_img_dir):
            init_dir = get_project_root()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Test Image",
            init_dir,
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if file_path:
            self.infer_source_edit.setText(file_path)

    def browse_infer_source_dir(self):
        cur_text = self.infer_source_edit.text().strip()
        init_dir = cur_text if cur_text and os.path.exists(cur_text) else get_project_root()
        if os.path.isfile(init_dir):
            init_dir = os.path.dirname(init_dir)
        origin_img_dir = os.path.join(get_project_root(), "data_sets", "Origin_dataset", "images")
        if os.path.abspath(init_dir) == os.path.abspath(origin_img_dir):
            init_dir = get_project_root()

        dir_path = QFileDialog.getExistingDirectory(self, "Select Test Image Directory", init_dir)
        if dir_path:
            self.infer_source_edit.setText(dir_path)

    def go_back_to_export(self):
        self.tab_widget.setCurrentIndex(3)

    # STEP 1: Model Loading Process
    def copy_inference_command(self):
        model_path = self.infer_model_path_edit.text().strip()
        source_path = self.infer_source_edit.text().strip()
        cmd = (
            f"python predict.py --model {model_path} --source {source_path} "
            f"--conf {self.infer_conf_spin.value():.2f} --iou {self.infer_iou_spin.value():.2f} "
            f"--imgsz {self.infer_imgsz_spin.value()} --device {self.infer_device_combo.currentText()}"
        )
        self.copy_to_clipboard(cmd)
        QMessageBox.information(self, "Inference Command", f"Inference Command copied to clipboard:\n\n{cmd}")

    def start_model_loading_process(self):
        model_path = self.infer_model_path_edit.text().strip()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Specified model weight file not found: {model_path}")
            return

        task_key = {
            "Segment": "segment",
            "Detect": "detect",
            "OBB": "obb",
            "Pose": "pose",
            "Classify": "classify"
        }.get(self.selected_task_type, "segment")

        device = self.infer_device_combo.currentText()
        self.append_infer_log(f"[1. Model Init] Loading and initializing model weights: {model_path} (Task={task_key}, Device={device})...")
        self.infer_load_btn.setEnabled(False)
        self.infer_load_btn.setText("⌛ Initializing Model...")
        self.infer_run_btn.setEnabled(False)

        self.loader_thread = ModelLoaderThread(model_path=model_path, task_key=task_key, device=device, parent=self)
        self.loader_thread.log_signal.connect(self.append_infer_log)
        self.loader_thread.success_signal.connect(self.on_model_load_success)
        self.loader_thread.error_signal.connect(self.on_model_load_error)
        self.loader_thread.start()

    def on_model_load_success(self, loaded_model, load_cost_ms):
        self.cached_loaded_model = loaded_model
        self.model_load_time_ms = load_cost_ms
        self.infer_load_btn.setEnabled(True)
        self.infer_load_btn.setText("📥 1. Initialize Model")
        self.infer_run_btn.setEnabled(True)

        self.infer_summary_label.setText(
            f"✅ Model Initialized! | Init Cost: {load_cost_ms} ms (Loaded in memory/GPU, ready for [2. Run Inference])"
        )
        QMessageBox.information(self, "Model Initialized", f"Model initialized successfully and loaded in memory/GPU!\n\n• Model Initialization Cost: {load_cost_ms} ms\n• Subsequent inferences will reuse loaded model.")

    def on_model_load_error(self, err_msg):
        self.cached_loaded_model = None
        self.infer_load_btn.setEnabled(True)
        self.infer_load_btn.setText("📥 1. Initialize Model")
        self.infer_run_btn.setEnabled(False)
        self.append_infer_log(f"[Initialization Error] {err_msg}")
        QMessageBox.warning(self, "Initialization Failed", f"Failed to initialize model:\n\n{err_msg}")

    # STEP 2: Model Inference Process
    def start_inference_runner_process(self):
        if not self.cached_loaded_model:
            QMessageBox.warning(self, "Notice", "Model not initialized! Please click [1. Initialize Model] first.")
            return

        source_path = self.infer_source_edit.text().strip()
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "Warning", f"Specified test input path not found: {source_path}")
            return

        conf = self.infer_conf_spin.value()
        iou = self.infer_iou_spin.value()
        imgsz = self.infer_imgsz_spin.value()
        device = self.infer_device_combo.currentText()
        classes_file = self.infer_classes_file_edit.text().strip()

        self.append_infer_log(f"[2. Inference] Executing prediction: Source={source_path}, Conf={conf}, IoU={iou}, ImgSz={imgsz}, Device={device}")
        self.infer_run_btn.setEnabled(False)
        self.infer_run_btn.setText("⌛ Inferencing...")

        self.runner_thread = InferenceRunnerThread(
            model=self.cached_loaded_model,
            source_path=source_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            classes_file=classes_file,
            parent=self
        )
        self.runner_thread.log_signal.connect(self.append_infer_log)
        self.runner_thread.success_signal.connect(self.on_inference_runner_success)
        self.runner_thread.error_signal.connect(self.on_inference_runner_error)
        self.runner_thread.start()

    def on_inference_runner_success(self, result_image_path, preview_count, total_count, infer_cost_ms):
        self.infer_run_btn.setEnabled(True)
        self.infer_run_btn.setText("⚡ 2. Run Inference")

        total_cost_ms = self.model_load_time_ms + infer_cost_ms
        if os.path.isdir(self.infer_source_edit.text().strip()):
            summary_text = (
                f"Directory Inference Complete! First preview targets: {preview_count}  targets | "
                f"Total targets: {total_count}  targets | "
                f"📥 Load Cost: {self.model_load_time_ms} ms | "
                f"🔍 Infer Cost: {infer_cost_ms} ms | "
                f"⏱️ Total Cost: {total_cost_ms} ms"
            )
        else:
            summary_text = (
                f"Inference Complete! Detected targets: {preview_count}  targets | "
                f"📥 Load Cost: {self.model_load_time_ms} ms | "
                f"🔍 Infer Cost: {infer_cost_ms} ms | "
                f"⏱️ Total Cost: {total_cost_ms} ms"
            )

        self.infer_summary_label.setText(summary_text)
        self.zoom_image_area.set_image(result_image_path)
        QMessageBox.information(self, "Inference Complete", summary_text)

    def on_inference_runner_error(self, err_msg):
        self.infer_run_btn.setEnabled(True)
        self.infer_run_btn.setText("⚡ 2. Run Inference")
        self.append_infer_log(f"[Inference Error] {err_msg}")
        QMessageBox.warning(self, "Inference Failed", f"Model inference encountered an error:\n\n{err_msg}")

    def append_infer_log(self, text):
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text).strip()
        if clean_text:
            self.infer_log_text.append(clean_text)

    # ----------------------------------------------------
    # STEP 6: MASK COMPARE & EVALUATION TAB
    # ----------------------------------------------------
    # ----------------------------------------------------
    # STEP 6: MASK COMPARE & EVALUATION TAB
    # ----------------------------------------------------
    def init_compare_tab(self):
        main_layout = QHBoxLayout(self.compare_tab)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        t = get_theme()

        # Left Panel: Controls & Data Import
        left_widget = QWidget()
        left_widget.setFixedWidth(380)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        ctrl_box = QGroupBox("Mask Comparison & Evaluation Parameters")
        ctrl_form = QFormLayout(ctrl_box)

        # 1. Ground Truth / Label Data Path
        default_gt_dir = "/home/u22/github/codex/随动相机标注数据"
        if not os.path.exists(default_gt_dir):
            default_gt_dir = getattr(self, "current_dataset", "")

        gt_row = QHBoxLayout()
        self.compare_gt_edit = CustomLineEdit(default_gt_dir)
        self.compare_gt_browse_btn = SecondaryButton("Browse GT...")
        self.compare_gt_browse_btn.clicked.connect(self.browse_compare_gt_folder)
        gt_row.addWidget(self.compare_gt_edit, 1)
        gt_row.addWidget(self.compare_gt_browse_btn)
        ctrl_form.addRow("GT Label Directory:", gt_row)

        # 2. Inference Data / Model Path
        infer_row = QHBoxLayout()
        default_infer_path = os.path.join(get_project_root(), "results")
        self.compare_infer_edit = CustomLineEdit(default_infer_path)
        self.compare_infer_browse_btn = SecondaryButton("Browse Inference...")
        self.compare_infer_browse_btn.clicked.connect(self.browse_compare_infer_folder)
        infer_row.addWidget(self.compare_infer_edit, 1)
        infer_row.addWidget(self.compare_infer_browse_btn)
        ctrl_form.addRow("Inference Results / Model:", infer_row)

        # 3. Custom Images Path (Optional)
        img_row = QHBoxLayout()
        self.compare_images_edit = CustomLineEdit("")
        self.compare_images_browse_btn = SecondaryButton("Browse Images...")
        self.compare_images_browse_btn.clicked.connect(self.browse_compare_images_folder)
        img_row.addWidget(self.compare_images_edit, 1)
        img_row.addWidget(self.compare_images_browse_btn)
        ctrl_form.addRow("Images Directory (Opt):", img_row)

        # 4. Classes File Path
        cls_row = QHBoxLayout()
        default_cls_file = os.path.join(default_gt_dir, "classes.txt")
        if not os.path.exists(default_cls_file):
            default_cls_file = os.path.join(get_project_root(), "classes.txt")
        self.compare_cls_edit = CustomLineEdit(default_cls_file if os.path.exists(default_cls_file) else "")
        self.compare_cls_browse_btn = SecondaryButton("Browse Classes...")
        self.compare_cls_browse_btn.clicked.connect(self.browse_compare_classes_file)
        cls_row.addWidget(self.compare_cls_edit, 1)
        cls_row.addWidget(self.compare_cls_browse_btn)
        ctrl_form.addRow("Classes File (txt):", cls_row)

        # 5. Conf Threshold
        conf_row = QHBoxLayout()
        self.compare_conf_spin = CustomDoubleSpinBox()
        self.compare_conf_spin.setRange(0.01, 1.00)
        self.compare_conf_spin.setSingleStep(0.05)
        self.compare_conf_spin.setValue(0.25)
        conf_row.addWidget(self.compare_conf_spin)
        ctrl_form.addRow("Confidence Threshold:", conf_row)

        left_layout.addWidget(ctrl_box)

        self.compare_run_btn = PrimaryButton("🔍 Run Mask Comparison & Evaluation")
        self.compare_run_btn.setFixedHeight(42)
        self.compare_run_btn.clicked.connect(self.run_mask_comparison)
        left_layout.addWidget(self.compare_run_btn)

        self.compare_progress_bar = QProgressBar()
        self.compare_progress_bar.setValue(0)
        self.compare_progress_bar.setVisible(False)
        left_layout.addWidget(self.compare_progress_bar)

        self.compare_status_lbl = QLabel("Ready. Please import GT directory and Inference results folder, then click Run Comparison.")
        self.compare_status_lbl.setWordWrap(True)
        self.compare_status_lbl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
        left_layout.addWidget(self.compare_status_lbl)

        left_layout.addStretch()
        main_layout.addWidget(left_widget)

        # Right Panel: Table, Text Conclusion & Image Viewer
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # 1. Metrics Summary Table
        table_box = QGroupBox("Mask Evaluation Metrics (per-class breakdown)")
        table_layout = QVBoxLayout(table_box)

        self.compare_metrics_table = CustomTable()
        self.compare_metrics_table.setColumnCount(8)
        self.compare_metrics_table.setHorizontalHeaderLabels([
            "Class Name", "GT Count", "Pred Count", "Precision@50", "Recall@50", "mask-mAP50", "mask-mAP50-95", "Mean IoU"
        ])
        header = self.compare_metrics_table.horizontalHeader()
        if hasattr(QHeaderView, "ResizeMode") and hasattr(QHeaderView.ResizeMode, "Stretch"):
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        elif hasattr(QHeaderView, "Stretch"):
            header.setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.compare_metrics_table)
        right_layout.addWidget(table_box, 1)

        # 2. Conclusion & Summary Text Panel
        conc_box = QGroupBox("Data Conclusion & Output Information")
        conc_layout = QVBoxLayout(conc_box)

        self.compare_conclusion_text = QTextEdit()
        self.compare_conclusion_text.setReadOnly(True)
        self.compare_conclusion_text.setFixedHeight(130)
        self.compare_conclusion_text.setHtml(
            "<p style='color:#86868b;'>Click 'Run Mask Comparison & Evaluation' to compare Ground Truth vs Inference mask contours. Rendered images (<b>GT = Green, Inference = Red</b>) will be automatically saved into the <b>compare</b> directory alongside mask-mAP50 metrics summary.</p>"
        )
        conc_layout.addWidget(self.compare_conclusion_text)
        right_layout.addWidget(conc_box)

        # 3. Result Image Viewer
        viewer_box = QGroupBox("Compare Image Result Previewer (GT = Green, Inference = Red)")
        v_layout = QHBoxLayout(viewer_box)

        self.compare_image_list = QListWidget()
        self.compare_image_list.setFixedWidth(220)
        self.compare_image_list.itemClicked.connect(self.on_compare_image_selected)
        v_layout.addWidget(self.compare_image_list)

        self.compare_image_viewer = ZoomableImageWidget()
        v_layout.addWidget(self.compare_image_viewer, 1)

        right_layout.addWidget(viewer_box, 2)

        main_layout.addWidget(right_widget, 1)

    def browse_compare_gt_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Ground Truth Label Directory", self.compare_gt_edit.text() or "/home/u22/github/codex")
        if folder:
            self.compare_gt_edit.setText(folder)

    def browse_compare_infer_folder(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Inference Results Folder or Model File", self.compare_infer_edit.text() or "", "Supported Files (*.pt *.onnx *.txt *.json);;All Files (*)")
        if path:
            self.compare_infer_edit.setText(path)
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Inference Results Directory", self.compare_infer_edit.text() or "")
            if folder:
                self.compare_infer_edit.setText(folder)

    def browse_compare_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Images Directory", self.compare_images_edit.text() or "")
        if folder:
            self.compare_images_edit.setText(folder)

    def browse_compare_classes_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select classes.txt File", self.compare_cls_edit.text() or "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.compare_cls_edit.setText(file_path)

    def run_mask_comparison(self):
        gt_dir = self.compare_gt_edit.text().strip()
        infer_source = self.compare_infer_edit.text().strip()
        images_dir = self.compare_images_edit.text().strip()
        classes_file = self.compare_cls_edit.text().strip()
        conf_thresh = self.compare_conf_spin.value()

        if not gt_dir or not os.path.exists(gt_dir):
            QMessageBox.warning(self, "Path Error", "Please select a valid Ground Truth label directory!")
            return
        if not infer_source or not os.path.exists(infer_source):
            QMessageBox.warning(self, "Path Error", "Please select a valid Inference results directory or model file!")
            return

        self.compare_run_btn.setEnabled(False)
        self.compare_run_btn.setText("⏳ Calculating Comparison...")
        self.compare_progress_bar.setVisible(True)
        self.compare_progress_bar.setValue(0)
        self.compare_status_lbl.setText("Initializing comparison thread...")

        self.compare_thread = CompareThread(
            gt_dir=gt_dir,
            infer_source=infer_source,
            images_dir=images_dir,
            classes_file=classes_file,
            conf_thresh=conf_thresh,
            parent=self
        )
        self.compare_thread.progress_signal.connect(self.on_compare_progress)
        self.compare_thread.finished_signal.connect(self.on_compare_finished)
        self.compare_thread.error_signal.connect(self.on_compare_error)
        self.compare_thread.start()

    def on_compare_progress(self, current, total, status_text):
        if total > 0:
            pct = int((current / total) * 100)
            self.compare_progress_bar.setValue(pct)
        self.compare_status_lbl.setText(status_text)

    def on_compare_finished(self, metrics, compare_dir, output_image_paths):
        self.compare_run_btn.setEnabled(True)
        self.compare_run_btn.setText("🔍 Run Mask Comparison & Evaluation")
        self.compare_progress_bar.setValue(100)
        self.compare_status_lbl.setText(f"Comparison completed! Results saved to: {compare_dir}")

        # 1. Populate metrics table
        self.compare_metrics_table.setRowCount(0)
        class_names = [k for k in metrics.keys() if k != "ALL (Average)"]
        if "ALL (Average)" in metrics:
            class_names.append("ALL (Average)")

        for row_idx, cname in enumerate(class_names):
            m = metrics[cname]
            self.compare_metrics_table.insertRow(row_idx)

            is_all = (cname == "ALL (Average)")
            items = [
                QTableWidgetItem(str(cname)),
                QTableWidgetItem(str(m["gt_count"])),
                QTableWidgetItem(str(m["pred_count"])),
                QTableWidgetItem(f"{m['precision']:.4f}"),
                QTableWidgetItem(f"{m['recall']:.4f}"),
                QTableWidgetItem(f"{m['map50']:.4f}"),
                QTableWidgetItem(f"{m['map50_95']:.4f}"),
                QTableWidgetItem(f"{m['mean_iou']:.4f}"),
            ]

            t = get_theme()
            for col_idx, item in enumerate(items):
                if is_all:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setForeground(QBrush(QColor(t["primary"])))
                self.compare_metrics_table.setItem(row_idx, col_idx, item)

        # 2. Populate text conclusion
        all_m = metrics.get("ALL (Average)", {})
        summary_html = f"""
        <div style='font-family:sans-serif;'>
            <h4 style='color:#0071e3;margin:0 0 6px 0;'>📊 Mask Comparison & Metrics Evaluation Summary</h4>
            <p style='margin:2px 0;'><b>Output Directory:</b> <code style='background:#f0f0f0;padding:2px 6px;border-radius:4px;'>{compare_dir}</code> (Filenames match original images)</p>
            <p style='margin:2px 0;'><b>Drawing Rules:</b> <span style='color:#10b981;font-weight:bold;'>■ Green Contours = Ground Truth</span> &nbsp;|&nbsp; <span style='color:#ef4444;font-weight:bold;'>■ Red Contours = Inference</span></p>
            <table border='0' cellspacing='6' style='margin-top:6px;width:100%;'>
                <tr>
                    <td><b>Total GT Instances:</b> {all_m.get('gt_count', 0)}</td>
                    <td><b>Total Pred Instances:</b> {all_m.get('pred_count', 0)}</td>
                    <td><b>mask-mAP50:</b> <b style='color:#30D158;'>{all_m.get('map50', 0):.4f}</b></td>
                    <td><b>mask-mAP50-95:</b> <b style='color:#0A84FF;'>{all_m.get('map50_95', 0):.4f}</b></td>
                </tr>
                <tr>
                    <td><b>Mean Precision@50:</b> {all_m.get('precision', 0):.4f}</td>
                    <td><b>Mean Recall@50:</b> {all_m.get('recall', 0):.4f}</td>
                    <td colspan='2'><b>Mean Mask IoU:</b> {all_m.get('mean_iou', 0):.4f}</td>
                </tr>
            </table>
        </div>
        """
        self.compare_conclusion_text.setHtml(summary_html)

        # 3. Populate image list & viewer
        self.compare_image_list.clear()
        user_role = Qt.UserRole if hasattr(Qt, "UserRole") else 32
        for img_path in output_image_paths:
            fname = Path(img_path).name
            item = QListWidgetItem(fname)
            item.setData(user_role, img_path)
            self.compare_image_list.addItem(item)

        if self.compare_image_list.count() > 0:
            first_item = self.compare_image_list.item(0)
            self.compare_image_list.setCurrentItem(first_item)
            self.on_compare_image_selected(first_item)

        QMessageBox.information(
            self,
            "Mask Comparison Complete",
            f"Mask contour comparison completed successfully!\n\n"
            f"Rendered output images saved into: {compare_dir}\n"
            f"Overall mask-mAP50: {all_m.get('map50', 0):.4f}\n"
            f"Overall mask-mAP50-95: {all_m.get('map50_95', 0):.4f}"
        )

    def on_compare_error(self, err_msg):
        self.compare_run_btn.setEnabled(True)
        self.compare_run_btn.setText("🔍 Run Mask Comparison & Evaluation")
        self.compare_progress_bar.setVisible(False)
        self.compare_status_lbl.setText("Comparison analysis error.")
        QMessageBox.warning(self, "Comparison Error", f"Execution process encountered an error:\n\n{err_msg}")

    def on_compare_image_selected(self, item):
        if not item:
            return
        user_role = Qt.UserRole if hasattr(Qt, "UserRole") else 32
        img_path = item.data(user_role)
        if img_path and os.path.exists(img_path):
            self.compare_image_viewer.set_image(img_path)

