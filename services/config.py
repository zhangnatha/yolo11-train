import os
import multiprocessing
from pathlib import Path

def get_project_root():
    return str(Path(__file__).resolve().parent.parent)

def get_trainer_root_dir():
    return os.path.join(get_project_root(), "runs", "train")

def get_dataset_path():
    return os.path.join(get_project_root(), "data_sets")

def get_default_project_dir():
    return os.path.join(get_project_root(), "runs")

DEFAULT_WINDOW_TITLE = "Ultralytics Training Platforms 🚀"
DEFAULT_WINDOW_SIZE = (1200, 800)

TASK_TYPES = ["Classify", "Detect", "OBB", "Segment", "Pose"]
TASK_SHAPE_MAPPINGS = {
    "Classify": ["flags"],
    "Detect": ["rectangle"],
    "OBB": ["rotation"],
    "Segment": ["polygon"],
    "Pose": ["point"],
}
TASK_LABEL_MAPPINGS = {
    "Classify": "classify",
    "Detect": "hbb",
    "OBB": "obb",
    "Segment": "seg",
    "Pose": "pose",
}

MIN_LABELED_IMAGES_THRESHOLD = 5
NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)

DEFAULT_TRAINING_CONFIG = {
    "epochs": 1000,
    "batch": 8,
    "imgsz": 640,
    "workers": 8,
    "classes": "",
    "single_cls": False,
    "time": 0,
    "patience": 100,
    "close_mosaic": 10,
    "optimizer": "auto",
    "cos_lr": False,
    "amp": True,
    "multi_scale": False,
    "lr0": 0.004,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "dropout": 0.0,
    "fraction": 1.0,
    "rect": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "flipud": 0.2,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "copy_paste": 0.3,
    "erasing": 0.4,
    "save_period": -1,
    "val": True,
    "plots": True,
    "save": True,
    "resume": False,
    "cache": False,
}

OPTIMIZER_OPTIONS = [
    "auto",
    "SGD",
    "Adam",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp",
]

TRAINING_STATUS_COLORS = {
    "idle": "#6c757d",
    "training": "#7ea6f6",
    "completed": "#a6e3a1",
    "error": "#f38ba8",
}

TRAINING_STATUS_TEXTS = {
    "idle": "Ready",
    "training": "Training...",
    "completed": "Training Completed",
    "error": "Training Error",
}

def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

IS_CUDA_AVAILABLE = is_cuda_available()
DEVICE_OPTIONS = (["0", "cuda"] if IS_CUDA_AVAILABLE else []) + ["cpu"]

def get_pretrained_model_dir():
    p = os.path.join(get_project_root(), "pre_trained_model")
    os.makedirs(p, exist_ok=True)
    return p

def init_pretrained_model_env():
    p_dir = get_pretrained_model_dir()
    os.environ["YOLO_SETTINGS_WEIGHTS_DIR"] = p_dir
    os.environ["TORCH_HOME"] = p_dir
    try:
        from ultralytics import settings
        settings.update({"weights_dir": p_dir})
    except Exception:
        pass
    return p_dir

