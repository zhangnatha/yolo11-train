#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Script (Step 3)
Standalone script for launching YOLO multi-task training supporting args.yaml baseline hyperparameters.
"""

import argparse
import os
import sys
from pathlib import Path

from services.config import DEFAULT_TRAINING_CONFIG, get_project_root
from utils.train_utils import YOLOTrainer


def create_parser():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--gui", action="store_true", help="Launch UI training interface")

    # Basic Training Hyperparameters
    parser.add_argument("--task", type=str, default="segment", choices=["detect", "segment", "classify", "pose", "obb"], help="Vision task type")
    parser.add_argument("--data", type=str, default="", help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default="", help="Base model weights file name or path (e.g., yolo11s-seg.pt)")
    parser.add_argument("--yolo-version", type=str, default="yolo11", help="YOLO version (yolo11, yolov8, yolo26)")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="Model size scale")
    parser.add_argument("--epochs", type=int, default=1000, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="Computing device (0, cuda, cpu)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer type (auto, SGD, Adam, AdamW, RMSProp)")
    parser.add_argument("--workers", type=int, default=8, help="Data loading worker threads")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience epochs")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in last N epochs")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument("--multi-scale", action="store_true", default=False, help="Enable multi-scale training")
    parser.add_argument("--cos-lr", action="store_true", default=False, help="Enable cosine LR scheduler")
    parser.add_argument("--single-cls", action="store_true", default=False, help="Enable single class training mode")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated class IDs to train on")

    # Advanced Hyperparameters (Aligned with args.yaml)
    parser.add_argument("--lr0", type=float, default=0.004, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate ratio")
    parser.add_argument("--momentum", type=float, default=0.937, help="Optimizer momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup initial momentum")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1, help="Warmup bias learning rate")

    # Augmentation Hyperparameters (Aligned with args.yaml)
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation fraction")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation fraction")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation fraction")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation fraction")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale fraction")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear intensity")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective intensity")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic augmentation ratio")
    parser.add_argument("--copy-paste", type=float, default=0.3, help="Copy-Paste augmentation ratio")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random Erasing ratio")
    parser.add_argument("--flipud", type=float, default=0.2, help="Probability of vertical flip")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Probability of horizontal flip")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction to use")

    # Loss Gain Hyperparameters
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain")
    parser.add_argument("--cls", type=float, default=0.5, help="Cls loss gain")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain")
    parser.add_argument("--pose", type=float, default=12.0, help="Pose loss gain")
    parser.add_argument("--kobj", type=float, default=1.0, help="Kobj loss gain")
    parser.add_argument("--rect", action="store_true", default=False, help="Enable rectangular training")

    # Run Settings
    parser.add_argument("--name", type=str, default="", help="Experiment output run name")
    parser.add_argument("--project", type=str, default="", help="Project directory to save runs")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.gui:
        from app import main as launch_gui
        launch_gui()
        return 0

    project_root = get_project_root()
    data_yaml = args.data
    if not data_yaml:
        fallback = os.path.join(project_root, "data_sets", "Train_dataset", f"data_{args.task}.yaml")
        if os.path.exists(fallback):
            data_yaml = fallback
        else:
            print(f"Error: Dataset YAML file required. Please specify --data or run split_dataset.py first.", file=sys.stderr)
            return 1

    task_suffix = {
        "segment": "-seg",
        "classify": "-cls",
        "obb": "-obb",
        "pose": "-pose",
        "detect": ""
    }.get(args.task, "")
    model_name = args.model or f"{args.yolo_version}{args.model_size}{task_suffix}.pt"
    trainer = YOLOTrainer(
        model_type=model_name,
        task=args.task,
        iteration_path=model_name,
    )

    classes_list = None
    if args.classes:
        classes_list = [int(c.strip()) for c in args.classes.split(",") if c.strip().isdigit()]

    trainer.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        workers=args.workers,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        amp=args.amp,
        multi_scale=args.multi_scale,
        cos_lr=args.cos_lr,
        single_cls=args.single_cls,
        classes=classes_list,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        flipud=args.flipud,
        fliplr=args.fliplr,
        dropout=args.dropout,
        fraction=args.fraction,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        rect=args.rect,
        name=args.name or "train",
        project=args.project,
        resume=args.resume,
    )
    print("Training completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
