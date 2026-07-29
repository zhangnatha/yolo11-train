#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Inference & Prediction Script (Step 5)
Standalone script for running inference on single images or whole image directories.
Rendered results are automatically saved into results/ folder named with original image filenames.
"""

import argparse
import os
import sys
from pathlib import Path

from services.config import get_project_root


def main():
    parser = argparse.ArgumentParser(description="Run Inference Prediction on Images or Folders")
    parser.add_argument("--model", required=True, type=str, help="Path to model file (.pt, .onnx, .engine, .torchscript)")
    parser.add_argument("--source", required=True, type=str, help="Path to input image file or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (Default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold (Default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (Default: 640)")
    parser.add_argument("--device", type=str, default="cpu", help="Computing device (0, cuda, cpu)")
    parser.add_argument("--classes-file", type=str, default="", help="Optional custom classes text file (followed.txt / classes.names)")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress detailed per-image log output")

    args = parser.parse_args()

    model_path = str(Path(args.model).resolve())
    source_path = str(Path(args.source).resolve())

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1
    if not os.path.exists(source_path):
        print(f"Error: Input source file/directory not found: {source_path}", file=sys.stderr)
        return 1

    try:
        import cv2
        from ultralytics import YOLO

        custom_names = {}
        if args.classes_file and os.path.exists(args.classes_file):
            try:
                with open(args.classes_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    custom_names = {i: name for i, name in enumerate(lines)}
            except Exception as ex:
                print(f"Warning: Failed to read classes file: {ex}", file=sys.stderr)

        model = YOLO(model_path)
        if hasattr(model, "model") and hasattr(model.model, "names") and custom_names:
            model.model.names = custom_names

        results = model.predict(
            source=source_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=not args.quiet,
        )

        project_root = get_project_root()
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)

        saved_files = []
        for idx, res in enumerate(results):
            if custom_names:
                res.names = custom_names

            plotted_bgr = res.plot()
            orig_file = os.path.basename(getattr(res, "path", ""))
            if not orig_file:
                orig_file = f"result_{idx+1}.jpg"

            dst_path = os.path.join(results_dir, orig_file)
            cv2.imwrite(dst_path, plotted_bgr)
            saved_files.append(dst_path)

        print(f"Inference completed successfully. Total processed images: {len(results)}")
        print(f"All rendered result images auto-saved into: {results_dir}")
        return 0

    except Exception as e:
        print(f"Inference Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
