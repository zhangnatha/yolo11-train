#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Export Script (Step 4)
Standalone script for exporting trained PyTorch weights (*.pt) into ONNX, TensorRT Engine, or TorchScript formats.
The exported model is stored exclusively at the specified --output path.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from utils.train_utils import YOLOTrainer


def main():
    parser = argparse.ArgumentParser(description="Export YOLO Weights to ONNX / TensorRT / TorchScript")
    parser.add_argument("--model", required=True, type=str, help="Path to input .pt weights model file")
    parser.add_argument("--task", type=str, default="segment", choices=["detect", "segment", "classify", "pose", "obb"], help="Vision task type")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "torchscript"], help="Export target format")
    parser.add_argument("--output", type=str, default="", help="Final save path for exported model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Enable dynamic shape/batch dimensions")
    parser.add_argument("--simplify", action="store_true", default=False, help="Simplify ONNX model structure")
    parser.add_argument("--device", type=str, default=None, help="Device for export (e.g. 'cpu' or '0')")

    args = parser.parse_args()

    model_path = str(Path(args.model).resolve())
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found: {model_path}", file=sys.stderr)
        return 1

    if not args.output:
        ext_map = {"onnx": ".onnx", "engine": ".engine", "torchscript": ".torchscript"}
        ext = ext_map.get(args.format, f".{args.format}")
        stem = Path(model_path).stem
        if stem.endswith(".pt"):
            stem = stem[:-3]
        output_path = str(Path("models").resolve() / f"{stem}{ext}")
    else:
        output_path = str(Path(args.output).resolve())

    trainer = YOLOTrainer(model_type=model_path, task=args.task, iteration_path=model_path)
    result = trainer.export(
        format=args.format,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=args.device,
    )

    out_file = str(result[0]) if isinstance(result, (list, tuple)) and len(result) > 0 else (str(result) if result else "")
    if out_file and os.path.exists(out_file):
        out_file_abs = str(Path(out_file).resolve())
        if out_file_abs != output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            shutil.move(out_file_abs, output_path)
            result = output_path
    print(f"Export completed successfully: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
