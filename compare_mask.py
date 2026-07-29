#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Comparison & Evaluation Script (Step 6)
Standalone CLI script for comparing Ground Truth masks/polygons against inference predictions (or running model predictions on GT images).
Calculates per-class Precision@50, Recall@50, mask-mAP50, mask-mAP50-95, Mean IoU and renders comparison images into compare/ folder.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import cv2

from services.config import get_project_root
from services.compare_service import CompareService


def create_parser():
    parser = argparse.ArgumentParser(description="Mask Comparison & Evaluation CLI Script")
    parser.add_argument("--gt-dir", required=True, type=str, help="Path to Ground Truth label directory (Labelme JSON or YOLO TXT)")
    parser.add_argument("--infer-source", required=True, type=str, help="Path to Inference results directory (TXT/JSON) or Model file (.pt, .onnx, .engine, .torchscript)")
    parser.add_argument("--images-dir", type=str, default="", help="Optional path to raw images directory")
    parser.add_argument("--classes-file", type=str, default="", help="Optional classes text file path (classes.txt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for model inference (Default: 0.25)")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save rendered comparison images (Default: compare/)")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    gt_dir = str(Path(args.gt_dir).resolve())
    infer_source = str(Path(args.infer_source).resolve())
    images_dir = str(Path(args.images_dir).resolve()) if args.images_dir else ""
    classes_file = str(Path(args.classes_file).resolve()) if args.classes_file else ""

    if not os.path.exists(gt_dir):
        print(f"Error: Ground Truth directory not found: {gt_dir}", file=sys.stderr)
        return 1
    if not os.path.exists(infer_source):
        print(f"Error: Inference results/model path not found: {infer_source}", file=sys.stderr)
        return 1

    def on_cli_progress(current, total, status_text):
        print(f"[{current}/{total}] {status_text}")

    try:
        metrics, output_dir, _ = CompareService.run_mask_comparison(
            gt_dir=gt_dir,
            infer_source=infer_source,
            images_dir=images_dir,
            classes_file=classes_file,
            conf_thresh=args.conf,
            output_dir=args.output_dir,
            progress_callback=on_cli_progress,
        )
    except Exception as e:
        print(f"Error during mask comparison: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 85)
    print("                      MASK EVALUATION METRICS SUMMARY")
    print("=" * 85)
    header = f"{'Class Name':<20} | {'GT':<5} | {'Pred':<5} | {'Prec@50':<8} | {'Rec@50':<8} | {'mAP50':<8} | {'mAP50-95':<8} | {'Mean IoU':<8}"
    print(header)
    print("-" * 85)

    class_names = [k for k in metrics.keys() if k != "ALL (Average)"]
    if "ALL (Average)" in metrics:
        class_names.append("ALL (Average)")

    for cname in class_names:
        m = metrics[cname]
        line = f"{cname:<20} | {m['gt_count']:<5} | {m['pred_count']:<5} | {m['precision']:<8.4f} | {m['recall']:<8.4f} | {m['map50']:<8.4f} | {m['map50_95']:<8.4f} | {m['mean_iou']:<8.4f}"
        if cname == "ALL (Average)":
            print("-" * 85)
            print(line)
        else:
            print(line)
    print("=" * 85)
    print(f"Rendered comparison images saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
