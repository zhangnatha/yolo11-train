#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Dataset Script (Step 1.2)
Standalone script for randomly splitting Origin_dataset into Train/Val/Test subsets and generating dataset YAML.
"""

import argparse
import sys
from pathlib import Path
from services.config import get_project_root
from services.dataset_service import create_train_dataset_split


def main():
    project_root = Path(get_project_root())
    default_origin = project_root / "data_sets" / "Origin_dataset"
    default_train = project_root / "data_sets" / "Train_dataset"

    parser = argparse.ArgumentParser(description="Split Origin_dataset into Train/Val/Test subsets")
    parser.add_argument("--origin-dir", type=str, default=str(default_origin), help="Path to Origin_dataset directory")
    parser.add_argument("--train-dir", type=str, default=str(default_train), help="Path to output Train_dataset directory")
    parser.add_argument("--task", type=str, default="segment", choices=["detect", "segment", "classify", "pose", "obb"], help="Vision task type")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated class names list")
    parser.add_argument("--seed", type=int, default=0, help="Random split seed")

    args = parser.parse_args()

    if args.classes:
        class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    else:
        candidates = [
            Path(args.origin_dir) / "classes.txt",
            Path(args.origin_dir) / "classes.names",
            project_root / "classes.txt",
            project_root / "classes.names",
        ]
        class_names = []
        for cand in candidates:
            if cand.exists():
                class_names = [line.strip() for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()]
                if class_names:
                    break

    yaml_path = create_train_dataset_split(
        origin_dataset_dir=args.origin_dir,
        train_dataset_dir=args.train_dir,
        task_type=args.task,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        class_names=class_names,
        seed=args.seed,
    )
    print(f"Train_dataset split generated: {args.train_dir}")
    print(f"Dataset YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
