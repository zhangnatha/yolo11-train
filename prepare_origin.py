#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Origin Dataset Script (Step 1.1)
Standalone script for organizing raw dataset images and annotations into standard Origin_dataset structure.
"""

import argparse
import sys
from services.config import get_project_root
from services.dataset_service import prepare_origin_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Origin Dataset for YOLO Training")
    parser.add_argument("--source-dir", required=True, type=str, help="Path to raw source images and labels directory")
    parser.add_argument("--task", type=str, default="segment", choices=["detect", "segment", "classify", "pose", "obb"], help="Vision task type")
    parser.add_argument("--classes-file", type=str, default="", help="Optional classes text file (classes.txt / classes.names)")
    parser.add_argument("--no-force", action="store_true", help="Do not clean existing Origin_dataset directory")
    parser.add_argument("--project-root", type=str, default="", help="Optional project root path")

    args = parser.parse_args()

    project_root = args.project_root or get_project_root()
    classes_file = args.classes_file
    if not classes_file or not os.path.exists(classes_file):
        root_classes = Path(project_root) / "classes.txt"
        if root_classes.exists():
            classes_file = str(root_classes)

    output_dir, class_names, image_count = prepare_origin_dataset(
        source_dir=args.source_dir,
        task_type=args.task.capitalize(),
        project_root=project_root,
        classes_file=classes_file,
        force=not args.no_force,
    )
    print(f"Origin_dataset generated successfully: {output_dir}")
    print(f"Total Images: {image_count}")
    print(f"Classes: {class_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
