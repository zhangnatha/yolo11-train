import os
import os.path as osp
import json
import yaml
import pathlib
import numpy as np
from itertools import chain
from pathlib import Path

class LabelConverter:
    def __init__(self, classes_file=None, pose_cfg_file=None):
        self.classes = []
        if classes_file and os.path.exists(classes_file):
            self.classes = self.read_lines(classes_file)
        else:
            root_classes = Path(__file__).resolve().parent.parent / "classes.txt"
            if root_classes.exists():
                self.classes = self.read_lines(str(root_classes))
        
        self.pose_classes = {}
        if pose_cfg_file and os.path.exists(pose_cfg_file):
            with open(pose_cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict) and "classes" in data:
                    self.has_visible = data.get("has_visible", True)
                    for class_name, keypoint_name in data["classes"].items():
                        self.pose_classes[class_name] = keypoint_name
                    self.classes = list(self.pose_classes.keys())

    @staticmethod
    def read_lines(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def read_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def clamp_points(self, points, width, height):
        clamped = []
        for x, y in points:
            cx = max(0.0, min(float(x), float(width)))
            cy = max(0.0, min(float(y), float(height)))
            clamped.append([cx, cy])
        return clamped

    def custom_to_yolo(
        self,
        input_file,
        output_file,
        mode="seg",
        skip_empty_files=False,
        obb_boundary_policy="skip",
    ):
        is_empty_file = True
        if not osp.exists(input_file):
            if not skip_empty_files:
                pathlib.Path(output_file).touch()
            return is_empty_file

        try:
            data = self.read_json(input_file)
        except Exception:
            if not skip_empty_files:
                pathlib.Path(output_file).touch()
            return is_empty_file

        image_width = data.get("imageWidth", 640)
        image_height = data.get("imageHeight", 640)
        image_size = np.array([[image_width, image_height]])

        shapes = data.get("shapes", [])
        if not shapes:
            if not skip_empty_files:
                pathlib.Path(output_file).touch()
            return is_empty_file

        # Auto-collect class names if self.classes is empty
        if not self.classes:
            collected = []
            for s in shapes:
                lbl = s.get("label")
                if lbl and lbl not in collected:
                    collected.append(lbl)
            self.classes = collected

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        lines = []

        for shape in shapes:
            shape_type = shape.get("shape_type", "")
            label = shape.get("label", "")
            if label not in self.classes:
                continue
            class_index = self.classes.index(label)

            # Segmentation task: polygon / rectangle
            if mode == "seg" and shape_type in ("polygon", "rectangle"):
                raw_pts = shape.get("points", [])
                if shape_type == "rectangle" and len(raw_pts) == 2:
                    (x1, y1), (x2, y2) = raw_pts
                    raw_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                
                points = np.array(self.clamp_points(raw_pts, image_width, image_height))
                if len(points) < 3:
                    continue

                norm_points = points / image_size
                pts_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in norm_points])
                lines.append(f"{class_index} {pts_str}")
                is_empty_file = False

            # Detection task: rectangle / polygon (converted to bounding box)
            elif mode == "hbb" and shape_type in ("rectangle", "polygon"):
                raw_pts = shape.get("points", [])
                if not raw_pts:
                    continue
                pts = np.array(self.clamp_points(raw_pts, image_width, image_height))
                xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
                xmax, ymax = pts[:, 0].max(), pts[:, 1].max()

                x_center = (xmin + xmax) / (2.0 * image_width)
                y_center = (ymin + ymax) / (2.0 * image_height)
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                is_empty_file = False

            # OBB task: rotation
            elif mode == "obb" and shape_type in ("rotation", "polygon"):
                raw_pts = shape.get("points", [])
                if len(raw_pts) < 4:
                    continue
                pts = self.clamp_points(raw_pts[:4], image_width, image_height)
                coords = []
                for pt in pts:
                    coords.extend([f"{pt[0] / image_width:.6f}", f"{pt[1] / image_height:.6f}"])
                lines.append(f"{class_index} " + " ".join(coords))
                is_empty_file = False

        if lines or not skip_empty_files:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

        return is_empty_file
