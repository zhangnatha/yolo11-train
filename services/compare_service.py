# services/compare_service.py

import os
import json
import glob
from pathlib import Path
import cv2
import numpy as np


class CompareService:
    def __init__(self, classes=None):
        self.classes = classes or []

    @staticmethod
    def read_classes_file(classes_file):
        if not classes_file or not os.path.exists(classes_file):
            return []
        with open(classes_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def extract_gt_objects(self, image_path, gt_dir, img_w, img_h):
        """
        Extract ground truth polygons/masks for a given image.
        Supports Labelme JSON, YOLO segmentation TXT, YOLO box TXT.
        """
        img_name = Path(image_path).stem
        gt_objs = []

        # 1. Search for Labelme JSON file
        json_candidates = [
            os.path.join(gt_dir, f"{img_name}.json"),
            os.path.join(gt_dir, "followed_camera", f"{img_name}.json"),
        ]
        json_file = None
        for cand in json_candidates:
            if os.path.exists(cand):
                json_file = cand
                break
        if not json_file:
            # Fallback recursive search
            matches = list(Path(gt_dir).rglob(f"{img_name}.json"))
            if matches:
                json_file = str(matches[0])

        if json_file and os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = shape.get("label", "")
                    pts = shape.get("points", [])
                    stype = shape.get("shape_type", "polygon")
                    if not label or not pts:
                        continue
                    if label not in self.classes:
                        self.classes.append(label)

                    if stype == "rectangle" and len(pts) == 2:
                        (x1, y1), (x2, y2) = pts
                        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    else:
                        poly = np.array(pts, dtype=np.int32)

                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 1)

                    gt_objs.append({
                        "class_name": label,
                        "class_id": self.classes.index(label),
                        "polygon": poly,
                        "mask": mask,
                    })
                return gt_objs
            except Exception as ex:
                print(f"[CompareService] Error reading JSON {json_file}: {ex}")

        # 2. Search for YOLO TXT file
        txt_candidates = [
            os.path.join(gt_dir, f"{img_name}.txt"),
            os.path.join(gt_dir, "labels", f"{img_name}.txt"),
        ]
        txt_file = None
        for cand in txt_candidates:
            if os.path.exists(cand):
                txt_file = cand
                break
        if not txt_file:
            matches = list(Path(gt_dir).rglob(f"{img_name}.txt"))
            if matches:
                txt_file = str(matches[0])

        if txt_file and os.path.exists(txt_file):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cname = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
                    coords = [float(x) for x in parts[1:]]

                    if len(coords) > 4:
                        # Polygon segmentation: x1, y1, x2, y2, ...
                        pts = []
                        for i in range(0, len(coords), 2):
                            px = int(coords[i] * img_w)
                            py = int(coords[i+1] * img_h)
                            pts.append([px, py])
                        poly = np.array(pts, dtype=np.int32)
                    else:
                        # Bounding box: cx, cy, w, h
                        cx, cy, w, h = coords
                        x1 = int((cx - w/2) * img_w)
                        y1 = int((cy - h/2) * img_h)
                        x2 = int((cx + w/2) * img_w)
                        y2 = int((cy + h/2) * img_h)
                        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 1)

                    gt_objs.append({
                        "class_name": cname,
                        "class_id": cid,
                        "polygon": poly,
                        "mask": mask,
                    })
                return gt_objs
            except Exception as ex:
                print(f"[CompareService] Error reading TXT {txt_file}: {ex}")

        return gt_objs

    def extract_pred_objects(self, image_path, pred_source, img_w, img_h, model_instance=None, conf_thresh=0.25):
        """
        Extract predicted polygons/masks for a given image.
        Can load from model_instance OR pred_source directory (.txt / .json).
        """
        pred_objs = []
        img_name = Path(image_path).stem

        # Case A: Live model inference
        if model_instance is not None:
            try:
                results = model_instance.predict(image_path, conf=conf_thresh, verbose=False)
                if results and len(results) > 0:
                    res = results[0]
                    names_dict = getattr(res, "names", {})
                    if hasattr(res, "masks") and res.masks is not None and len(res.masks) > 0:
                        xy_list = res.masks.xy
                        classes = res.boxes.cls.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()

                        for idx, poly_pts in enumerate(xy_list):
                            if len(poly_pts) < 3:
                                continue
                            cid = int(classes[idx])
                            conf = float(confs[idx])
                            cname = names_dict.get(cid, self.classes[cid] if cid < len(self.classes) else f"class_{cid}")
                            if cname not in self.classes:
                                self.classes.append(cname)

                            poly = np.array(poly_pts, dtype=np.int32)
                            mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            cv2.fillPoly(mask, [poly], 1)

                            pred_objs.append({
                                "class_name": cname,
                                "class_id": cid,
                                "polygon": poly,
                                "mask": mask,
                                "score": conf,
                            })
                    elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                        xyxy_list = res.boxes.xyxy.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()

                        for idx, box in enumerate(xyxy_list):
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cid = int(classes[idx])
                            conf = float(confs[idx])
                            cname = names_dict.get(cid, self.classes[cid] if cid < len(self.classes) else f"class_{cid}")
                            if cname not in self.classes:
                                self.classes.append(cname)

                            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                            mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            cv2.fillPoly(mask, [poly], 1)

                            pred_objs.append({
                                "class_name": cname,
                                "class_id": cid,
                                "polygon": poly,
                                "mask": mask,
                                "score": conf,
                            })
                return pred_objs
            except Exception as ex:
                print(f"[CompareService] Error during model predict on {image_path}: {ex}")

        # Case B: Read from directory of predictions (.txt or .json)
        if pred_source and os.path.isdir(pred_source):
            # Check JSON
            json_file = os.path.join(pred_source, f"{img_name}.json")
            if not os.path.exists(json_file):
                matches = list(Path(pred_source).rglob(f"{img_name}.json"))
                if matches:
                    json_file = str(matches[0])

            if os.path.exists(json_file):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for shape in data.get("shapes", []):
                        label = shape.get("label", "")
                        pts = shape.get("points", [])
                        score = shape.get("score", 1.0)
                        if score is None:
                            score = 1.0
                        if not label or not pts or float(score) < conf_thresh:
                            continue
                        poly = np.array(pts, dtype=np.int32)
                        mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly], 1)
                        pred_objs.append({
                            "class_name": label,
                            "class_id": self.classes.index(label) if label in self.classes else 0,
                            "polygon": poly,
                            "mask": mask,
                            "score": float(score),
                        })
                    return pred_objs
                except Exception:
                    pass

            # Check TXT
            txt_file = os.path.join(pred_source, f"{img_name}.txt")
            if not os.path.exists(txt_file):
                matches = list(Path(pred_source).rglob(f"{img_name}.txt"))
                if matches:
                    txt_file = str(matches[0])

            if os.path.exists(txt_file):
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    for line in lines:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(parts[0])
                        cname = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
                        coords = [float(x) for x in parts[1:]]
                        score = 1.0
                        if len(coords) % 2 == 1:
                            score = coords.pop()

                        if len(coords) > 4:
                            pts = []
                            for i in range(0, len(coords), 2):
                                px = int(coords[i] * img_w) if coords[i] <= 1.0 else int(coords[i])
                                py = int(coords[i+1] * img_h) if coords[i+1] <= 1.0 else int(coords[i+1])
                                pts.append([px, py])
                            poly = np.array(pts, dtype=np.int32)
                        else:
                            cx, cy, w, h = coords
                            x1 = int((cx - w/2) * img_w) if cx <= 1.0 else int(cx - w/2)
                            y1 = int((cy - h/2) * img_h) if cy <= 1.0 else int(cy - h/2)
                            x2 = int((cx + w/2) * img_w) if w <= 1.0 else int(cx + w/2)
                            y2 = int((cy + h/2) * img_h) if h <= 1.0 else int(cy + h/2)
                            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

                        if score < conf_thresh:
                            continue

                        mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly], 1)

                        pred_objs.append({
                            "class_name": cname,
                            "class_id": cid,
                            "polygon": poly,
                            "mask": mask,
                            "score": float(score),
                        })
                    return pred_objs
                except Exception as ex:
                    print(f"[CompareService] Error reading TXT {txt_file}: {ex}")

        return pred_objs

    @staticmethod
    def compute_mask_iou(mask1, mask2):
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def draw_comparison_image(self, image_path, gt_objs, pred_objs, output_path):
        """
        Draw comparison contours on image:
        - Ground Truth (真值): GREEN (0, 255, 0)
        - Inference (推理): RED (0, 0, 255)
        """
        img = cv2.imread(image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        overlay_gt = img.copy()
        overlay_pred = img.copy()

        # 1. Draw GT (Green)
        for obj in gt_objs:
            poly = obj["polygon"]
            cv2.fillPoly(overlay_gt, [poly], (0, 255, 0))
            cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            if len(poly) > 0:
                lx, ly = poly[0]
                cv2.putText(img, f"GT:{obj['class_name']}", (int(lx), max(15, int(ly) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # 2. Draw Pred (Red)
        for obj in pred_objs:
            poly = obj["polygon"]
            cv2.fillPoly(overlay_pred, [poly], (0, 0, 255))
            cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
            if len(poly) > 0:
                lx, ly = poly[0]
                score_str = f" {obj['score']:.2f}" if "score" in obj else ""
                cv2.putText(img, f"Pred:{obj['class_name']}{score_str}", (int(lx), min(h - 5, int(ly) + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        alpha = 0.20
        img = cv2.addWeighted(overlay_gt, alpha, img, 1 - alpha, 0)
        img = cv2.addWeighted(overlay_pred, alpha, img, 1 - alpha, 0)

        # 3. Legend Box at top-left
        legend_w, legend_h = 240, 50
        cv2.rectangle(img, (10, 10), (10 + legend_w, 10 + legend_h), (30, 30, 30), -1)
        cv2.rectangle(img, (10, 10), (10 + legend_w, 10 + legend_h), (200, 200, 200), 1)

        cv2.rectangle(img, (20, 20), (35, 30), (0, 255, 0), -1)
        cv2.putText(img, "GT (Ground Truth)", (42, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(img, (20, 38), (35, 48), (0, 0, 255), -1)
        cv2.putText(img, "Inference (Prediction)", (42, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True

    def calculate_metrics(self, all_gt_data, all_pred_data):
        """
        Calculate per-class mask-mAP50, mask-mAP50-95, Precision, Recall, Mean IoU metrics.
        """
        all_classes = sorted(list(set(self.classes)))
        if not all_classes:
            all_classes = ["object"]

        class_metrics = {}
        iou_thresholds = np.linspace(0.50, 0.95, 10)

        total_gt_count_all = 0
        total_pred_count_all = 0
        sum_map50_all = 0.0
        sum_map50_95_all = 0.0
        sum_precision_all = 0.0
        sum_recall_all = 0.0
        sum_iou_all = 0.0
        valid_classes_count = 0

        for cname in all_classes:
            gt_instances = []
            pred_instances = []

            for img_idx in range(len(all_gt_data)):
                img_gt = [o for o in all_gt_data[img_idx] if o["class_name"] == cname]
                img_pred = [o for o in all_pred_data[img_idx] if o["class_name"] == cname]
                gt_instances.append(img_gt)
                pred_instances.append(img_pred)

            n_gt = sum(len(gts) for gts in gt_instances)
            n_pred = sum(len(preds) for preds in pred_instances)

            total_gt_count_all += n_gt
            total_pred_count_all += n_pred

            if n_gt == 0 and n_pred == 0:
                class_metrics[cname] = {
                    "gt_count": 0,
                    "pred_count": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "map50": 0.0,
                    "map50_95": 0.0,
                    "mean_iou": 0.0,
                }
                continue

            ap_per_iou = []
            tp50, fp50, fn50 = 0, 0, 0
            iou_list_tp = []

            for iou_th in iou_thresholds:
                tp_count, fp_count, fn_count = 0, 0, 0

                for img_idx in range(len(gt_instances)):
                    gts = gt_instances[img_idx]
                    preds = pred_instances[img_idx]

                    if not preds:
                        fn_count += len(gts)
                        continue
                    if not gts:
                        fp_count += len(preds)
                        continue

                    preds_sorted = sorted(preds, key=lambda p: p.get("score", 1.0), reverse=True)
                    matched_gt = set()

                    for p in preds_sorted:
                        best_iou = 0.0
                        best_gt_idx = -1
                        for g_idx, g in enumerate(gts):
                            if g_idx in matched_gt:
                                continue
                            iou = self.compute_mask_iou(g["mask"], p["mask"])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx

                        if best_iou >= iou_th and best_gt_idx >= 0:
                            tp_count += 1
                            matched_gt.add(best_gt_idx)
                            if abs(iou_th - 0.50) < 1e-5:
                                iou_list_tp.append(best_iou)
                        else:
                            fp_count += 1

                    fn_count += (len(gts) - len(matched_gt))

                prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
                ap_per_iou.append(prec * rec if (prec + rec) > 0 else prec)

                if abs(iou_th - 0.50) < 1e-5:
                    tp50, fp50, fn50 = tp_count, fp_count, fn_count

            precision50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
            recall50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
            map50 = ap_per_iou[0] if ap_per_iou else 0.0
            map50_95 = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0
            mean_iou = float(np.mean(iou_list_tp)) if iou_list_tp else 0.0

            class_metrics[cname] = {
                "gt_count": n_gt,
                "pred_count": n_pred,
                "precision": round(precision50, 4),
                "recall": round(recall50, 4),
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
                "mean_iou": round(mean_iou, 4),
            }

            sum_map50_all += map50
            sum_map50_95_all += map50_95
            sum_precision_all += precision50
            sum_recall_all += recall50
            sum_iou_all += mean_iou
            valid_classes_count += 1

        avg_map50 = sum_map50_all / valid_classes_count if valid_classes_count > 0 else 0.0
        avg_map50_95 = sum_map50_95_all / valid_classes_count if valid_classes_count > 0 else 0.0
        avg_precision = sum_precision_all / valid_classes_count if valid_classes_count > 0 else 0.0
        avg_recall = sum_recall_all / valid_classes_count if valid_classes_count > 0 else 0.0
        avg_iou = sum_iou_all / valid_classes_count if valid_classes_count > 0 else 0.0

        class_metrics["ALL (Average)"] = {
            "gt_count": total_gt_count_all,
            "pred_count": total_pred_count_all,
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "map50": round(avg_map50, 4),
            "map50_95": round(avg_map50_95, 4),
            "mean_iou": round(avg_iou, 4),
        }

        return class_metrics

    @classmethod
    def run_mask_comparison(
        cls,
        gt_dir,
        infer_source,
        images_dir="",
        classes_file="",
        conf_thresh=0.25,
        output_dir=None,
        progress_callback=None,
    ):
        """
        Unified execution pipeline for Mask Comparison & Evaluation.
        Shared by both GUI (CompareThread) and CLI (compare_mask.py).
        """
        classes = cls.read_classes_file(classes_file)
        cs = cls(classes=classes)

        candidate_dirs = [images_dir, gt_dir, infer_source]
        image_files = []
        exts = ["*.bmp", "*.jpg", "*.png", "*.jpeg", "*.BMP", "*.JPG", "*.PNG", "*.JPEG"]

        for cdir in candidate_dirs:
            if cdir and os.path.exists(cdir):
                if os.path.isfile(cdir) and cdir.lower().endswith(tuple(e.replace("*", "") for e in exts)):
                    image_files.append(cdir)
                elif os.path.isdir(cdir):
                    for ext in exts:
                        image_files.extend(glob.glob(os.path.join(cdir, ext)))
                        image_files.extend(glob.glob(os.path.join(cdir, "**", ext), recursive=True))

        seen_stems = set()
        unique_image_files = []
        for img in image_files:
            stem = Path(img).stem
            if stem not in seen_stems and os.path.isfile(img) and "compare" not in img.lower():
                seen_stems.add(stem)
                unique_image_files.append(img)

        if not unique_image_files:
            raise ValueError("No valid image files found! Please check Ground Truth or Inference result folders.")

        total_imgs = len(unique_image_files)
        if progress_callback:
            progress_callback(0, total_imgs, f"Starting comparison: found {total_imgs} target image(s)...")

        model_instance = None
        if infer_source and os.path.isfile(infer_source) and infer_source.endswith((".pt", ".onnx", ".engine", ".torchscript")):
            try:
                from ultralytics import YOLO
                model_instance = YOLO(infer_source)
            except Exception as ex:
                raise RuntimeError(f"Failed to load model file: {ex}")

        from services.config import get_project_root
        compare_dir = output_dir or os.path.join(get_project_root(), "compare")
        os.makedirs(compare_dir, exist_ok=True)

        all_gt_data = []
        all_pred_data = []
        output_image_paths = []

        for idx, img_path in enumerate(unique_image_files):
            img_name = Path(img_path).name
            if progress_callback:
                progress_callback(idx + 1, total_imgs, f"Processing image {idx+1}/{total_imgs}: {img_name}")

            img_mat = cv2.imread(img_path)
            if img_mat is None:
                continue
            img_h, img_w = img_mat.shape[:2]

            gt_objs = cs.extract_gt_objects(img_path, gt_dir, img_w, img_h)
            pred_objs = cs.extract_pred_objects(img_path, infer_source, img_w, img_h,
                                                model_instance=model_instance, conf_thresh=conf_thresh)

            out_path = os.path.join(compare_dir, img_name)
            cs.draw_comparison_image(img_path, gt_objs, pred_objs, out_path)

            all_gt_data.append(gt_objs)
            all_pred_data.append(pred_objs)
            output_image_paths.append(out_path)

        if progress_callback:
            progress_callback(total_imgs, total_imgs, "Calculating mask-mAP50 metrics and summary...")

        metrics = cs.calculate_metrics(all_gt_data, all_pred_data)
        return metrics, compare_dir, output_image_paths

