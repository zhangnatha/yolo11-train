import os
import glob
import json
import yaml
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
from services.label_converter import LabelConverter

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

def scan_dataset(dataset_dir: str) -> Tuple[List[str], str]:
    """Scans dataset directory for image files and determines label dir"""
    dataset_path = Path(dataset_dir)
    image_list = []
    
    # Check if images directory exists inside dataset_dir
    images_sub = dataset_path / "images"
    search_dir = images_sub if images_sub.exists() else dataset_path

    for ext in IMAGE_EXTENSIONS:
        image_list.extend([str(p) for p in search_dir.glob(f"*{ext}")])
        image_list.extend([str(p) for p in search_dir.glob(f"*{ext.upper()}")])

    image_list = sorted(list(set(image_list)))
    output_dir = str(dataset_path / "labels") if (dataset_path / "labels").exists() else str(search_dir)
    return image_list, output_dir

def get_project_classes(project_root: str = None, dataset_dir: str = None) -> List[str]:
    """Finds and reads classes.txt or classes.names in project root or dataset directory."""
    candidates = []
    if dataset_dir:
        dp = Path(dataset_dir)
        candidates.extend([
            dp / "classes.txt",
            dp / "classes.names",
            dp / "segment_classes.txt",
            dp / "detect_classes.txt",
        ])
    if project_root:
        pr = Path(project_root)
        candidates.extend([
            pr / "classes.txt",
            pr / "classes.names",
            pr / "data_sets" / "Origin_dataset" / "classes.txt",
            pr / "data_sets" / "Origin_dataset" / "classes.names",
            pr / "data_sets" / "Train_dataset" / "classes.txt",
            pr / "data_sets" / "Train_dataset" / "classes.names",
        ])
    else:
        root = Path(__file__).resolve().parent.parent
        candidates.extend([
            root / "classes.txt",
            root / "classes.names",
            root / "data_sets" / "Origin_dataset" / "classes.txt",
            root / "data_sets" / "Origin_dataset" / "classes.names",
        ])

    for cand in candidates:
        if cand.exists():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        return lines
            except Exception:
                pass

    return ['leg', 'milkcup', 'nipple', 'tail']


def compute_dataset_summary(image_list: List[str], label_dir: str = None, class_names: List[str] = None) -> Tuple[List[str], List[List[str]], List[str]]:
    """Calculates dataset label & shape statistics from JSON or TXT labels, including 0-count classes in order."""
    headers = ["Label", "Polygon", "Rectangle", "Rotation", "Point", "Total"]
    if not class_names:
        class_names = get_project_classes(dataset_dir=label_dir)

    ordered_classes = list(class_names)
    stats = {lbl: {"polygon": 0, "rectangle": 0, "rotation": 0, "point": 0, "_total": 0} for lbl in ordered_classes}

    for img_path in image_list:
        img_p = Path(img_path)
        base_dir = Path(label_dir) if label_dir and os.path.exists(label_dir) else img_p.parent
        
        json_file = base_dir / (img_p.stem + ".json")
        txt_file = base_dir / (img_p.stem + ".txt")

        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    lbl = shape.get("label", "unknown")
                    stype = shape.get("shape_type", "polygon")
                    if lbl not in stats:
                        stats[lbl] = {"polygon": 0, "rectangle": 0, "rotation": 0, "point": 0, "_total": 0}
                        ordered_classes.append(lbl)
                    if stype in stats[lbl]:
                        stats[lbl][stype] += 1
                    stats[lbl]["_total"] += 1
            except Exception:
                pass
        elif txt_file.exists():
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                        lbl = ordered_classes[cls_idx] if 0 <= cls_idx < len(ordered_classes) else f"class_{cls_idx}"
                    except Exception:
                        lbl = f"class_{parts[0]}"
                    stype = "polygon" if len(parts) > 5 else "rectangle"
                    if lbl not in stats:
                        stats[lbl] = {"polygon": 0, "rectangle": 0, "rotation": 0, "point": 0, "_total": 0}
                        ordered_classes.append(lbl)
                    stats[lbl][stype] += 1
                    stats[lbl]["_total"] += 1
            except Exception:
                pass

    table_data = [headers]
    total_poly, total_rect, total_rot, total_pt, grand_total = 0, 0, 0, 0, 0

    for lbl in ordered_classes:
        counts = stats.get(lbl, {"polygon": 0, "rectangle": 0, "rotation": 0, "point": 0, "_total": 0})
        p, r, rot, pt, tot = counts["polygon"], counts["rectangle"], counts["rotation"], counts["point"], counts["_total"]
        total_poly += p
        total_rect += r
        total_rot += rot
        total_pt += pt
        grand_total += tot
        table_data.append([lbl, str(p), str(r), str(rot), str(pt), str(tot)])

    total_row = ["Total", str(total_poly), str(total_rect), str(total_rot), str(total_pt), str(grand_total)]
    table_data.append(total_row)

    return headers, table_data, ordered_classes


def convert_json_dataset_to_yolo(
    image_list: List[str],
    task_type: str,
    output_dataset_dir: str,
    classes_file: str = None
) -> Tuple[str, List[str]]:
    """Converts AnyLabeling JSON annotations to YOLO format labels and images in output_dataset_dir"""
    output_path = Path(output_dataset_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not classes_file or not os.path.exists(classes_file):
        root_classes = Path(__file__).resolve().parent.parent / "classes.txt"
        if root_classes.exists():
            classes_file = str(root_classes)

    classes_content = ""
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as sf:
            classes_content = sf.read()

        for name in ("classes.txt", "classes.names", f"{task_type.lower()}_classes.txt"):
            with open(output_path / name, "w", encoding="utf-8") as df:
                df.write(classes_content)

    mode_map = {
        "Classify": "classify",
        "Detect": "hbb",
        "OBB": "obb",
        "Segment": "seg",
        "Pose": "pose",
    }
    mode = mode_map.get(task_type, "seg")

    converter = LabelConverter(classes_file=classes_file)

    for img_file in image_list:
        img_p = Path(img_file)
        dst_img = images_dir / img_p.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        json_file = img_p.parent / (img_p.stem + ".json")
        dst_label = labels_dir / (img_p.stem + ".txt")
        if json_file.exists():
            converter.custom_to_yolo(str(json_file), str(dst_label), mode=mode)
        elif not dst_label.exists():
            dst_label.touch()

    class_names = converter.classes
    if not class_names:
        class_names = get_project_classes(dataset_dir=output_dataset_dir)

    return str(output_path), class_names


def prepare_origin_dataset(
    source_dir: str,
    task_type: str,
    project_root: str,
    classes_file: str = None,
    force: bool = True,
) -> Tuple[str, List[str], int]:
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    origin_dir = Path(project_root).resolve() / "data_sets" / "Origin_dataset"
    if force and origin_dir.exists():
        shutil.rmtree(origin_dir)
    origin_dir.mkdir(parents=True, exist_ok=True)

    image_list, label_dir = scan_dataset(str(source_path))
    if not image_list:
        raise RuntimeError(f"No image files found in directory: {source_dir}")

    resolved_classes_file = classes_file
    if not resolved_classes_file or not os.path.exists(resolved_classes_file):
        root_classes = Path(project_root).resolve() / "classes.txt"
        if root_classes.exists():
            resolved_classes_file = str(root_classes)
        else:
            candidate_names = source_path / "classes.names"
            candidate_txt = source_path / "classes.txt"
            if candidate_names.exists():
                resolved_classes_file = str(candidate_names)
            elif candidate_txt.exists():
                resolved_classes_file = str(candidate_txt)

    output_dir, class_names = convert_json_dataset_to_yolo(
        image_list=image_list,
        task_type=task_type,
        output_dataset_dir=str(origin_dir),
        classes_file=resolved_classes_file,
    )

    return output_dir, class_names, len(image_list)


def create_train_dataset_split(
    origin_dataset_dir: str,
    train_dataset_dir: str,
    task_type: str,
    train_ratio: float,
    val_ratio: float,
    class_names: List[str],
    seed: int = 0,
) -> str:
    origin_path = Path(origin_dataset_dir).resolve()
    train_path = Path(train_dataset_dir).resolve()
    if not origin_path.exists():
        raise FileNotFoundError(f"Origin_dataset does not exist: {origin_dataset_dir}")

    if not class_names:
        class_names = get_project_classes(dataset_dir=origin_dataset_dir)

    images_dir = origin_path / "images"
    labels_dir = origin_path / "labels"
    if not images_dir.exists():
        raise RuntimeError(f"Origin_dataset missing images directory: {images_dir}")

    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(images_dir.glob(f"*{ext}"))
        all_images.extend(images_dir.glob(f"*{ext.upper()}"))
    all_images = sorted(set(all_images))
    if not all_images:
        raise RuntimeError(f"No images found to split in Origin_dataset: {images_dir}")

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be > 0 and sum < 1")

    if train_path.exists():
        shutil.rmtree(train_path)

    for split in ("train", "val", "test"):
        (train_path / split / "images").mkdir(parents=True, exist_ok=True)
        if task_type.lower() != "classify":
            (train_path / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    split_map = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:],
    }

    for split, images in split_map.items():
        for image_path in images:
            shutil.copy2(image_path, train_path / split / "images" / image_path.name)
            if task_type.lower() != "classify":
                label_path = labels_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, train_path / split / "labels" / label_path.name)
                else:
                    (train_path / split / "labels" / f"{image_path.stem}.txt").touch()

    # Copy class text files to train dataset directory
    classes_str = "\n".join(class_names) + "\n"
    for filename in ("classes.txt", "classes.names", f"{task_type.lower()}_classes.txt"):
        with open(train_path / filename, "w", encoding="utf-8") as cf:
            cf.write(classes_str)

    yaml_names = {i: name for i, name in enumerate(class_names)}
    yaml_data = {
        "path": str(train_path),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": yaml_names,
    }
    yaml_path = train_path / f"data_{task_type.lower()}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)

    return str(yaml_path)
