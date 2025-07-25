# data_utils.py
import os
import shutil
import yaml
from pathlib import Path


class DataPreparer:
    """准备自定义数据集"""

    def __init__(self, dataset_dir, output_dir, tasks=['detect'], class_names=None):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.tasks = tasks
        self.class_names = class_names
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    def validate_label(self, label_path, task, kpt_count=0):
        """验证标签文件格式"""
        if not label_path.exists():
            return False
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                return False
            if task == 'detect' and len(parts) != 5:
                print(f"Invalid detect label format in {label_path}: Expected 5 values, got {len(parts)}")
                return False
            elif task == 'segment' and len(parts) < 6:
                print(f"Invalid segment label format in {label_path}: Expected 6+ values, got {len(parts)}")
                return False
            elif task == 'pose' and len(parts) < 5 + kpt_count:
                print(f"Invalid pose label format in {label_path}: Expected 5 + {kpt_count} values, got {len(parts)}")
                return False
            elif task == 'obb' and len(parts) != 9:
                print(f"Invalid obb label format in {label_path}: Expected 9 values, got {len(parts)}")
                return False
        return True

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """划分训练集、验证集和测试集"""
        images_dir = self.dataset_dir / "images"
        labels_dir = self.dataset_dir / "labels" if 'classify' not in self.tasks else images_dir

        # 创建目录
        for split in ['train', 'val', 'test']:
            os.makedirs(self.output_dir / f"{split}/images", exist_ok=True)
            if 'classify' not in self.tasks:
                os.makedirs(self.output_dir / f"{split}/labels", exist_ok=True)

        if 'classify' in self.tasks:
            # 分类任务：图像按类别组织在子文件夹中，支持所有self.image_extensions格式
            for class_name in self.class_names:
                class_dir = images_dir / class_name
                if not class_dir.is_dir():
                    print(f"Warning: Class directory {class_dir} not found")
                    continue
                images = []
                for ext in self.image_extensions:
                    images.extend(class_dir.glob(f"*{ext}"))

                num_images = len(images)
                train_end = int(num_images * train_ratio)
                val_end = train_end + int(num_images * val_ratio)

                for i, img in enumerate(images):
                    split = "train" if i < train_end else "val" if i < val_end else "test"
                    target_dir = self.output_dir / f"{split}/{class_name}"
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(img, target_dir)

            # 清理多余的images和train_split目录
            for split in ['train', 'val', 'test']:
                images_dir_to_remove = self.output_dir / f"{split}/images"
                if images_dir_to_remove.exists() and images_dir_to_remove.is_dir():
                    try:
                        shutil.rmtree(images_dir_to_remove)
                    except Exception as e:
                        print(f"Warning: Failed to remove {images_dir_to_remove}: {e}")
            train_split_dir = self.output_dir / "train_split"
            if train_split_dir.exists() and train_split_dir.is_dir():
                try:
                    shutil.rmtree(train_split_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove {train_split_dir}: {e}")
        else:
            # 其他任务（detect, segment, pose, obb）
            images = []
            for ext in self.image_extensions:
                images.extend(images_dir.glob(f"*{ext}"))

            num_images = len(images)
            train_end = int(num_images * train_ratio)
            val_end = train_end + int(num_images * val_ratio)

            # 读取 pose_classes.yaml 以获取关键点数量（用于验证）
            kpt_count = 0
            if 'pose' in self.tasks:
                pose_yaml_path = self.dataset_dir / 'pose_classes.yaml'
                if pose_yaml_path.exists():
                    with open(pose_yaml_path, 'r') as f:
                        pose_config = yaml.safe_load(f)
                    first_class = list(pose_config['classes'].keys())[0]
                    kpt_count = len(pose_config['classes'][first_class]) * (
                        3 if pose_config.get('has_visible', False) else 2)

            for i, img in enumerate(images):
                label = labels_dir / (img.stem + ".txt")
                split = "train" if i < train_end else "val" if i < val_end else "test"
                shutil.copy(img, self.output_dir / f"{split}/images")
                if label.exists():
                    if self.validate_label(label, self.tasks[0], kpt_count):  # Validate first task's format
                        shutil.copy(label, self.output_dir / f"{split}/labels")
                    else:
                        print(f"Skipping {img} due to invalid label format")
                else:
                    print(f"Warning: Label file {label} not found for image {img}")

    def generate_yaml(self):
        """生成data.yaml配置文件"""
        for task in self.tasks:
            if task == 'classify':
                continue  # 分类任务无需YAML

            # 初始化默认配置
            dataset_name = self.output_dir.name
            data = {
                'path': f'./data_sets/{dataset_name}',
                'train': 'train',
                'val': 'val',
                'test': 'test',
                'names': {i: name for i, name in enumerate(self.class_names)}
            }

            # 处理 pose 任务
            if task == 'pose':
                pose_yaml_path = self.dataset_dir / 'pose_classes.yaml'
                if pose_yaml_path.exists():
                    with open(pose_yaml_path, 'r') as f:
                        pose_config = yaml.safe_load(f)
                    # 提取类名
                    data['names'] = {i: cls for i, cls in enumerate(pose_config['classes'].keys())}
                    # 提取关键点数量和维度
                    first_class = list(pose_config['classes'].keys())[0]
                    kpt_count = len(pose_config['classes'][first_class])
                    kpt_dims = 3 if pose_config.get('has_visible', False) else 2
                    data['kpt_shape'] = [kpt_count, kpt_dims]
                    # 生成 flip_idx（假设简单顺序）
                    data['flip_idx'] = list(range(kpt_count))
                else:
                    print(f"Warning: pose_classes.yaml not found in {self.dataset_dir}, using default names")

            # 处理 segment, detect, obb 任务
            elif task in ['segment', 'detect', 'obb']:
                # 为每个任务使用对应的类文件
                task_class_file = f"{task}_classes.txt"
                txt_path = self.dataset_dir / task_class_file
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                    data['names'] = {i: name for i, name in enumerate(class_names)}
                else:
                    print(f"Warning: {task_class_file} not found in {self.dataset_dir}, using default names")

            # 格式化 YAML 输出
            yaml_content = "# Train/val/test sets\n"
            yaml_content += f"path: ./data_sets/{dataset_name}\n"
            yaml_content += "train: train\n"
            yaml_content += "val: val\n"
            yaml_content += "test: test\n\n"

            if task == 'pose':
                yaml_content += "# Keypoints\n"
                yaml_content += f"kpt_shape: [{kpt_count}, {kpt_dims}] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)\n"
                yaml_content += f"flip_idx: {list(range(kpt_count))}\n\n"

            yaml_content += "# Classes\n"
            yaml_content += "names:\n"
            for i, name in data['names'].items():
                yaml_content += f"  {i}: {name}\n"

            yaml_path = self.output_dir / f"data_{task}.yaml"
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)