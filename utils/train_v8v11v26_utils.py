# utils/train_v8v11v26_utils.py

from ultralytics import YOLO, settings
import os
import torch
import urllib.request
import sys
import inspect


class YOLOTrainer:
    """YOLOv8/YOLO11/YOLO26 多任务训练器"""

    def __init__(self, model_type='yolo11n', task='detect', yolo_version='yolo11', iteration_path=None):
        self.task = task
        self.yolo_version = yolo_version

        # 核心逻辑：如果传入了具体的迭代权重物理路径且存在，直接加载，不触发云端官方模型下载
        if iteration_path and os.path.exists(iteration_path):
            print(f"Loading custom target weight for training: {iteration_path}")
            self.model = YOLO(iteration_path)
            return

        # 否则走标准官方模型初始化与自动云端下载逻辑
        model_file = f"{model_type}.pt"
        project_root = os.path.dirname(os.path.dirname(__file__))
        local_model_path = os.path.join(project_root, model_file)

        settings.update({"weights_dir": project_root})

        release_tag = "v8.4.0"
        download_url = f"https://github.com/ultralytics/assets/releases/download/{release_tag}/{model_file}"

        if not os.path.exists(local_model_path):
            print(f"Model file {model_file} not found. Downloading to {project_root}...")
            try:
                def _progress_callback(block_num, block_size, total_size):
                    read_so_far = block_num * block_size
                    if total_size > 0:
                        percent = min(100, read_so_far * 100 / total_size)
                        nav = int(percent // 4)
                        sys.stdout.write(f"\rProgress: [{'=' * nav}{' ' * (25 - nav)}] {percent:.2f}%")
                        sys.stdout.flush()

                urllib.request.urlretrieve(download_url, local_model_path, _progress_callback)
                print(f"\nSuccessfully downloaded {model_file}")
            except Exception as e:
                print(f"\nDownload failed: {str(e)}. Trying fallback mechanism...")
                if os.path.exists(local_model_path):
                    os.remove(local_model_path)
                try:
                    temp_model = YOLO(model_file, verbose=True)
                    downloaded_path = temp_model.model.pt_path if hasattr(temp_model.model, 'pt_path') else local_model_path
                    if downloaded_path != local_model_path and os.path.exists(downloaded_path):
                        os.rename(downloaded_path, local_model_path)
                except Exception as inner_e:
                    raise RuntimeError(f"Failed to download {model_file}: {str(inner_e)}")

        # 验证预训练权重文件完整性
        try:
            load_args = inspect.signature(torch.load).parameters
            if 'weights_only' in load_args:
                torch.load(local_model_path, map_location="cpu", weights_only=False)
            else:
                torch.load(local_model_path, map_location="cpu")
        except Exception as e:
            if os.path.exists(local_model_path):
                os.remove(local_model_path)
            raise RuntimeError(f"Model file {model_file} is corrupted: {str(e)}.")

        self.model = YOLO(local_model_path)

    def train(self, data_yaml, epochs=50, imgsz=640, batch_size=16, device='0', **kwargs):
        """训练模型"""
        task_configs = {
            'detect': {'imgsz': 640, 'batch_size': 16},
            'segment': {'imgsz': 640, 'batch_size': 16},
            'classify': {'imgsz': 224, 'batch_size': 32},
            'pose': {'imgsz': 640, 'batch_size': 16},
            'obb': {'imgsz': 640, 'batch_size': 16}
        }
        config = task_configs.get(self.task, {'imgsz': 640, 'batch_size': 16})

        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': config['imgsz'],
            'batch': batch_size or config['batch_size'],
            'device': device,
            'task': self.task,
            **kwargs  # 接收来自外部的 resume 状态控制开关
        }

        self.model.train(**train_args)
        return self.model

    def validate(self, data_yaml):
        """验证模型"""
        return self.model.val(data=data_yaml, task=self.task)

    def export(self, format='onnx', name=None, project=None, opset=17, dynamic=False, simplify=False):
        """导出模型"""
        self.model.export(
            format=format,
            name=name,
            project=project,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify
        )
