# train_utils.py
from ultralytics import YOLO, settings
import os
import torch


class YOLOTrainer:
    """YOLO多任务训练器"""

    def __init__(self, model_type='yolo11n', task='detect', yolo_version='yolo11'):
        # 动态选择模型文件
        model_file = f"{model_type}.pt"
        # 确定模型文件路径（与 train.py 同级目录）
        project_root = os.path.dirname(os.path.dirname(__file__))  # utils/ 的父目录
        local_model_path = os.path.join(project_root, model_file)

        # 设置 ultralytics 权重保存目录
        settings.update({"weights_dir": project_root})

        # 检测模型文件是否存在，不存在则下载
        if not os.path.exists(local_model_path):
            print(f"模型文件 {model_file} 不存在，正在下载到 {project_root}...")
            try:
                # 使用 YOLO 构造函数触发自动下载
                temp_model = YOLO(model_file, verbose=True)
                # 确保文件保存到指定路径
                downloaded_path = temp_model.model.pt_path if hasattr(temp_model.model, 'pt_path') else local_model_path
                if downloaded_path != local_model_path and os.path.exists(downloaded_path):
                    os.rename(downloaded_path, local_model_path)
                print(f"成功下载模型文件 {model_file}")
            except Exception as e:
                raise FileNotFoundError(
                    f"下载模型文件 {model_file} 失败: {str(e)}。请手动从 "
                    f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_file} "
                    f"下载并放置到 {project_root}"
                )

        # 验证模型文件完整性
        try:
            torch.load(local_model_path, map_location="cpu")
            print(f"验证成功：{model_file} 为有效的 PyTorch 模型文件")
        except Exception as e:
            os.remove(local_model_path)  # 删除损坏的文件
            raise RuntimeError(
                f"模型文件 {model_file} 损坏: {str(e)}。已删除文件。请重新运行脚本或手动从 "
                f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_file} "
                f"下载并放置到 {project_root}"
            )

        self.model = YOLO(local_model_path)
        self.task = task
        self.yolo_version = yolo_version  # 存储 yolo_version 以备将来使用

    def train(self, data_yaml, epochs=50, imgsz=640, batch_size=16, device='0'):
        """训练模型"""
        task_configs = {
            'detect': {'imgsz': 640, 'batch_size': 16},
            'segment': {'imgsz': 640, 'batch_size': 16},
            'classify': {'imgsz': 224, 'batch_size': 32},
            'pose': {'imgsz': 640, 'batch_size': 16},
            'obb': {'imgsz': 640, 'batch_size': 16}
        }
        config = task_configs.get(self.task, {'imgsz': 640, 'batch_size': 16})

        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=config['imgsz'],
            batch=batch_size or config['batch_size'],
            device=device,
            task=self.task
        )
        return self.model

    def validate(self, data_yaml):
        """验证模型"""
        metrics = self.model.val(data=data_yaml, task=self.task)
        return metrics

    def export(self, format='onnx', name=None, project=None, opset=17, dynamic=False, simplify=False):
        """导出模型"""
        '''
        --opset：ONNX opset版本（默认：17）
        --dynamic：是否使用动态批处理大小（默认：False）
        --simplify：是否简化模型（默认：False）
        '''

        self.model.export(
            format=format,
            name=name,
            project=project,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify
        )