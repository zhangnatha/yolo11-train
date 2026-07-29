# utils/train_utils.py

from ultralytics import YOLO, settings
import os
import torch
import urllib.request
import sys
import inspect


class YOLOTrainer:
    """YOLO 多任务训练器"""

    def __init__(self, model_type='yolo11n', task='detect', yolo_version='yolo11', iteration_path=None):
        self.task = task
        self.yolo_version = yolo_version

        # 核心逻辑：如果传入了具体的迭代权重物理路径且存在，直接加载，不触发云端官方模型下载
        if iteration_path and os.path.exists(iteration_path):
            print(f"Loading custom target weight for training: {iteration_path}")
            self.model = YOLO(iteration_path)
            return

        # 如果 model_type 本身是一个已存在的物理文件路径，直接加载
        if os.path.exists(model_type):
            print(f"Loading weight from file: {model_type}")
            self.model = YOLO(model_type)
            return

        # 否则走标准官方模型初始化与自动云端下载逻辑（保存在 pre_trained_model/ 目录下）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pretrained_dir = os.path.join(project_root, "pre_trained_model")
        os.makedirs(pretrained_dir, exist_ok=True)

        settings.update({"weights_dir": pretrained_dir})

        filename = os.path.basename(model_type)
        model_file = filename if filename.endswith(".pt") else f"{filename}.pt"

        # 自动校验并修复缺少尺寸字母的错误模型文件名（例如 yolo11-seg.pt -> yolo11s-seg.pt）
        import re
        m = re.match(r"^(yolo\d+)(-[a-z]+)?\.pt$", model_file, re.IGNORECASE)
        if m:
            model_file = f"{m.group(1)}s{m.group(2) or ''}.pt"

        local_model_path = os.path.join(pretrained_dir, model_file)
        root_model_path = os.path.join(project_root, model_file)

        # 若根目录下存在该预训练权重，自动迁移至 pre_trained_model 文件夹下
        if not os.path.exists(local_model_path) and os.path.exists(root_model_path):
            print(f"Moving {model_file} from project root to {pretrained_dir}...")
            os.rename(root_model_path, local_model_path)

        if not os.path.exists(local_model_path):
            print(f"Model file {model_file} not found. Downloading to {pretrained_dir}...")
            release_tags = ["v8.3.0", "v8.2.0", "v8.0.0", "v8.4.0"]
            download_success = False
            for tag in release_tags:
                download_url = f"https://github.com/ultralytics/assets/releases/download/{tag}/{model_file}"
                try:
                    def _progress_callback(block_num, block_size, total_size):
                        read_so_far = block_num * block_size
                        if total_size > 0:
                            percent = min(100, read_so_far * 100 / total_size)
                            nav = int(percent // 4)
                            sys.stdout.write(f"\rProgress: [{'=' * nav}{' ' * (25 - nav)}] {percent:.2f}%")
                            sys.stdout.flush()

                    urllib.request.urlretrieve(download_url, local_model_path, _progress_callback)
                    print(f"\nSuccessfully downloaded {model_file} to {pretrained_dir}")
                    download_success = True
                    break
                except Exception:
                    if os.path.exists(local_model_path):
                        os.remove(local_model_path)
                    continue

            if not download_success:
                print(f"\nDirect release URL download failed. Trying fallback mechanism...")
                try:
                    import shutil
                    temp_model = YOLO(model_file, verbose=True)
                    downloaded_path = temp_model.model.pt_path if hasattr(temp_model, 'model') and hasattr(temp_model.model, 'pt_path') else os.path.join(os.getcwd(), model_file)
                    if os.path.exists(downloaded_path) and downloaded_path != local_model_path:
                        shutil.copy2(downloaded_path, local_model_path)
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

    def train(self, data_yaml=None, data=None, epochs=50, imgsz=None, batch_size=None, batch=None, device='0', **kwargs):
        """训练模型"""
        target_data_yaml = data_yaml or data or kwargs.pop('data', None)
        if not target_data_yaml:
            raise ValueError("Dataset YAML path is required (pass data_yaml or data).")

        task_configs = {
            'detect': {'imgsz': 640, 'batch_size': 16},
            'segment': {'imgsz': 640, 'batch_size': 16},
            'classify': {'imgsz': 224, 'batch_size': 32},
            'pose': {'imgsz': 640, 'batch_size': 16},
            'obb': {'imgsz': 640, 'batch_size': 16}
        }
        config = task_configs.get(self.task, {'imgsz': 640, 'batch_size': 16})

        effective_batch = batch if batch is not None else (batch_size or config['batch_size'])
        effective_imgsz = imgsz or config['imgsz']

        train_args = {
            'data': target_data_yaml,
            'epochs': epochs,
            'imgsz': effective_imgsz,
            'batch': effective_batch,
            'device': device,
            'task': self.task,
            **kwargs  # 接收来自外部的 resume 状态控制开关
        }

        def _on_fit_epoch_end(trainer):
            cur = getattr(trainer, 'epoch', 0) + 1
            tot = getattr(trainer, 'epochs', epochs)
            print(f"[EPOCH_PROGRESS] {cur}/{tot}", flush=True)

        try:
            self.model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        except Exception:
            pass

        self.model.train(**train_args)
        return self.model

    def validate(self, data_yaml):
        """验证模型"""
        return self.model.val(data=data_yaml, task=self.task)

    def export(self, format='onnx', name=None, project=None, opset=17, dynamic=False, simplify=False, device=None, **kwargs):
        """导出模型，自动筛选特定格式所支持的参数，并在 TensorRT 导出前预检 GPU 显卡架构"""
        if format == 'engine':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TensorRT 导出失败: 未检测到可用 GPU (CUDA)。TensorRT 引擎导出需要 GPU 环境。\n"
                    "建议解决方案:\n"
                    "1. 导出格式请选择 ONNX (*.onnx) 或 TorchScript (*.torchscript)。\n"
                    "2. 若需导出 TensorRT .engine 格式，请在拥有 NVIDIA GPU 的环境运行。"
                )
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            if major < 7:
                raise RuntimeError(
                    f"TensorRT 导出失败: 当前 GPU ({gpu_name}, SM {major}.{minor}) 不支持 TensorRT 10/11 版本 (TensorRT 10+ 已废弃对 SM 6.x 架构的支持)。\n"
                    f"建议解决方案:\n"
                    f"1. 导出格式请切换选择 ONNX (*.onnx) 或 TorchScript (*.torchscript)。\n"
                    f"2. 若必须导出 TensorRT .engine 格式，请在配备 RTX 20/30/40 系列 (SM 7.5+) 显卡的设备上运行。"
                )

        export_args = {'format': format}
        if name is not None:
            export_args['name'] = name
        if project is not None:
            export_args['project'] = project

        valid_params = None
        try:
            from ultralytics.engine.exporter import export_formats
            df = export_formats()
            fmt_map = dict(zip(df['Argument'], df['Arguments']))
            valid_params = fmt_map.get(format)
        except Exception:
            pass

        candidate_args = {
            'opset': opset,
            'dynamic': dynamic,
            'simplify': simplify,
            'device': device,
            **kwargs
        }

        for key, val in candidate_args.items():
            if val is None:
                continue
            if valid_params is not None:
                if key in valid_params:
                    export_args[key] = val
            else:
                if key == 'opset' and format not in ('onnx', 'engine', 'openvino', 'paddle', 'mnn', 'rknn', 'qnn'):
                    continue
                if key == 'simplify' and format not in ('onnx', 'engine', 'mnn', 'rknn', 'qnn', 'hailo'):
                    continue
                export_args[key] = val

        try:
            return self.model.export(**export_args)
        except Exception as e:
            err_str = str(e)
            if format == "engine":
                if "Unsupported SM" in err_str or "nullptr" in err_str or "createInferBuilder" in err_str or "pybind11" in err_str:
                    raise RuntimeError(
                        f"TensorRT 导出失败: 当前 GPU 显卡架构 (Pascal SM 6.1, GTX 10xx 显卡) 不支持 TensorRT 10/11 版本 (TensorRT 10+ 已废弃对 SM 6.x 架构的支持)。\n"
                        f"建议解决方案:\n"
                        f"1. 导出格式请切换选择 ONNX (*.onnx) 或 TorchScript (*.torchscript)。\n"
                        f"2. 若必须导出 TensorRT .engine 格式，请在配备 RTX 20/30/40 系列 (SM 7.5+) 显卡的设备上运行。\n\n"
                        f"原始报错: {err_str}"
                    ) from e
                if "nvidia-smi" in err_str or "GPU" in err_str or "TensorRT" in err_str:
                    raise RuntimeError(
                        "TensorRT engine 导出失败: 当前环境缺少可用的 TensorRT/GPU 导出条件，或显卡架构不受支持。\n"
                        "建议先使用 ONNX 或 TorchScript 验证模型，再在支持 TensorRT 的 RTX 环境执行 engine 导出。\n\n"
                        f"原始报错: {err_str}"
                    ) from e
            raise


