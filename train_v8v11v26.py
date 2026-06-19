# train_v8v11v26.py

import os
from utils.data_utils import DataPreparer
from utils.train_v8v11v26_utils import YOLOTrainer


def main():
    current_dir = os.getcwd()
    
    # 数据准备路径
    dataset_dir = os.path.join(current_dir, "data_sets", "Origin_dataset")
    output_dir = os.path.join(current_dir, "data_sets", "Train_dataset")

    # 训练配置
    class_names = ['leg', 'breast', 'cord', 'milkcup']  # 固定相机类别
    tasks = ['segment']                                 # 支持: detect, segment, classify, pose, obb
    yolo_versions = ['yolo11']                         # 支持: yolov8, yolo11, yolo26
    model_sizes = ['s']                                 # 支持: n, s, m, l, x

    # ================= 迭代训练 / 断点续训配置 =================
    # 填入你上一次训练完毕的具体的 .pt 文件物理路径
    # 如果不需要，将其设为 None 即可（系统会自动去云端下载官方白板模型）
    iteration_path = "/home/zja/github/yolo11-train/runs/segment/train17/weights/last.pt"
    
    # 模式选择开关：
    # True  -> 用于意外中断后的“断点续训”（继承旧的 Epoch 计数、继承原有学习率进度，继续跑完剩下的轮数）
    # False -> 用于在新基础上的“迭代训练 / 微调”（重置 Epoch 为 0，重置优化器状态，重新跑满全新生命周期）
    is_resume = False  
    # =========================================================

    # 数据集拆分校验与 YAML 生成
    preparer = DataPreparer(dataset_dir, output_dir, tasks, class_names)
    preparer.split_dataset(force=False)
    preparer.generate_yaml()

    # 循环遍历训练
    for yolo_version in yolo_versions:
        for task in tasks:
            for size in model_sizes:
                print(f"Running task: {task} using {yolo_version}{size}")

                # 动态映射模型名称
                task_model_map = {
                    'detect': f'{yolo_version}{size}',
                    'classify': f'{yolo_version}{size}-cls',
                    'segment': f'{yolo_version}{size}-seg',
                    'obb': f'{yolo_version}{size}-obb',
                    'pose': f'{yolo_version}{size}-pose'
                }
                model_type = task_model_map.get(task, f'{yolo_version}{size}')
                
                # 初始化训练器 (若 iteration_path 有效且文件存在，内部直接加载该物理路径)
                trainer = YOLOTrainer(
                    model_type=model_type, 
                    task=task, 
                    yolo_version=yolo_version, 
                    iteration_path=iteration_path
                )
                
                data_yaml = f"{output_dir}/data_{task}.yaml" if task != 'classify' else f"{output_dir}/train"

                # 启动新一轮训练并追加参数详解注释
                trainer.train(
                    data_yaml=data_yaml,  # 数据集配置文件路径
                    epochs=1000,         # 最大训练总轮数
                    imgsz=640,           # 输入图像的尺寸大小 (会将图片缩放到 640x640 进网络)
                    batch_size=2,        # 每批次读入的图像数量 (根据显存大小调整)
                    device='0',          # 指定训练的 GPU 设备号 (例如 '0' 或 '0,1'，使用 CPU 则填 'cpu')
                    mosaic=1.0,          # Mosaic 数据增强概率 (1.0 代表 100% 开启 4 图拼接)
                    lr0=0.002,           # 初始学习率 (全新迭代/微调时，通常建议调小该值，如 0.001~0.004)
                    copy_paste=0.2,      # Copy-Paste 数据增强概率 (分割任务专用，跨图像拷贝粘帖目标)
                    patience=100,        # 早停(Early Stopping)耐心轮数 (若连续 100 轮指标不提升则提前结束训练)
                    resume=is_resume,    # 是否启动恢复模式 (True 为断点接续训练，False 为作为预训练权重开启新一轮迭代)
                )

                # 验证与导出
                print(f"Validating model: {model_type}...")
                trainer.validate(data_yaml)

                print(f"Exporting model: {model_type} to ONNX...")
                output_path = f"{current_dir}/models/{model_type}-{task}.onnx"
                project_dir = os.path.dirname(output_path)
                name = os.path.splitext(os.path.basename(output_path))[0]
                
                trainer.export(format='onnx', opset=17, name=name, project=project_dir, simplify=False)


if __name__ == "__main__":
    main()
