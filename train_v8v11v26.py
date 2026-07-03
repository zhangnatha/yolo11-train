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
    class_names = ['stick', 'hole', 'small', 'circle']  # 固定相机类别
    tasks = ['segment']                                 # 支持: detect, segment, classify, pose, obb
    yolo_versions = ['yolo11']                          # 支持: yolov8, yolo11, yolo26
    model_sizes = ['s']                                 # 支持: n, s, m, l, x

    # ================= 训练模式配置 =================
    # True  -> 基于给定的 pt 模型继续训练 / 迭代
    # False -> 彻底从零开始训练（不加载任何本地 pt 权重）
    is_resume = False

    # 根据你的开关，动态决定权重路径
    if is_resume:
        # 请在此处自行替换为你实际的 .pt 模型绝对路径
        iteration_path = "/path/to/your/custom_model.pt"
        print(f"模式已选择：基于给定模型 [{iteration_path}] 进行训练。")
    else:
        iteration_path = None
        print("模式已选择：彻底从零开始训练。")
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
                    iteration_path=iteration_path  # 当 is_resume=False 时传入 None，底层走官方标准初始化
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
                    resume=False,        # 固定为 False，防范 Ultralytics 断点误判
                    multi_scale=False,   # 强制规避旧权重带来的 0.0 浮点数类型报错
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
