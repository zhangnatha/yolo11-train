import os
from utils.data_utils import DataPreparer
from utils.train_v8v11_utils import YOLOTrainer

def main():
    current_dir = os.getcwd()
    # 数据准备
    dataset_dir = os.path.join(current_dir, "data_sets", "Origin_dataset")
    output_dir = os.path.join(current_dir, "data_sets", "Train_dataset")

    '''
    class_names = ['stick', 'glass', 'hole', 'circle']  # 替换为你的类别
    tasks = ['detect', 'segment', 'classify', 'pose', 'obb']  # 支持所有任务
    yolo_versions = ['yolov8', 'yolo11']  # 支持 YOLOv8, YOLOv11
    model_sizes = ['n', 's', 'm', 'l', 'x']  # 支持所有模型大小
    '''
    class_names = ['wirerope']  # 替换为你的类别
    tasks = ['segment']  # 支持所有任务
    yolo_versions = ['yolov8']  # 支持 YOLOv8, YOLOv11
    model_sizes = ['s']  # 支持所有模型大小

    # 数据准备
    preparer = DataPreparer(dataset_dir, output_dir, tasks, class_names)
    preparer.split_dataset()
    preparer.generate_yaml()

    # 训练每个 YOLO 版本、任务和模型大小
    for yolo_version in yolo_versions:
        for task in tasks:
            for size in model_sizes:
                print(f"训练任务: {task}，使用 {yolo_version}{size}")

                # 模型选择
                task_model_map = {
                    'detect': f'{yolo_version}{size}',
                    'classify': f'{yolo_version}{size}-cls',
                    'segment': f'{yolo_version}{size}-seg',
                    'obb': f'{yolo_version}{size}-obb',
                    'pose': f'{yolo_version}{size}-pose'
                }
                model_type = task_model_map.get(task, f'{yolo_version}{size}')
                trainer = YOLOTrainer(model_type=model_type, task=task, yolo_version=yolo_version)
                # 使用任务特定的 YAML 文件（分类任务除外）
                data_yaml = f"{output_dir}/data_{task}.yaml" if task != 'classify' else f"{output_dir}/train"

                # 训练模型
                model = trainer.train(data_yaml=data_yaml, epochs=1000, imgsz=640, batch_size=16, device='0')

                # 验证模型
                trainer.validate(data_yaml)

                # 导出模型
                output_path = f"{current_dir}/models/{model_type}-{task}.onnx"
                project_dir = os.path.dirname(output_path)
                name = os.path.splitext(os.path.basename(output_path))[0]
                trainer.export(format='onnx', opset=17, name=name, project=project_dir, simplify=False)

if __name__ == "__main__":
    main()