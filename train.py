# +----------+------------------+-------------+
# | 模型     | 文件名             | 任务         |
# +----------+------------------+-------------+
# | YOLO11-  | yolo11n.pt       | 检测         |
# | det      | yolo11s.pt       |             |
# |          | yolo11m.pt       |             |
# |          | yolo11l.pt       |             |
# |          | yolo11x.pt       |             |
# +----------+------------------+-------------+
# | YOLO11-  | yolo11n-seg.pt   | 实例分割     |
# | seg      | yolo11s-seg.pt   |             |
# |          | yolo11m-seg.pt   |             |
# |          | yolo11l-seg.pt   |             |
# |          | yolo11x-seg.pt   |             |
# +----------+------------------+-------------+
# | YOLO11-  | yolo11n-pose.pt  | 姿势/关键点   |
# | pose     | yolo11s-pose.pt  |             |
# |          | yolo11m-pose.pt  |             |
# |          | yolo11l-pose.pt  |             |
# |          | yolo11x-pose.pt  |             |
# +----------+------------------+-------------+
# | YOLO11-  | yolo11n-obb.pt   | 定向检测     |
# | obb      | yolo11s-obb.pt   |             |
# |          | yolo11m-obb.pt   |             |
# |          | yolo11l-obb.pt   |             |
# |          | yolo11x-obb.pt   |             |
# +----------+------------------+-------------+
# | YOLO11-  | yolo11n-cls.pt   | 分类         |
# | cls      | yolo11s-cls.pt   |             |
# |          | yolo11m-cls.pt   |             |
# |          | yolo11l-cls.pt   |             |
# |          | yolo11x-cls.pt   |             |
# +----------+------------------+-------------+

# train.py
import os
from utils.data_utils import DataPreparer
from utils.train_utils import YOLOTrainer


def main():
    current_dir = os.getcwd()
    # 数据准备
    dataset_dir = current_dir + "/data_sets/Origin_dataset"
    output_dir = current_dir + "/data_sets/Train_dataset"
    class_names = ['stick', 'hole', 'small', 'circle']  # 替换为你的类别
    # tasks = ['detect', 'segment', 'classify', 'pose', 'obb']  # 支持所有任务
    # tasks = ['obb']
    # tasks = ['detect']
    tasks = ['segment']
    # tasks = ['classify']
    # tasks = ['pose']
    yolo_version = 'yolo11'  # 使用 yolov8 或者 yolo11

    # 数据准备
    preparer = DataPreparer(dataset_dir, output_dir, tasks, class_names)
    preparer.split_dataset()
    preparer.generate_yaml()

    # 训练每个任务
    for task in tasks:
        print(f"训练任务: {task}，使用 {yolo_version}")

        # 模型选择
        task_model_map = {
            'detect': f'{yolo_version}n',
            'classify': f'{yolo_version}n-cls',
            'segment': f'{yolo_version}n-seg',
            'obb': f'{yolo_version}n-obb',
            'pose': f'{yolo_version}n-pose'
        }
        model_type = task_model_map.get(task, f'{yolo_version}n')
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
