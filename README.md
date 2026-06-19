# yolo-train (v8 / v11 / v26)

本仓库提供了一套全自动化的 YOLOv8、YOLO11 和 YOLO26 多任务训练、验证及导出工作流。支持自定义数据集的自动划分、规范校验（支持检测、分割、旋转框、关键点、分类等任务），并能全自动从官方云端安全下载对应版本的模型权重。

----



## 1. 配置训练环境

修改 `environment.yml` 中的 `prefix` 路径，然后一键创建 Conda 环境：

```shell
conda env create -f environment.yml
conda activate yolov8v11v26

```

---



## 2. 准备与标注数据

推荐使用 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling.git) 工具进行辅助标注：

1. **标注并导出标签**：根据任务类型导出标准的 YOLO 格式标签（水平矩形框、旋转矩形框 OBB、实例分割或关键点 Pose）。
2. **生成类别定义文件**（存放于原始数据集根目录下）：
* 检测、分割、旋转框任务：创建统一的 `classes.names` 文件，每行一个类别名称。
* 关键点 Pose 任务：创建 `pose_classes.yaml`，格式示例如下：
```yaml
has_visible: true
classes:
  rack:
    - k1
    - k2
  block:
    - p1
    - p2

```

---



## 3. 数据集组织结构

将所有准备好的图像、标签及类别配置文件放入指定的原始数据集中（即 `data_sets/Origin_dataset`）。执行训练脚本前后的完整目录结构如下呈现：

```shell
.
├── data_sets                         # 数据集根目录
│   ├── Origin_dataset                # 原始未划分的数据集
│   │   ├── classes.names             # 检测/分割/旋转框任务使用的类别定义文件
│   │   ├── images/                   # 存放所有原始图像 (.jpg, .png 等)
│   │   ├── labels/                   # 存放标注生成的 txt 标签文件 (分类任务除外)
│   │   ├── pose_classes.yaml         # 关键点任务配置文件
│   │   └── README.md                 # 原始数据自述文件
│   └── Train_dataset                 # 脚本自动划分生成的标准数据集 (无需手动创建)
│       ├── data_segment.yaml         # 自动生成的对应任务 YAML 配置文件
│       ├── train/                    # 划分出的训练集 (包含 images 和 labels)
│       ├── val/                      # 划分出的验证集
│       └── test/                     # 划分出的测试集
├── environment.yml                   # 环境配置文件
├── train_v8v11v26.py                 # 训练主入口脚本
└── utils/                            # 工具包目录
    ├── data_utils.py                 # 数据集校验与划分工具
    └── train_v8v11v26_utils.py       # 模型自动下载、训练与导出工具

```

---



## 4. 启动自动化训练

### 步骤 1：修改训练脚本

编辑主入口脚本 `train_v8v11v26.py`，指定您想要训练的参数配置：

* `class_names`: 类别列表（如 `['wirerope']`）。
* `tasks`: 训练任务类型（支持 `['detect']`, `['segment']`, `['pose']`, `['obb']`, `['classify']`）。
* `yolo_versions`: 需要运行的 YOLO 系列版本（如 `['yolov8', 'yolo11', 'yolo26']`）。
* `model_sizes`: 模型体量大小（如 `['n', 's', 'm', 'l', 'x']`）。

### 步骤 2：运行脚本

脚本调用 `utils/data_utils.py` 自动划分数据集并分配至 `Train_dataset`、生成对应任务的 `data_*.yaml`（例如 `data_segment.yaml`），随后通过 `utils/train_v8v11v26_utils.py` 触发流式下载（统一托管于官方最新 `v8.4.0` 库）与多任务序列训练：

```shell
python train_v8v11v26.py

```

----



## 5. 训练产物

训练完成后，系统会自动在项目根目录下生成并归档以下内容：

* **`data_sets/Train_dataset/`**: 自动按比例划分好并附带对应任务配置 YAML 文件的标准数据集。
* **`runs/`**: 分门别类存储的各版本、各任务的训练日志、图表及最佳模型权重 (`best.pt`)。
* **`*.pt`**: 运行期间全自动从云端拉取并经过安全反序列化校验（兼容 PyTorch 2.6+）的官方预训练权重基底。

