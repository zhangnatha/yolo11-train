

# YOLO 训练与推理工作台 (YOLO Training & Inference Platform)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/PyQt-5%20%2F%206-41CD52?style=flat-square&logo=qt&logoColor=white" alt="PyQt 5/6">
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-blueviolet?style=flat-square" alt="YOLO Ultralytics">
  <img src="https://img.shields.io/badge/CUDA-11.8_%7C_12.1_%7C_12.8-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 11.8 / 12.1 / 12.8">
  <img src="https://img.shields.io/badge/OS-Windows_10%2F11_%7C_Ubuntu_18%2F20%2F22%2F24-0078D6?style=flat-square&logo=windows&logoColor=white" alt="Windows & Ubuntu">
</p>

这是一个面向本地调试和桌面使用的一体化 YOLO 多任务工作台，完整支持 **detect**（目标检测）、**segment**（实例分割）、**classify**（图像分类）、**pose**（姿态估计）、**obb**（定向/旋转框检测）五类核心视觉任务，提供 1:1 对齐的图形界面 (GUI) 与独立的命令行脚本通道。

> **解耦与兼容说明**：采用通用化架构设计，通用支持 **YOLOv8**、**YOLO11**、**YOLO26** 等全系列 YOLO 架构，命令行按功能模块拆分为独立最小单元脚本。

---

## 跨平台环境配置 SOP (Windows & Ubuntu)

本文档适用于在 **Windows 10/11** 或 **Ubuntu 18.04 / 20.04 / 22.04 / 24.04** 上一步到位配置项目运行环境。

### 1. 创建 Conda 虚拟环境

打开终端（Linux）或 Anaconda Prompt（Windows）：
```bash
# 创建 Python 3.10 独立虚拟环境
conda create -n yolo python=3.10 -y

# 激活环境
conda activate yolo
```

### 2. 安装 PyTorch 与 CUDA 支持

根据显卡驱动 / GPU 架构选择对应的 PyTorch 安装指令：

```bash
# A. CUDA 11.8 推荐指令 (Windows / Linux 通用)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# B. CUDA 12.1 推荐指令 (Windows / Linux 通用)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# C. RTX 50 系列（如 RTX 5090，Blackwell / sm_120）必须使用 CUDA 12.8 构建
#    若仍安装 cu118 / cu121，训练时会报错：
#    CUDA error: no kernel image is available for execution on the device
pip install --upgrade \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# D. 若仅使用 CPU 运行（无 NVIDIA 显卡）：
# pip install torch torchvision torchaudio
```

> **RTX 5090 说明**：该卡计算能力为 `sm_120`，旧版 PyTorch（如 `2.5.1+cu121`）仅支持到 `sm_90`，无法在 GPU 上执行内核。请使用上方 **C** 的 `cu128` 安装方式，并确认驱动已支持 50 系显卡。

**验证 PyTorch 可用性：**

```bash
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

**RTX 50 系列额外验证（确认已包含 sm_120）：**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); x=torch.randn(64,64,device='cuda'); print('OK', (x@x.T).shape)"
```

### 3. 按操作系统安装依赖 (GUI 与 Ultralytics & 模型导出)

不同操作系统的依赖兼容规则不同，请**按你的操作系统**选择执行对应的代码块：

#### 选项 A：Ubuntu 环境安装

##### Ubuntu 20.04 / 22.04 / 24.04 (推荐)

```shell
# 1. 安装图形界面 GUI 依赖 (推荐 PyQt6)
pip install pyqt6

# 2. 安装 Ultralytics 及基础库
pip install ultralytics opencv-python pillow pyyaml matplotlib requests

# 3. 安装 ONNX 导出与推理依赖 (注意：切勿同时安装 onnxruntime 与 onnxruntime-gpu)
pip install onnx onnxscript onnxruntime-gpu

# 4. 安装 TensorRT Engine 导出依赖 (仅需 GPU 导出 .engine 时安装)
pip install tensorrt-cu12
```

##### Ubuntu 24.04：运行 `python app.py` 前需安装的系统库

在 **Ubuntu 24.04** 上启动 GUI（`python app.py`）时，若出现 Qt / xcb / OpenGL / GLib 相关缺失库报错（例如 `Could not load the Qt platform plugin "xcb"`、`libxcb-cursor0`、`libGL` 等），请先安装下列系统依赖：

```shell
sudo apt-get update
sudo apt-get install -y \
  libxcb-cursor0 libxcb-util1 libxcb-xinerama0 \
  libgl1-mesa-dri libgl1 \
  libglib2.0-0t64 libsm6 libxext6 libxrender-dev libgomp1
```

安装完成后重新执行：

```shell
conda activate yolo
python app.py
```

##### Ubuntu 18.04 旧系统特别支持

```shell
# 由于 18.04 系统的 GLIBC 及核心库限制，需要使用限定版本的依赖组合：
# 1. 安装基础依赖（限制 numpy<2 避免 ONNX Runtime 崩溃）
pip install "numpy<2" \
            "onnxruntime-gpu==1.15.1" \
            pyqt5 pillow pyyaml matplotlib requests \
            onnx onnxscript --no-cache-dir

# 2. 安装 ultralytics 并补充独立依赖（避免自动回滚 OpenCV 版本）
pip install ultralytics --no-deps
pip install psutil polars nvidia-ml-py ultralytics-thop

# 3. 强制指定 OpenCV 5.x 无 GUI 版本（彻底规避 xcb 崩溃）
pip install "opencv-python-headless>=5.0.0" --no-cache-dir

# 4. TensorRT Engine 导出依赖 (适用旧版 SM 架构)
pip install "tensorrt<9.0.0" --extra-index-url https://pypi.nvidia.com

# 5. 拉回 numpy 版本，否则会报错
pip install "numpy<2.0.0"
```

> **提示 (Ubuntu)**：
>
> - **Ubuntu 20.04 / 22.04**：若启动界面时提示缺失 OpenCV 动态库，可补全：
>   ```shell
>   sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
>   ```
> - **Ubuntu 24.04**：请优先按上文「Ubuntu 24.04：运行 `python app.py` 前需安装的系统库」一节安装完整 xcb / OpenGL / GLib 依赖（包名含 `libglib2.0-0t64` 等，与 22.04 不同）。

#### 选项 B：Windows 环境安装 (Windows 10 / 11)

```powershell
:: 1. 安装图形界面 GUI 依赖 (推荐 PyQt6，亦可安装 pyqt5)
pip install pyqt6

:: 2. 安装 Ultralytics 及基础依赖
pip install ultralytics opencv-python pillow pyyaml matplotlib requests

:: 3. 安装 ONNX 导出与推理依赖 (注意：切勿同时安装 onnxruntime 与 onnxruntime-gpu)
pip install onnx onnxscript onnxruntime-gpu

:: 4. TensorRT Engine 导出依赖 (可选：仅 RTX 系列显卡且需导出 .engine 时安装)
pip install tensorrt
```

## 预训练模型管理机制

运行时下载或调用的所有官方预训练权重文件（如 `yolo11s-seg.pt`、`yolov8s.pt` 等），系统将**自动在项目根目录下创建 `pre_trained_model/` 文件夹**并集中下载存放，保持根目录极其清洁。

## 快速启动与 6 步骤 UI 使用 SOP

### 启动 GUI 界面

```shell
conda activate yolo
python app.py
```

也可以通过 `python train.py --gui` 启动 GUI 界面。

> **Ubuntu 24.04**：若 `python app.py` 因 Qt/xcb/OpenGL 报错无法启动，请先安装系统库（见上文「Ubuntu 24.04：运行 `python app.py` 前需安装的系统库」）。  
> **RTX 5090**：若训练时报 `no kernel image is available for execution on the device`，请改用 `cu128` 版 PyTorch（见上文「安装 PyTorch 与 CUDA 支持」选项 C）。

UI 界面采用 6 步骤单窗口引导流：

1. **Step 1: Data（数据准备与切分）**
   - 选择任务类型（如 `Segment`），指定数据源目录，点击 `1. Organize Origin Dataset` 生成标准化原始数据集，点击 `2. Split Dataset & Generate YAML` 划分训练/验证/测试集并自动生成 `data_<task>.yaml`。
2. **Step 2: Config（模型配置与超参）**
   - 选择 YOLO 版本与规模 (`n`/`s`/`m`/`l`/`x`)，配置 Epochs、Batch Size、ImgSz、Device、Optimizer，展开并微调高级超参（学习率、动量、权重衰减、Warmup 等）、数据增强超参（Mosaic, Copy-Paste, Random Erasing, FlipUD, FlipLR 等）及损失函数增益。
3. **Step 3: Train（训练与实时监控）**
   - 点击 `🚀 Start Training` 启动训练。实时查看数值进度条、耗时计时与训练过程产生的动态指标图表（点击可放大）。支持随时通过 `⏹ Force Stop` 终止进程。
4. **Step 4: Export（模型导出）**
   - 选择权重文件与导出格式（`onnx` / `engine` / `torchscript`），指定自定义保存路径（导出的模型将**仅存放在你指定的路径**下，源 `weights/` 目录不留副本）。
5. **Step 5: Inference（模型推理与自动存储）**
   - 点击 `1. Initialize Model` 初始化载入权重，点击 `2. Run Inference` 执行预测。支持单图或整个图片文件夹批量推理，渲染后的预测图像将**自动创建 `results/` 文件夹并按原图名称存储**。支持鼠标拖拽平移与以光标为中心缩放。
6. **Step 6: Mask Compare（掩膜对比与精度评估）**
   - 指定 Ground Truth 标签目录（支持 Labelme JSON 与 YOLO TXT 标注）与推理结果/模型路径，点击 `🔍 Run Mask Comparison & Evaluation` 计算掩膜与预测轮廓。可视化对比图（**GT = 绿色轮廓, Inference = 红色轮廓**）自动输出至 `compare/` 目录，并按类别生成 Precision@50、Recall@50、mask-mAP50、mask-mAP50-95、Mean IoU 统计表。

---

## 独立命令行脚本 (CLI Minimal Units) 使用 SOP

为了保持代码高内聚、低耦合，GUI 界面上的所有功能及参数（包括 Data / Config / Train / Export / Inference / Mask Compare 六大标签页）均已 1:1 对齐拆分为独立的命令行脚本通道：

### 1. Data 模块：数据准备与切分

#### 1.1 整理原始数据 (`prepare_origin.py`)

*将原始标注图片及标签自动归一化整理为标准 Origin_dataset 目录结构*

```bash
python prepare_origin.py \
  --source-dir /path/to/raw_dataset \
  --task segment \
  --classes-file /path/to/classes.txt \
  --project-root .
```

- `--source-dir`: 原始图像与标注文件所在目录（必填）
- `--task`: 视觉任务类型 (`detect`, `segment`, `classify`, `pose`, `obb`，默认 `segment`)
- `--classes-file`: 可选类别定义文件路径 (`classes.txt`)
- `--no-force`: 保留既有 Origin_dataset，不清空重建
- `--project-root`: 指定项目根目录（可选）

#### 1.2 切分数据集并生成 YAML (`split_dataset.py`)

*按照设定比例随机划分 Train / Val / Test 数据集，并自动生成训练用 `.yaml` 配置文件*

```bash
python split_dataset.py \
  --origin-dir data_sets/Origin_dataset \
  --train-dir data_sets/Train_dataset \
  --task segment \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --classes "cat,dog,car" \
  --seed 0
```

- `--origin-dir`: 原始数据集目录 (默认 `data_sets/Origin_dataset`)
- `--train-dir`: 训练数据集输出目录 (默认 `data_sets/Train_dataset`)
- `--task`: 视觉任务类型 (`detect`, `segment`, `classify`, `pose`, `obb`)
- `--train-ratio`: 训练集划分比例（默认 `0.7`）
- `--val-ratio`: 验证集划分比例（默认 `0.2`，测试集自动计算为 `1.0 - train - val`）
- `--classes`: 逗号分隔的类别列表（可选，默认自动读取 `classes.txt`）
- `--seed`: 随机切分种子（默认 `0`）

>**💡 传参规则与特殊参数说明：**
>
>1. *`--seed` 随机切分种子**：控制划分训练集/验证集/测试集时打乱数据集顺序的随机数种子（`random.seed(seed)`）。客户端 GUI 界面为了简化交互固定使用了默认种子（`0`），而在 CLI 命令行中暴露该参数，是为了方便用户固定随机种子以获得**完全可复现的数据集划分结果**。

---

### 2. Config & Train 模块：启动模型训练 (`train.py`)

*客户端训练及 Config/Train 界面中包含的所有基础参数与高级参数已全量追加至 `train.py` 命令行中，支持命令行直接进行深度配置：*

```bash
python train.py \
  --task segment \
  --data data_sets/Train_dataset/data_segment.yaml \
  --model yolo11l-seg.pt \
  --yolo-version yolo11 \
  --model-size l \
  --epochs 1000 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --optimizer auto \
  --workers 8 \
  --patience 100 \
  --close-mosaic 10 \
  --amp \
  --multi-scale \
  --cos-lr \
  --single-cls \
  --classes "" \
  --lr0 0.0001 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight-decay 0.0005 \
  --warmup-epochs 3.0 \
  --warmup-momentum 0.8 \
  --warmup-bias-lr 0.1 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --degrees 0.0 \
  --translate 0.1 \
  --scale 0.5 \
  --shear 0.0 \
  --perspective 0.0 \
  --mosaic 0.0 \
  --copy-paste 0.3 \
  --erasing 0.4 \
  --flipud 0.2 \
  --fliplr 0.5 \
  --dropout 0.0 \
  --fraction 1.0 \
  --box 7.5 \
  --cls 0.5 \
  --dfl 1.5 \
  --pose 12.0 \
  --kobj 1.0 \
  --rect \
  --name cow \
  --project 20260729
```

> **💡 传参规则与特殊参数说明：**
> 1. **开关/布尔标志参数 (Flag)**：在 Python CLI (`argparse`) 标准中，所有 `flag` 类型的参数（对应客户端中的复选框 Enable ** 勾选）**无需且切勿额外传递 bool 值（即不要写 `--rect True` 或 `--multi-scale True`）**。在命令行中**只需直接写出参数名（如 `--rect`、`--multi-scale`、`--cos-lr`、`--single-cls`）即代表启用/勾选该功能（`True`）**；若不写该参数则保持默认值（`False`）。对于默认启用的参数（如 `--amp`），若需关闭则写 `--no-amp`。

#### `train.py` 全量命令行参数详解表

| 参数分类 | 参数名称 | 类型 | 默认值 | 参数说明 |
| :--- | :--- | :--- | :--- | :--- |
| **基础配置** | `--task` | str | `segment` | 视觉任务类型 (`detect`, `segment`, `classify`, `pose`, `obb`) |
| | `--data` | str | `""` | 数据集 YAML 描述文件路径 (默认自动推导 `data_<task>.yaml`) |
| | `--model` | str | `""` | 基础模型权重文件名或绝对路径 (例如 `yolo11s-seg.pt`) |
| | `--yolo-version`| str | `yolo11` | YOLO 架构版本 (`yolo11`, `yolov8`, `yolo26`) |
| | `--model-size` | str | `s` | 模型规模等级 (`n`, `s`, `m`, `l`, `x`) |
| | `--epochs` | int | `1000` | 总训练轮数 Epochs |
| | `--batch` | int | `8` | Batch Size 批次大小 |
| | `--imgsz` | int | `640` | 输入图像分辨率尺寸 |
| | `--device` | str | `0` | 计算设备 (`0`, `cuda`, `cpu`) |
| | `--optimizer` | str | `auto` | 优化器类型 (`auto`, `SGD`, `Adam`, `AdamW`, `RMSProp`) |
| | `--workers` | int | `8` | 数据加载多线程数 |
| | `--patience` | int | `100` | 早停机制 Patience 轮数 (无提升自动停止) |
| | `--close-mosaic` | int | `10` | 最后 N 个 Epoch 关闭 Mosaic 增强以稳定收敛 |
| | `--amp` / `--no-amp` | flag | `True` | 是否开启自动混合精度 (AMP) 训练 |
| | `--multi-scale` | flag | `False` | 是否开启多尺度训练 (Multi-Scale) |
| | `--cos-lr` | flag | `False` | 是否开启余弦退火学习率调度器 |
| | `--single-cls` | flag | `False` | 是否开启单类别训练模式 |
| | `--classes` | str | `""` | 指定训练类别 ID 列表（逗号分隔，如 `0,1`；留空代表训练数据集中所有类别） |
| | `--rect` | flag | `False` | 是否开启矩形训练 (Rectangular Training) |
| **高级优化超参** | `--lr0` | float | `0.004` | 初始学习率 (Initial Learning Rate) |
| | `--lrf` | float | `0.01` | 最终学习率比例 (Final LR Ratio) |
| | `--momentum` | float | `0.937` | 优化器动量 (Momentum) |
| | `--weight-decay` | float | `0.0005` | 权重衰减系数 (Weight Decay) |
| | `--warmup-epochs`| float | `3.0` | 预热 Epochs 轮数 |
| | `--warmup-momentum`| float| `0.8` | 预热初始动量 |
| | `--warmup-bias-lr`| float | `0.1` | 预热偏置学习率 |
| **数据增强超参** | `--hsv-h` | float | `0.015` | 色调 (HSV-Hue) 数据增强幅度 |
| | `--hsv-s` | float | `0.7` | 饱和度 (HSV-Saturation) 数据增强幅度 |
| | `--hsv-v` | float | `0.4` | 明度 (HSV-Value) 数据增强幅度 |
| | `--degrees` | float | `0.0` | 旋转角度限制 |
| | `--translate` | float | `0.1` | 平移增益比例 |
| | `--scale` | float | `0.5` | 缩放增益比例 |
| | `--shear` | float | `0.0` | 剪切角度 |
| | `--perspective` | float | `0.0` | 透视变换幅度 |
| | `--mosaic` | float | `0.0` | Mosaic 增强概率 |
| | `--copy-paste` | float | `0.3` | Copy-Paste 复制粘贴增强概率 |
| | `--erasing` | float | `0.4` | Random Erasing 随机擦除概率 |
| | `--flipud` | float | `0.2` | 上下翻转 (FlipUD) 概率 |
| | `--fliplr` | float | `0.5` | 左右翻转 (FlipLR) 概率 |
| | `--dropout` | float | `0.0` | Dropout 丢弃率 |
| | `--fraction` | float | `1.0` | 使用数据集的比例 (0.01~1.0) |
| **损失函数权重** | `--box` | float | `7.5` | 边界框损失权重 (Box Loss Gain) |
| | `--cls` | float | `0.5` | 类别损失权重 (Cls Loss Gain) |
| | `--dfl` | float | `1.5` | DFL 损失权重 (DFL Loss Gain) |
| | `--pose` | float | `12.0` | 姿态损失权重 (Pose Loss Gain) |
| | `--kobj` | float | `1.0` | 关键点目标损失权重 (Kobj Loss Gain) |
| **运行与控制** | `--name` | str | `train` | 训练实验输出子目录名称 |
| | `--project` | str | `""` | 训练日志与权重保存总根目录（runs/segmentation/**） |
| | `--resume` | flag | `False` | 从上次中断的 checkpoint 继续训练 |
| | `--gui` | flag | `False` | 启动图形界面 GUI |

---

### 3. Export 模块：模型格式导出 (`export.py`)

*将训练好的 `.pt` PyTorch 权重文件导出为 ONNX, TensorRT Engine 或 TorchScript 部署格式。导出的模型仅存放在指定的 `--output` 路径下，保持输出干净明确。*

```bash
python export.py \
  --model runs/segment/train/weights/best.pt \
  --task segment \
  --format onnx \
  --output models/best.onnx \
  --opset 17 \
  --dynamic \
  --simplify \
  --device cpu
```

- `--model`: 输入的 PyTorch `.pt` 权重文件路径（必填）
- `--task`: 视觉任务类型 (`detect`, `segment`, `classify`, `pose`, `obb`)
- `--format`: 目标导出格式 (`onnx`, `engine`, `torchscript`，默认 `onnx`)
- `--output`: 最终导出的模型文件绝对/相对路径（可选，默认自动存放至 `models/` 目录）
- `--opset`: ONNX opset 算子集版本（默认 `17`）
- `--dynamic`: 是否启用动态 Batch/Shape 维度
- `--simplify`: 是否对 ONNX 模型结构进行图优化简化
- `--device`: 导出计算设备 (如 `cpu` 或 `0`)

---

### 4. Inference 模块：推理预测 (`predict.py`)

*支持对单张图片或整个图片文件夹进行批量推理预测，支持直接载入 `.pt` / `.onnx` / `.engine` / `.torchscript` 格式。渲染结果自动保存至 `results/` 目录并保持原图文件名。*

```bash
python predict.py \
  --model models/best.onnx \
  --source /path/to/test_images \
  --conf 0.25 \
  --iou 0.45 \
  --imgsz 640 \
  --device cpu \
  --classes-file classes.txt \
  --quiet
```

- `--model`: 模型文件路径 (`.pt`, `.onnx`, `.engine`, `.torchscript`，必填)
- `--source`: 待推理的单图路径或包含图片的文件夹路径（必填）
- `--conf`: 置信度过滤阈值 Confidence Threshold（默认 `0.25`）
- `--iou`: 非极大值抑制 NMS IOU 阈值（默认 `0.45`）
- `--imgsz`: 推理输入图像尺寸（默认 `640`）
- `--device`: 推理设备 (`0`, `cuda`, `cpu`)
- `--classes-file`: 可选自定义类别映射文件 (`classes.txt`)
- `--quiet`: 静默模式，不输出逐张图片的日志详细打印

---

### 5. Mask Compare 模块：掩膜对比与精度评估 (`compare_mask.py`)

*对应 GUI 上的 **Mask Compare** 标签页，支持直接对比 Ground Truth 真实标注（Labelme JSON 或 YOLO TXT）与模型的推理预测输出，并在图像上绘制可视化对比（**GT 绿框/绿胶膜，Inference 红框/红胶膜**），自动计算输出包含 Precision@50, Recall@50, mask-mAP50, mask-mAP50-95, Mean IoU 的定量评估报告。*

```bash
python compare_mask.py \
  --gt-dir data_sets/Origin_dataset \
  --infer-source models/best.onnx \
  --images-dir data_sets/Origin_dataset \
  --classes-file classes.txt \
  --conf 0.25 \
  --output-dir compare/
```

- `--gt-dir`: Ground Truth 真实标注目录路径 (支持 Labelme JSON 或 YOLO TXT，必填)
- `--infer-source`: 推理结果路径，可为预测保存目录（含 TXT/JSON）或直接指定模型权重路径 (`.pt`, `.onnx`, `.engine` 等)（必填）
- `--images-dir`: 原始图像目录（可选，若未指定则自动从 GT 或 Inference 目录搜索对应图片）
- `--classes-file`: 类别定义 txt 文件路径（可选）
- `--conf`: 模型推理运行时的置信度阈值（默认 `0.25`）
- `--output-dir`: 可视化对比渲染图像的输出文件夹（默认自动创建并保存至 `compare/` 目录）

---

## 目录结构

```shell
.
├── app.py                     # GUI 主界面启动入口
├── prepare_origin.py          # 1. Data 模块: 整理原始数据独立脚本
├── split_dataset.py           # 2. Data 模块: 切分数据集与生成 YAML 独立脚本
├── train.py                   # 3. Config/Train 模块: 启动模型训练独立脚本 (可带 --gui 启动 GUI)
├── export.py                  # 4. Export 模块: 模型格式导出独立脚本
├── predict.py                 # 5. Inference 模块: 模型推理预测独立脚本
├── compare_mask.py            # 6. Mask Compare 模块: 掩膜对比与精度评估独立脚本
├── ENVIRONMENT_SOP.md         # 详细环境搭建 SOP 指南
├── README.md                  # 说明文档
├── pre_trained_model/         # 统一存储官方预训练权重 (*.pt)
├── models/                    # 导出模型存储目录
├── results/                   # 推理预测渲染结果图像输出目录
├── compare/                   # 掩膜对比与评估渲染图像输出目录
├── data_sets/
│   ├── Origin_dataset/        # 整理后的原始 YOLO 数据
│   └── Train_dataset/         # 切分后的数据集与 YAML 描述文件
├── gui/                       # PyQt5/PyQt6 界面组件与样式系统
├── services/                  # 数据集服务、比较服务与后台训练进程管理
├── utils/                     # 通用 YOLO 训练器与数据处理工具集
└── runs/                      # 训练日志、实时指标图表输出
```

## 常见问题排查 (Troubleshooting)

1. **TensorRT 导出 Engine 报错 `Unsupported SM: 0x601`**
   - **原因**：显卡为 GTX 10xx 系列 (Pascal 架构, SM 6.1)，TensorRT 10+ 已停止支持 SM 6.x 架构。
   - **解决**：推荐导出 **ONNX (`.onnx`)** 或 **TorchScript (`.torchscript`)** 格式运行；若必须导出 `.engine`，请在 RTX 系列显卡环境下运行。
2. **ONNX 推理提示 CUDA 不可用 (Using CPU)**
   - **原因**：环境内同时安装了 `onnxruntime` 与 `onnxruntime-gpu` 冲突。
   - **解决**：执行 `pip uninstall -y onnxruntime && pip install --force-reinstall onnxruntime-gpu`。
3. **Ubuntu 18.04 缺失 OpenCV 动态库 `libGL.so.1`**
   - **解决**：执行 `sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0`。
