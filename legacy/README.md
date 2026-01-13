# PRML 课程项目：基于模仿学习的机械臂基本操作

> ⚠️ **项目失败说明**：尽管模型在训练集上的交叉熵损失能够降至0.02，但测试时仍无法完成任务。训练损失低不代表模型在实际执行时能够完成精确的机械臂操作任务。

## 项目简介

本项目旨在通过模仿学习训练机械臂完成复杂的操作任务。

我们使用 PyBullet 仿真环境搭建了包含 Panda 机械臂、可交互刚体棒和目标点的仿真场景，通过人工采集数据，并使用Transformer encoder学习时序策略，实现机械臂自主完成物体连接和目标任务。

### 任务描述

机械臂需要完成以下子任务：

1. **连接两根刚体棒**：将两根独立的刚体棒 A 和 B 正确连接
2. **抓取组合棒**：抓取连接后的组合棒
3. **刺向目标**：使用组合棒的一端刺向墙上的目标点

若使用 `--easy` 参数，则只有连接两根刚体棒的任务

## 环境配置

### 系统要求

- Python 3.13.11
- CUDA 13.0+（如需使用GPU）
- Linux/Windows/macOS

### 安装步骤

1. **创建conda环境**

```bash
conda create -n prml python=3.13.11
conda activate prml
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

### 依赖项

主要依赖包括：
- PyTorch 2.9.1 (with CUDA 13.0 support)
- PyBullet 3.2.6
- NumPy, Scikit-learn, Matplotlib
- 其他辅助库（详见 requirements.txt）

## 快速开始

### 1. 数据采集

使用键盘鼠标控制机械臂完成任务，采集专家演示数据。

**启动数据采集：**
```bash
# 采集训练数据
python collect_data.py -t train

# 采集验证数据
python collect_data.py -t val
```

**键盘控制说明：**

| 按键 | 功能 |
|------|------|
| `C/Z` | 末端执行器上下移动 |
| `方向键` | 末端执行器左右/前后移动 |
| `J/L` | 绕X轴旋转（Roll） |
| `I/K` | 绕Y轴旋转（Pitch） |
| `U/O` | 绕Z轴旋转（Yaw） |
| `Space` | 闭合夹爪 |
| `B` | 张开夹爪 |
| `1/2/3` | 切换视角（前/顶/侧） |
| `ESC` | 退出（不保存） |

**可选参数：**
```bash
-d, --debug              显示调试信息
-r, --randomize          随机化两根棒的初始位置与目标点位置
-b, --show-boundary      显示边界标记
--hard                   困难模式（棒可能平躺，墙的位置也会随机化）
```

**注意事项：**
- 只有成功击中目标的轨迹才会自动保存
- 每条轨迹至少需要10步
- 最大步数限制为5000步
- 轨迹保存至 `data/train/` 或 `data/val/` 目录

### 2. 轨迹回放

查看和验证采集的轨迹数据。

**回放单条轨迹：**
```bash
python replay.py -f data/train/trajectory_001.pkl
```

**回放所有训练轨迹：**
```bash
python replay.py -t train -a
```

**可选参数：**
```bash
-s, --speed SPEED        播放速度倍数（默认：1.0）
-v, --view VIEW          初始视角（front/top/side）
-d, --debug              显示调试信息
-b, --show-boundary      显示边界标记
--no-render              不渲染
```

### 3. 模型训练

使用采集的数据训练 Transformer 策略网络。

**基础训练：**
```bash
python train.py
```
可编辑 train.sh

**主要参数：**
```bash
--epochs EPOCHS          训练轮数（默认：100）
--batch-size BATCH_SIZE  批次大小（默认：8）
--lr LR                  学习率（默认：1e-3）
--d-model D_MODEL        D_MODEL维度（默认：128）
--nhead NHEAD            注意力头数（默认：8）
--num-layers NUM_LAYERS  Transformer层数（默认：4）
--dropout DROPOUT        Dropout率（默认：0.1）
--save-dir SAVE_DIR      模型保存目录（默认：checkpoints）
--device DEVICE          设备（cuda/cpu）
--early-stopping-patience PATIENCE  早停耐心值（默认：15）
--use-class-weights      使用类别权重处理类别不平衡
--entropy-weight WEIGHT  熵正则化权重（默认：0.01）
--easy                   简易模式（仅使用xyz移动+夹爪控制）
```

**训练输出：**
- `checkpoints/best_model.pth` - 最佳验证损失模型
- `checkpoints/final_model.pth` - 最终训练模型
- `checkpoints/training_curves.png` - 交叉熵损失曲线可视化

### 4. 模型测试

在随机化环境中测试训练好的模型。

**基础测试：**
```bash
python test.py --model-path checkpoints/best_model.pth
```
可编辑 test.sh

**主要参数：**
```bash
--episodes EPISODES      测试回合数（默认：10）
--max-steps MAX_STEPS    每回合最大步数（默认：5000）
--view VIEW              视角（front/top/side）
--randomize              随机化初始位置
--hard                   困难模式
--debug                  显示调试信息
--show-boundary          显示边界标记
--speed SPEED            仿真速度（默认：1.0）
--temperature TEMP       Softmax温度（默认：1.0）
--no-op-threshold THRESH  连续无操作阈值（默认：50）
--save-dir SAVE_DIR      结果保存目录
--all-modes              测试所有环境模式
```


## 项目结构

```
PRML/
├── README.md                 # 项目文档
├── requirements.txt          # Python依赖
├── collect_data.py           # 数据采集脚本
├── replay.py                 # 轨迹回放脚本
├── train.py                  # 模型训练脚本
├── test.py                   # 模型测试脚本
├── train.sh                  # 训练脚本（可选）
├── test.sh                   # 测试脚本（可选）
├── data/                     # 数据目录
│   ├── train/                # 训练集
│   │   └── trajectory_*.pkl
│   └── val/                  # 验证集
│       └── trajectory_*.pkl
├── model/                    # 模型定义
│   ├── __init__.py
│   └── transformer_policy.py
├── envs/                     # 环境定义
│   ├── __init__.py
│   ├── arm_env.py            # 机械臂环境
│   └── simple_env.py         # 简化环境
├── checkpoints/              # 模型权重（训练后生成）
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_curves.png
└── test_env_results/         # 测试结果（测试后生成）
    ├── *_results.pkl
    ├── all_results.pkl
    └── success_rates.png
```

## 技术细节

### 观测空间

观测包含以下68维信息：

| 索引范围 | 维度 | 说明 |
|---------|------|------|
| *0 | 1 | 连接状态（0/1） |
| 1-9 | 9 | 关节位置（7手臂+2夹爪） |
| 10-18 | 9 | 关节速度 |
| 19-21 | 3 | 末端执行器位置 |
| 22-25 | 4 | 末端执行器姿态（四元数） |
| 26-38 | 13 | 棒A状态（位置3+姿态4+速度3+角速度3） |
| 39-51 | 13 | 棒B状态 |
| *52-64 | 13 | 组合棒状态 |
| *65-67 | 3 | 目标位置 |

若使用 `--easy` 则不使用带 `*` 的状态。

### 动作空间

动作包含7个连续值，通过离散化为3类：

| 索引 | 含义 | 离散化方式 |
|------|------|-----------|
| 0-2 | 末端XYZ移动 | -1/0/1 × pos_speed |
| *3-5 | 末端姿态旋转 | -1/0/1 × rot_speed |
| 6 | 夹爪控制 | -1(闭合)/0(保持)/1(张开) |

若使用 `--easy` 则不使用带 `*` 的动作。
