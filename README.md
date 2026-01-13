# PRML - Panda机械臂刚体棒抬升项目

## 项目简介

本项目实现了一个强化学习系统，用于训练Panda机械臂将一根刚体棒抬升到目标高度，使用模仿学习结合Transformer策略网络。

## 项目结构

```
.
├── envs/
│   ├── __init__.py           
│   ├── simple_env.py         # 简化环境
│   └── arm_env.py            # 机械臂环境
├── model/
│   ├── __init__.py           
│   └── transformer_policy.py # Transformer编码器策略网络
├── data/
│   ├── train/                # 训练数据目录
│   └── val/                  # 验证数据目录
├── checkpoints/              # 模型检查点
├── collect_data.py           # 数据收集脚本
├── replay.py                 # 轨迹回放脚本
├── train.py                  # 训练脚本
├── train.sh                  # 训练Shell脚本
├── test.py                   # 测试脚本
└── test.sh                   # 测试Shell脚本
```

## 任务描述

机械臂需要抓取刚体棒，并将刚体棒（的中心）抬升至0.5m

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

通过键盘控制机器人收集训练数据。

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
| `Space` | 闭合夹爪 |
| `B` | 张开夹爪 |
| `1/2/3` | 切换视角（前/顶/侧） |
| `ESC` | 退出（不保存） |

**可选参数：**
```bash
-d, --debug              显示调试信息
-r, --randomize          随机化两根棒的初始位置与目标点位置
-b, --show-boundary      显示边界标记
```

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
--max-steps MAX_STEPS    每回合最大步数（默认：2000）
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

## 技术细节

### 观测空间

观测包含以下38维信息：

| 索引范围 | 维度 | 说明 |
|---------|------|------|
| 0-8 | 9 | 关节位置（7手臂+2夹爪） |
| 9-17 | 9 | 关节速度 |
| 18-20 | 3 | 末端执行器位置 |
| 21-24 | 4 | 末端执行器姿态（四元数） |
| 25-37 | 13 | 刚体棒状态（位置3+姿态4+速度3+角速度3） |

### 动作空间

动作包含4个连续值，通过离散化为3类：

| 索引 | 含义 | 离散化方式 |
|------|------|-----------|
| 0-2 | 末端XYZ移动 | -1/0/1 × pos_speed |
| 3 | 夹爪控制 | -1(闭合)/0(保持)/1(张开) |