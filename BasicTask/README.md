# PRML - Panda机械臂刚体棒抬升项目

## 项目简介

本项目实现了一个模仿学习系统，用于训练 Panda 机械臂将一根 0.2m 长的刚体棒抬升到目标高度（0.5m），同时保持刚体棒在以原点为圆心的1m半径范围内。系统使用键盘控制收集数据，并通过 Transformer Encoder 策略网络进行模仿学习。

## 项目结构

```
.
├── envs/
│   ├── __init__.py           
│   ├── simple_env.py         # 简化环境（仅含刚体棒）
│   └── arm_env.py            # 机械臂环境（Panda + 刚体棒）
├── model/
│   ├── __init__.py           
│   └── transformer_policy.py # Transformer Encoder 策略网络
├── data/
│   ├── train/                # 训练数据目录
│   └── val/                  # 验证数据目录
├── checkpoints/              # 模型权重
├── collect_data.py           # 数据收集脚本
├── replay.py                 # 轨迹回放脚本
├── train.py                  # 训练脚本
├── train_nr.sh               # 训练 Shell 脚本，刚体棒位置固定为 (0, 0.3)
├── train_r.sh                # 训练 Shell 脚本，刚体棒位置在范围内随机
├── test.py                   # 测试脚本
├── test_nr.sh                # 测试 Shell 脚本，刚体棒位置固定为 (0, 0.3)
└── test_r.sh                 # 测试 Shell 脚本，刚体棒位置在范围内随机
```

## 任务描述

**主要任务：** 控制Panda机械臂抓取一根 0.2m 长的刚体棒，并将其中心抬升至 0.5m 高度。

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.0+（如需使用GPU加速）
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
- PyTorch 2.0+ (with CUDA support)
- PyBullet 3.2+
- NumPy, Matplotlib, tqdm

## 快速开始

### 1. 数据采集

通过键盘控制机械臂收集训练数据。

**启动数据采集：**
```bash
# 采集一条训练数据
python collect_data.py -t train

# 采集一条验证数据
python collect_data.py -t val
```

**键盘控制说明：**

| 按键 | 功能 |
|------|------|
| `C/Z` | 末端执行器上升/下降 |
| `方向键` | 末端执行器移动（前后/左右） |
| `Space` | 闭合夹爪 |
| `B` | 张开夹爪 |
| `1/2/3` | 切换视角（前/顶/侧） |
| `ESC` | 退出（不保存） |

**可选参数：**
```bash
-t, --type TYPE          数据集类型（train/val）
-d, --debug              显示调试信息
-r, --randomize          随机化刚体棒初始位置
-b, --show-boundary      显示边界标记
```

**操作建议：**
1. 使用方向键将机械臂移动到刚体棒上方
2. 使用Z键下降到适当位置
3. 使用Space键闭合夹爪抓取刚体棒
4. 使用C键将刚体棒抬升至0.5m高度
5. 成功后会自动保存轨迹

### 2. 轨迹回放

查看和验证采集的轨迹数据。

**回放单条轨迹：**
```bash
python replay.py --file data/train/trajectory_001.pkl
```

**回放所有训练轨迹：**
```bash
python replay.py --type train --all
```

**可选参数：**
```bash
-f, --file FILE          指定轨迹文件
-a, --all                回放指定类型的所有轨迹
-t, --type TYPE          轨迹类型（train/val）
-s, --speed SPEED        播放速度倍数（默认：1.0）
-d, --debug              显示调试信息
-b, --show-boundary      显示边界标记
--no-render              不渲染（仅统计数据）
```

### 3. 模型训练

使用采集的数据训练Transformer策略网络。

**基础训练：**
```bash
python train.py --use-class-weights
```

**或使用Shell脚本：**
```bash
bash train_nr.sh
bash train_r.sh
```

**主要参数：**
```bash
--data-dir DATA_DIR      数据目录路径（默认：data）
--train-dir TRAIN_DIR    训练数据目录
--val-dir VAL_DIR        验证数据目录
--epochs EPOCHS          训练轮数（默认：100）
--batch-size BATCH_SIZE  批次大小（默认：8）
--lr LR                  学习率（默认：1e-3）
--d-model D_MODEL        模型维度（默认：128）
--nhead NHEAD            注意力头数（默认：8）
--num-layers NUM_LAYERS  Transformer层数（默认：4）
--dim-feedforward DIM_FF 前馈网络维度（默认：512）
--dropout DROPOUT        Dropout率（默认：0.1）
--max-seq-len MAX_LEN    最大序列长度（默认：2000）
--no-pad                 禁用padding
--seed SEED              随机种子（默认：42）
--save-dir SAVE_DIR      模型保存目录（默认：checkpoints）
--device DEVICE          设备（cuda/cpu，默认：cuda）
--early-stopping-patience PATIENCE  早停耐心值（默认：15）
--early-stopping-delta DELTA        最小改进阈值（默认：1e-6）
--obs-embed-hidden HIDDEN 观测嵌入隐藏层维度（默认：256）
--obs-embed-layers LAYERS 观测嵌入层数（默认：2）
--use-class-weights      使用类别权重处理类别不平衡
--entropy-weight WEIGHT  熵正则化权重（默认：0.01）
```

**训练输出：**
- `checkpoints/best_model.pth` - 在验证集上最佳的模型
- `checkpoints/final_model.pth` - 最终训练模型
- `checkpoints/checkpoint_epoch_*.pth` - 定期保存的检查点
- `checkpoints/training_curves.png` - 训练和验证损失曲线


**训练建议：**
- 首次训练建议使用 `--use-class-weights` 处理类别不平衡

### 4. 模型测试

在环境中测试训练好的模型。

**基础测试：**
```bash
python test.py --model-path checkpoints/best_model.pth
```

**或使用Shell脚本：**
```bash
bash test.sh
```

**主要参数：**
```bash
--model-path PATH        模型路径（必需）
--episodes EPISODES      测试回合数（默认：10）
--max-steps MAX_STEPS    每回合最大步数（默认：2000）
--view VIEW              视角（front/top/side，默认：front）
--randomize              随机化刚体棒初始位置
--debug                  显示调试信息
--show-boundary          显示边界标记
--speed SPEED            仿真速度倍数（默认：1.0）
--step-by-step           逐步执行模式（每步等待用户输入）
--no-render              不渲染（仅统计数据）
--save-dir SAVE_DIR      结果保存目录（默认：test_env_results）
--all-modes              测试所有环境模式（默认+随机化）
--seed SEED              随机种子
```

**测试输出：**
- 成功率统计
- 平均步数
- 达到目标高度率
- 边界违规率
- 详细的episode-by-episode结果日志
- 成功率可视化图表

## 技术细节

### 观测空间（38维）

| 索引范围 | 维度 | 说明 |
|---------|------|------|
| 0-6 | 7 | Panda机械臂7个关节位置 |
| 7-8 | 2 | 夹爪关节位置（左/右） |
| 9-17 | 9 | 所有关节速度（7手臂 + 2夹爪） |
| 18-20 | 3 | 末端执行器位置（x, y, z） |
| 21-24 | 4 | 末端执行器姿态（四元数qx, qy, qz, qw） |
| 25-27 | 3 | 刚体棒位置（x, y, z） |
| 28-31 | 4 | 刚体棒姿态（四元数qx, qy, qz, qw） |
| 32-34 | 3 | 刚体棒线速度（vx, vy, vz） |
| 35-37 | 3 | 刚体棒角速度（ωx, ωy, ωz） |

### 动作空间（4维）

每个动作维度通过离散化为3类：

| 索引 | 含义 | 类别含义 |
|------|------|---------|
| 0 | X轴移动 | 0: 后退, 1: 保持, 2: 前进 |
| 1 | Y轴移动 | 0: 右移, 1: 保持, 2: 左移 |
| 2 | Z轴移动 | 0: 下降, 1: 保持, 2: 上升 |
| 3 | 夹爪控制 | 0: 闭合, 1: 保持, 2: 张开 |

### 网络架构

**Transformer Encoder 网络：**

1. **观测嵌入层**
   - 多层MLP将35维观测映射到d_model维度
   - 可配置隐藏层维度和层数

2. **Transformer编码器**
   - 多头自注意力机制
   - 位置编码
   - 可配置层数、注意力头数

3. **输出层**
   - 为每个动作维度输出 3 类 logits
   - 使用 argmax 选择动作

**训练策略：**
- 损失函数：交叉熵损失 + 熵正则化
- 优化器：AdamW
- 学习率调度：余弦退火
- 类别权重：处理类别不平衡

### 环境参数

**刚体棒：**
- 长度：0.2m
- 半径：0.02m
- 质量：0.25kg

**Panda机械臂：**
- 7自由度
- 二指夹爪
- 反动力学控制

**约束条件：**
- 目标高度：0.5m
- 边界半径：1.0m
- 最大步数：2000
