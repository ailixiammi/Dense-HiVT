# Argoverse 1.1 离线预处理脚本使用文档

## 📋 概述

`preprocess_offline.py` 是一个用于生成**稠密 Tensor** 格式的 Argoverse 1.1 预处理脚本。与原始 HiVT 的稀疏表示不同，此脚本生成固定维度的 Tensor，包含完整的语义特征，便于批处理和模型训练。

## 🎯 核心特性

### 1. 稠密 Tensor 表示
- **固定维度**：所有场景使用相同的 Tensor shape
- **Agent 维度**：`[MAX_AGENTS, T, 2]` (默认 64)
- **Map 维度**：`[MAX_LANES, MAX_POINTS, 2]` (默认 256×10)
- **批处理友好**：无需动态 padding

### 2. 丰富的语义特征

**Agent 特征：**
- 历史轨迹位置 `[64, 20, 2]`
- 未来轨迹位置 `[64, 30, 2]`
- 历史速度向量 `[64, 20, 2]`
- Agent 类型标签 `[64]` (0=AV, 1=AGENT, 2=OTHERS)
- Target 标记 `[64]` (是否为预测目标)
- 有效性 mask

**Map 特征：**
- Lane 中心线坐标 `[256, 10, 2]`
- Lane 是否在路口 `[256]`
- Lane 转向方向 `[256]` (0=NONE, 1=LEFT, 2=RIGHT)
- Lane 交通控制 `[256]` (是否有红绿灯等)
- 有效性 mask

### 3. 局部坐标系转换
- **原点**：AV 在 t=19 (第20帧) 的位置
- **朝向**：AV 的前进方向对齐到 X 轴正方向
- **速度**：也转换到局部坐标系

## 📦 输出格式

每个 `.pt` 文件包含一个字典：

```python
{
    # ===== Agent 特征 =====
    'agent_history_positions': Tensor[64, 20, 2],       # 历史位置
    'agent_history_positions_mask': Tensor[64, 20],     # 历史mask
    'agent_history_speed': Tensor[64, 20, 2],           # 历史速度
    'agent_future_positions': Tensor[64, 30, 2],        # 未来位置
    'agent_future_positions_mask': Tensor[64, 30],      # 未来mask
    'agent_type': Tensor[64],                           # Agent类型
    'agent_is_target': Tensor[64],                      # 是否为目标
    
    # ===== Map 特征 =====
    'map_lane_positions': Tensor[256, 10, 2],           # Lane点坐标
    'map_lane_positions_mask': Tensor[256, 10],         # Lane点mask
    'map_is_intersection': Tensor[256],                 # 是否路口
    'map_turn_direction': Tensor[256],                  # 转向
    'map_traffic_control': Tensor[256],                 # 交通控制
    
    # ===== Meta 信息 =====
    'origin': Tensor[2],                                # 原点 (全局坐标)
    'theta': Tensor[1]                                  # 旋转角度
}
```

## 🚀 快速开始

### 基本用法

```bash
python scripts/preprocess_offline.py \
    --data_dir /root/vc/data/train/data \
    --output_dir /root/vc/data/train/processed_dense
```

### 处理验证集

```bash
python scripts/preprocess_offline.py \
    --data_dir /root/vc/data/val/data \
    --output_dir /root/vc/data/val/processed_dense
```

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | **必需** | CSV 文件所在目录 |
| `--output_dir` | str | **必需** | 输出 .pt 文件的目录 |
| `--map_dir` | str | None | 地图文件目录 (可选) |
| `--radius` | float | 100.0 | 地图查询半径 (米) |
| `--max_agents` | int | 64 | 最大 Agent 数量 |
| `--max_lanes` | int | 256 | 最大 Lane 数量 |
| `--max_points` | int | 10 | 每条 Lane 的最大点数 |
| `--log_file` | str | None | 日志文件路径 (可选) |

## 📊 进阶用法

### 1. 自定义 Tensor 维度

根据您的数据集统计结果调整维度：

```bash
python scripts/preprocess_offline.py \
    --data_dir /root/vc/data/train/data \
    --output_dir /root/vc/data/train/processed_dense \
    --max_agents 80 \
    --max_lanes 300 \
    --max_points 15
```

### 2. 调整地图查询半径

```bash
# 更大的感受野
python scripts/preprocess_offline.py \
    --data_dir /root/vc/data/train/data \
    --output_dir /root/vc/data/train/processed_dense \
    --radius 150.0
```

### 3. 启用日志记录

```bash
python scripts/preprocess_offline.py \
    --data_dir /root/vc/data/train/data \
    --output_dir /root/vc/data/train/processed_dense \
    --log_file preprocess.log
```

### 4. 指定地图目录

如果地图文件不在默认位置：

```bash
python scripts/preprocess_offline.py \
    --data_dir /root/vc/data/train/data \
    --output_dir /root/vc/data/train/processed_dense \
    --map_dir /root/vc/data/maps
```

## 📁 输出目录结构

```
output_dir/
├── 1.pt
├── 2.pt
├── 3.pt
├── ...
└── 205942.pt
```

每个文件对应一个场景，文件名为场景 ID。

## 🔍 数据验证

### 加载并查看数据

```python
import torch

# 加载单个场景
sample = torch.load('output_dir/1.pt')

# 查看维度
print("Agent positions:", sample['agent_history_positions'].shape)
print("Map lanes:", sample['map_lane_positions'].shape)

# 查看有效 Agent 数量
num_valid_agents = sample['agent_history_positions_mask'][:, 0].sum()
print(f"Valid agents: {num_valid_agents}")

# 查看有效 Lane 数量
num_valid_lanes = sample['map_lane_positions_mask'][:, 0].sum()
print(f"Valid lanes: {num_valid_lanes}")
```

### 批量验证

```python
from pathlib import Path
import torch

output_dir = Path('output_dir')
pt_files = list(output_dir.glob('*.pt'))

print(f"Total scenes: {len(pt_files)}")

# 随机检查几个文件
import random
for pt_file in random.sample(pt_files, 5):
    sample = torch.load(pt_file)
    
    # 验证维度
    assert sample['agent_history_positions'].shape == (64, 20, 2)
    assert sample['map_lane_positions'].shape == (256, 10, 2)
    
    print(f"✓ {pt_file.name} passed")
```

## 🎓 与原始 HiVT 的差异

| 特性 | 原始 HiVT | 新 Dense 版本 |
|------|-----------|---------------|
| **表示方式** | 稀疏（动态长度） | 稠密（固定维度） |
| **Lane 格式** | 差分向量 | 完整点坐标 |
| **Agent 数量** | 动态 | 固定 64 |
| **Lane 数量** | 动态 | 固定 256 |
| **Lane 点数** | 动态 | 固定 10 |
| **批处理** | 需要动态 collate | 直接堆叠 |
| **语义特征** | Per-point | Per-lane |
| **存储开销** | 较小 | 较大 (~1-2MB/场景) |
| **计算效率** | GPU 利用率较低 | GPU 利用率更高 |

## ⚠️ 注意事项

### 1. 内存占用

- **单场景内存**：约 1-2 MB
- **处理时内存峰值**：约 500 MB - 1 GB
- **输出总大小**：训练集约 200-400 GB

### 2. 处理时间

- **速度**：约 10-20 场景/秒（取决于 Map API 查询）
- **训练集**：约 3-6 小时（205,942 个场景）
- **验证集**：约 30-60 分钟（39,472 个场景）

### 3. 截断策略

- 如果场景中 Agent 数量 > 64，**截断**前 64 个
- 如果场景中 Lane 数量 > 256，**截断**前 256 条
- 如果 Lane 点数 > 10，**均匀采样** 10 个点

### 4. Padding 策略

- Agent 填充：位置填0，mask填False，type填-1
- Lane 填充：位置填0，mask填False，语义特征填默认值

## 🐛 常见问题

### Q1: Map API 初始化失败

**错误信息：** `Map API 初始化失败`

**解决方案：**
```bash
# 方法1：设置环境变量
export ARGOVERSE_DATA_DIR=/root/vc/data

# 方法2：使用 --map_dir 参数
python scripts/preprocess_offline.py \
    --map_dir /root/vc/data/maps \
    ...
```

### Q2: 某些场景处理失败

**原因：** 数据不完整或格式异常

**解决方案：** 脚本会自动跳过失败的场景，在日志中记录。这是正常现象，不影响其他场景的处理。

### Q3: 输出文件已存在

**行为：** 脚本会自动跳过已存在的 .pt 文件

**如需重新处理：** 删除对应的 .pt 文件或整个输出目录

### Q4: 如何选择合适的维度参数？

**建议：**
1. 先运行数据统计脚本（如 `analyze_argoverse_stats.py`）
2. 根据 95% 或 99% 分位数设置 `--max_agents` 和 `--max_lanes`
3. 根据 Lane 点数分布设置 `--max_points`

## 📚 代码结构

```
scripts/preprocess_offline.py
│
├── GlobalConfig               # 全局配置类
├── CoordinateTransform        # 坐标转换工具
├── AgentFeatureExtractor      # Agent 特征提取器
├── MapFeatureExtractor        # Map 特征提取器
├── SceneProcessor             # 场景处理主类
└── main()                     # 主函数
```

## 🔗 相关工具

- **数据统计脚本**：`tools/analyze_argoverse_stats.py` - 统计 Agent 数量分布
- **地图统计脚本**：`tools/analyze_map_stats.py` - 统计 Lane 数量和点数分布

## 📝 更新日志

- **v1.0** (2026-02-17)
  - 初始版本
  - 支持稠密 Tensor 输出
  - 完整的语义特征提取
  - 自动 padding 和截断