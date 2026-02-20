# Dense-HiVT 训练脚本使用文档

## 📋 概述

`train.py` 是 Dense-HiVT 模型的训练脚本,实现了高效的端到端训练流程。该脚本针对 GPU 训练进行了优化,支持自动混合精度(AMP)、学习率预热、梯度裁剪等现代训练技术。

## 🎯 核心特性

### 1. 自动混合精度训练 (AMP)
- **Tensor Core 加速**：充分利用 RTX 系列 GPU 的 FP16 计算能力
- **GradScaler**：自动处理梯度缩放,防止下溢
- **内存优化**：相比 FP32 训练可节省约 40% 显存

### 2. 先进的优化策略

**优化器：AdamW**
- 解耦权重衰减,改善泛化性能
- 默认参数：lr=5e-4, weight_decay=1e-4

**学习率调度：Warmup + Cosine Annealing**
- **Phase 1 (Warmup)**：前 5 个 epoch 从 5e-6 线性增长到 5e-4
- **Phase 2 (Cosine)**：后续 epoch 余弦衰减到 1e-6
- 避免训练初期的梯度爆炸,后期精细调优

**梯度裁剪**
- Max Norm = 5.0
- 防止 Laplace NLL Loss 引起的梯度爆炸

### 3. 完善的训练监控

**TensorBoard 日志**
- 实时记录训练/验证损失
- 学习率曲线追踪
- 评测指标可视化 (minADE, minFDE, MR)

**终端进度条**
- 实时显示训练进度
- Loss 分解 (Reg Loss + Cls Loss)
- 当前学习率

### 4. 自动化模型管理

**Checkpoint 保存策略**
- `latest.pth`：每个 epoch 自动更新
- `best_dense_hivt.pth`：基于验证集 minFDE 保存最佳模型
- 完整保存训练状态(模型、优化器、调度器、Scaler)

## 🚀 快速开始

### 基本训练命令

```bash
python scripts/train.py \
    --train_dir /path/to/processed/train \
    --val_dir /path/to/processed/val \
    --output_dir outputs
```

### 使用示例数据路径

```bash
# 服务器环境
python scripts/train.py \
    --train_dir /root/devdata/Dense-HiVT/data/processed/train \
    --val_dir /root/devdata/Dense-HiVT/data/processed/val \
    --batch_size 64 \
    --epochs 64
```

## ⚙️ 参数说明

### 数据相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_dir` | str | 必需 | 训练集目录 (.pt 文件) |
| `--val_dir` | str | 必需 | 验证集目录 (.pt 文件) |
| `--output_dir` | str | `outputs` | 输出目录(保存在项目根目录) |

### 模型超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--embed_dim` | int | 128 | Transformer 嵌入维度 |
| `--num_heads` | int | 8 | Multi-Head Attention 头数 |
| `--num_local_encoder_layers` | int | 4 | Local Encoder 层数 |
| `--num_global_interactor_layers` | int | 3 | Global Interactor 层数 |
| `--num_decoder_layers` | int | 4 | Decoder 层数 |
| `--dropout` | float | 0.1 | Dropout 概率 |
| `--num_modes` | int | 6 | 多模态预测数量 |
| `--future_steps` | int | 30 | 预测未来时间步 (3秒) |

### 训练超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 64 | 训练总轮数 |
| `--batch_size` | int | 64 | 批次大小 |
| `--lr` | float | 5e-4 | 基础学习率 |
| `--lr_min` | float | 1e-6 | 最小学习率 (Cosine) |
| `--warmup_epochs` | int | 5 | 学习率预热轮数 |
| `--weight_decay` | float | 1e-4 | 权重衰减系数 |
| `--grad_clip_norm` | float | 5.0 | 梯度裁剪阈值 |
| `--use_amp` | bool | true | 使用自动混合精度 |

### DataLoader 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_workers` | int | 8 | 数据加载进程数 |
| `--pin_memory` | bool | true | 使用 Pinned Memory |
| `--prefetch_factor` | int | 2 | 预取批次数 |

## 📊 进阶用法

### 1. 调整批次大小

根据 GPU 显存调整 batch size：

```bash
# RTX 4090 (24GB) - 推荐
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --batch_size 64

# RTX 3090 (24GB)
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --batch_size 48

# RTX 3080 (10GB)
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --batch_size 24
```

### 2. 调整学习率策略

```bash
# 更激进的学习率
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --lr 1e-3 \
    --warmup_epochs 10

# 更保守的学习率
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --lr 1e-4 \
    --lr_min 1e-7 \
    --warmup_epochs 3
```

### 3. 长时间训练

```bash
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --epochs 128 \
    --lr 5e-4 \
    --lr_min 5e-7
```

### 4. 关闭 AMP (调试用)

```bash
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --use_amp false
```

### 5. 调整数据加载性能

```bash
# 高性能 NVMe SSD
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --num_workers 16 \
    --prefetch_factor 4

# 低速机械硬盘
python scripts/train.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --num_workers 4 \
    --prefetch_factor 1
```

## 📁 输出目录结构

```
outputs/
├── checkpoints/
│   ├── latest.pth              # 最新 Checkpoint
│   └── best_dense_hivt.pth     # 最佳模型 (基于 Val minFDE)
│
└── logs/
    ├── run_20260220_143052/    # TensorBoard 日志 (带时间戳)
    │   └── events.out.tfevents.*
    ├── run_20260220_180432/
    └── run_20260221_091523/
```

### Checkpoint 内容

```python
checkpoint = {
    'epoch': 当前轮数,
    'model_state_dict': 模型权重,
    'optimizer_state_dict': 优化器状态,
    'scheduler_state_dict': 学习率调度器状态,
    'scaler_state_dict': AMP Scaler 状态,
    'best_val_fde': 历史最佳 minFDE,
    'val_metrics': {
        'minADE': 平均位移误差,
        'minFDE': 最终位移误差,
        'MR': 错过率
    },
    'args': 训练参数配置
}
```

## 🔍 训练监控

### 1. 启动 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/logs --port 6006

# 在浏览器中打开
# http://localhost:6006
```

### 2. 关键指标解读

**训练指标 (Scalars)**

- `Train/Loss`：总损失 = Reg Loss + Cls Loss
- `Train/RegLoss`：回归损失 (Laplace NLL)
- `Train/ClsLoss`：分类损失 (交叉熵)
- `Train/LR`：当前学习率

**验证指标 (Scalars)**

- `Epoch/Val_minADE`：最小平均位移误差 (米)
  - 衡量整体轨迹预测精度
  - **目标**：< 1.0 米 (SOTA)
  
- `Epoch/Val_minFDE`：最小最终位移误差 (米)
  - 衡量 3 秒后的位置预测精度
  - **目标**：< 1.5 米 (SOTA)
  
- `Epoch/Val_MR`：错过率 (%)
  - FDE > 2.0 米的样本比例
  - **目标**：< 10% (SOTA)

**学习率曲线**

- `Epoch/LR`：学习率变化
  - Epoch 1-5：线性增长 (Warmup)
  - Epoch 6-64：余弦衰减

### 3. 训练曲线诊断

**正常训练曲线**
- Train Loss 平稳下降
- Val minFDE 在前 20 epoch 快速下降
- Epoch 30-50 进入平台期,缓慢优化

**过拟合信号**
- Train Loss 持续下降,Val minFDE 上升
- 解决方案：增大 dropout 或 weight_decay

**欠拟合信号**
- Train Loss 和 Val minFDE 都很高
- 解决方案：增大模型容量或训练轮数

**学习率过大**
- Loss 剧烈震荡
- 解决方案：降低 `--lr` 或增加 `--warmup_epochs`

## 💡 训练技巧

### 1. 学习率调优

**Warmup 的重要性**
- 防止训练初期梯度爆炸
- 给 BatchNorm 统计量稳定的初始化时间
- 建议 warmup_epochs = 总 epochs 的 5-10%

**Cosine Annealing 优势**
- 后期学习率逐渐降低,有助于收敛到更好的局部最优
- 避免 Step Decay 的突然下降导致的震荡

### 2. 批次大小权衡

**大批次 (64+)**
- ✅ 训练速度快
- ✅ 梯度估计更稳定
- ❌ 需要更多显存
- ❌ 可能泛化性能略差

**小批次 (16-32)**
- ✅ 显存占用低
- ✅ 可能泛化性能更好
- ❌ 训练速度慢
- ❌ 梯度噪声大

**建议**：尽量用大批次,但保持 `lr ∝ √batch_size`

### 3. AMP 使用建议

**何时使用 AMP**
- ✅ RTX 20/30/40 系列 GPU
- ✅ 正常训练场景
- ✅ 显存受限时

**何时关闭 AMP**
- ❌ 调试模型时 (避免精度问题干扰)
- ❌ 出现 NaN/Inf 时
- ❌ GTX 系列 GPU (无 Tensor Core)

### 4. 梯度裁剪

**为什么需要**
- Laplace NLL Loss 在 scale 参数接近 0 时梯度会爆炸
- 多模态预测中某些模式可能产生极端梯度

**调优建议**
- 默认 5.0 适用于大多数场景
- 如果仍有梯度爆炸,降低到 1.0-3.0
- 如果训练过于保守,增大到 10.0

### 5. 数据加载优化

**num_workers 设置**
- **CPU 核心数 ≥ 16**：`num_workers = 8-16`
- **CPU 核心数 8-16**：`num_workers = 4-8`
- **CPU 核心数 < 8**：`num_workers = 2-4`

**prefetch_factor**
- **NVMe SSD**：2-4 (I/O 不是瓶颈)
- **SATA SSD**：2
- **HDD**：1 (避免过度预取)

## ⚠️ 注意事项

### 1. 显存占用

**单卡显存需求**

| Batch Size | 模型参数 | 显存占用 (AMP) | 显存占用 (FP32) |
|------------|----------|----------------|-----------------|
| 16 | ~10M | ~6 GB | ~10 GB |
| 32 | ~10M | ~10 GB | ~18 GB |
| 64 | ~10M | ~18 GB | ~32 GB |

**显存不足解决方案**
1. 减小 `--batch_size`
2. 减小 `--embed_dim` (如 128 → 96)
3. 确保 `--use_amp` 开启
4. 减小 `--num_global_interactor_layers`

### 2. 训练时间估算

**训练速度** (RTX 4090 + Batch 64)
- 每个 epoch：~8-12 分钟
- 64 epochs：~9-13 小时

**影响因素**
- DataLoader 效率 (磁盘 I/O)
- GPU 利用率
- 验证集大小

### 3. 断点续训

从 Checkpoint 恢复训练：

```python
# 修改 train.py，在 TrainingEngine.__init__ 中添加：
if args.resume_from:
    checkpoint = torch.load(args.resume_from)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    self.current_epoch = checkpoint['epoch']
    self.best_val_fde = checkpoint['best_val_fde']
    print(f"✓ 从 Epoch {self.current_epoch} 恢复训练")
```

### 4. 多 GPU 训练

当前脚本仅支持单 GPU,如需多 GPU 训练,需要使用 `torch.nn.DataParallel` 或 `DistributedDataParallel` 包装模型。

## 🐛 常见问题

### Q1: CUDA Out of Memory

**错误信息：** `RuntimeError: CUDA out of memory`

**解决方案：**
```bash
# 方案 1: 减小批次大小
python scripts/train.py --batch_size 32

# 方案 2: 减小模型尺寸
python scripts/train.py --embed_dim 96 --num_global_interactor_layers 2

# 方案 3: 清空 GPU 缓存后重试
nvidia-smi  # 查看 GPU 占用
```

### Q2: Loss 出现 NaN

**原因：**
1. 学习率过大
2. 梯度爆炸
3. 数据异常 (包含 NaN/Inf)

**解决方案：**
```bash
# 降低学习率
python scripts/train.py --lr 1e-4

# 加强梯度裁剪
python scripts/train.py --grad_clip_norm 1.0

# 关闭 AMP (排查精度问题)
python scripts/train.py --use_amp false
```

### Q3: 训练速度慢

**原因分析：**
1. DataLoader 成为瓶颈 (CPU 或磁盘 I/O)
2. GPU 利用率低

**优化方案：**
```bash
# 增加数据加载进程
python scripts/train.py --num_workers 16 --prefetch_factor 4

# 检查 GPU 利用率
nvidia-smi dmon -i 0 -s u  # 实时监控
```

### Q4: 验证指标不收敛

**可能原因：**
1. 学习率过大或过小
2. 模型容量不足
3. 数据质量问题

**排查步骤：**
1. 检查 TensorBoard 中的学习率曲线
2. 对比训练集和验证集 Loss
3. 可视化预测结果 (使用 `eval.py`)

### Q5: 最佳模型没有保存

**原因：** 验证集 minFDE 从未低于初始的 `float('inf')`

**检查：**
```bash
# 查看训练日志
cat outputs/logs/run_*/events.out.tfevents.* | grep minFDE

# 手动检查 Checkpoint
python -c "import torch; ckpt = torch.load('outputs/checkpoints/latest.pth'); print(ckpt['val_metrics'])"
```

## 📚 代码结构

```python
scripts/train.py
│
├── TrainingEngine                    # 训练引擎主类
│   ├── __init__()                   # 初始化模型、优化器、调度器
│   ├── train_one_epoch()            # 单 Epoch 训练循环
│   ├── validate()                   # 验证集评估
│   ├── save_checkpoint()            # Checkpoint 保存
│   └── train()                      # 主训练循环
│
├── parse_args()                      # 命令行参数解析
└── main()                           # 入口函数
```

### TrainingEngine 核心流程

```
1. __init__
   ├── 初始化模型 (DenseHiVT)
   ├── 初始化损失函数 (DenseHiVTLoss)
   ├── 初始化优化器 (AdamW)
   ├── 初始化学习率调度器 (Warmup + Cosine)
   ├── 初始化 GradScaler (AMP)
   └── 创建输出目录和 TensorBoard Writer

2. train()
   └── for epoch in range(1, epochs+1):
       ├── train_one_epoch()
       │   └── for batch in train_loader:
       │       ├── 前向传播 (with autocast)
       │       ├── 计算损失
       │       ├── 反向传播 + 梯度裁剪
       │       └── 记录到 TensorBoard
       │
       ├── validate()
       │   └── for batch in val_loader:
       │       ├── 前向传播 (with autocast)
       │       ├── 计算评测指标 (minADE, minFDE, MR)
       │       └── 累积并返回平均值
       │
       ├── scheduler.step()         # 更新学习率
       ├── 记录 Epoch 级指标
       ├── 检查是否为最佳模型
       └── save_checkpoint()         # 保存模型
```

## 📈 性能基准

**参考性能** (Argoverse 1.1 验证集)

| Epoch | minADE (m) | minFDE (m) | MR (%) | 训练时长 |
|-------|------------|------------|--------|----------|
| 10    | ~1.8       | ~2.8       | ~25%   | ~1.5 小时 |
| 30    | ~1.2       | ~1.9       | ~15%   | ~5 小时 |
| 64    | ~0.9       | ~1.4       | ~9%    | ~10 小时 |

**SOTA 对比** (原始 HiVT)
- minADE: 0.90 m
- minFDE: 1.39 m
- MR: 8.1%

## 📝 训练日志示例

```
================================================================================
                           开始训练
================================================================================

总 Epochs: 64
训练集大小: 205942 样本
验证集大小: 39472 样本
Base LR: 0.0005
Warmup Epochs: 5 (从 5.00e-06 增长到 5.00e-04)
Min LR: 1e-06
Weight Decay: 0.0001
Gradient Clip Norm: 5.0
AMP 启用: True

================================================================================

Epoch 1/64 [Train] |█████████████| 3218/3218 [08:32<00:00, 6.28it/s]
Epoch 1/64 [Val]   |█████████████| 617/617 [01:24<00:00, 7.33it/s]

================================================================================
                         Epoch 1/64 总结
================================================================================

[训练]
  - Total Loss: 3.8542
  - Reg Loss:   3.2156
  - Cls Loss:   0.6386

[验证]
  - minADE: 2.1534 米
  - minFDE: 3.4782 米
  - MR:     35.24%

[优化器]
  - Learning Rate: 0.000100

🎉 新的最佳 minFDE: 3.4782 米

✓ 最佳模型已保存: outputs/checkpoints/best_dense_hivt.pth
  - minFDE: 3.4782 米

================================================================================
```

## 🔗 相关文档

- **数据预处理**：`0_README_preprocess_offline.md`
- **模型评估**：`2_README_val.md`
- **项目主文档**：`../README.md`

## 📝 更新日志

- **v1.0** (2026-02-20)
  - 初始版本
  - 支持 AMP 训练
  - Warmup + Cosine 学习率调度
  - 自动保存最佳模型
  - TensorBoard 集成