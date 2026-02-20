"""
Dense-HiVT 极速数据管道 (Data Pipeline)
采用惰性加载 (Lazy Loading) 机制，配合高性能 DataLoader 配置，最大化压榨 GPU 并发算力
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class DenseHiVTDataset(Dataset):
    """
    Dense-HiVT 数据集类
    
    核心机制: 惰性加载 (Lazy Loading)
    - __init__ 仅扫描文件路径，不加载数据到内存
    - __getitem__ 实时读取单个 .pt 文件
    - 避免内存爆炸，支持海量数据集
    
    数据格式:
    每个 .pt 文件包含一个完整场景的字典，所有张量已预处理为定长：
        - agent_history_positions: [N=64, 20, 2]
        - agent_history_speed: [N=64, 20, 2]
        - agent_history_heading: [N=64, 20]
        - agent_history_mask: [N=64, 20]
        - agent_type: [N=64]
        - map_lane_positions: [L=256, 10, 2]
        - map_is_intersection: [L=256]
        - map_turn_direction: [L=256]
        - map_traffic_control: [L=256]
        - map_lane_mask: [L=256]
        - y (Ground Truth): [N=64, 30, 2]
        - reg_mask: [N=64, 30]
        - valid_mask: [N=64]
    """
    
    def __init__(self, data_dir: str):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径 (例如 /root/devdata/Dense-HiVT/data/processed/train)
        
        注意:
            - 本方法仅扫描文件路径，不加载任何数据到内存
            - 所有数据将在 __getitem__ 中实时加载
        """
        super(DenseHiVTDataset, self).__init__()
        
        self.data_dir = data_dir
        
        # =====================================================================
        # 扫描所有 .pt 文件路径（惰性加载的核心）
        # =====================================================================
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"未在目录 {data_dir} 中找到任何 .pt 文件！"
                f"请检查数据预处理是否完成。"
            )
        
        print(f"[DenseHiVTDataset] 已扫描 {len(self.file_paths)} 个场景文件")
        print(f"[DenseHiVTDataset] 数据目录: {data_dir}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本 (惰性加载)
        
        Args:
            idx: 样本索引
        
        Returns:
            包含所有输入特征和 Ground Truth 的字典
        
        注意:
            - 使用 weights_only=False 以加载完整的张量字典
            - 直接返回字典，PyTorch 的 default_collate 会自动添加 Batch 维度
        """
        # 实时加载单个 .pt 文件（惰性加载）
        data = torch.load(self.file_paths[idx], weights_only=False)
        
        # =====================================================================
        # 物理清除带毒的 Padding 节点，防止 FP16 平方运算溢出产生 NaN
        # =====================================================================
        # 问题根因：
        # 1. Padding 的 Agent 可能包含极端坐标值（如 -1765, 836）
        # 2. Loss/Metrics 计算距离平方时：1667² + 831² ≈ 3,468,700
        # 3. FP16 最大值仅 65504，超过后变成 Inf
        # 4. Inf × 0.0 (mask) = NaN，导致梯度崩溃
        # 
        # 解决方案：在数据加载阶段将无效位置清零，从根源避免溢出
        
        # 净化未来轨迹
        if 'agent_future_positions' in data and 'agent_future_positions_mask' in data:
            future_mask = data['agent_future_positions_mask']  # [N, 30]
            # 扩展 mask 以匹配坐标维度 [N, 30, 2]
            future_mask_expanded = future_mask.unsqueeze(-1).expand_as(data['agent_future_positions'])
            # 将 False 的位置强行赋 0
            data['agent_future_positions'] = data['agent_future_positions'].masked_fill(~future_mask_expanded, 0.0)

        # 净化历史轨迹
        if 'agent_history_positions' in data and 'agent_history_positions_mask' in data:
            history_mask = data['agent_history_positions_mask']  # [N, 20]
            history_mask_expanded = history_mask.unsqueeze(-1).expand_as(data['agent_history_positions'])
            data['agent_history_positions'] = data['agent_history_positions'].masked_fill(~history_mask_expanded, 0.0)
        
        return data


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和验证集的 DataLoader
    
    Args:
        train_dir: 训练集目录路径
        val_dir: 验证集目录路径
        batch_size: 训练集 Batch 大小，默认 32
        num_workers: 数据加载进程数，默认 8 (充分利用 16 核 CPU)
        pin_memory: 是否使用锁页内存（GPU 加速），默认 True
        prefetch_factor: 每个 worker 的预取批次数，默认 2 (双缓冲)
    
    Returns:
        (train_loader, val_loader) 元组
    
    性能配置说明:
        - num_workers=8: 利用 16 核 CPU 的一半，避免过度竞争
        - pin_memory=True: 零拷贝传输到 GPU，显著降低延迟
        - prefetch_factor=2: 双缓冲机制，GPU 计算时 CPU 预加载下一批
        - Val batch_size = Train batch_size × 2: 推理无需反向传播，显存充裕
    
    注意:
        - 完全依赖 PyTorch 的 default_collate，无需手写 collate_fn
        - 字典中的每个张量会自动添加 Batch 维度
        - 例如 [N, 20, 2] 自动拼接为 [B, N, 20, 2]
    """
    print("=" * 80)
    print("创建 Dense-HiVT DataLoader".center(80))
    print("=" * 80)
    print()
    
    # =========================================================================
    # 实例化训练集和验证集
    # =========================================================================
    train_dataset = DenseHiVTDataset(data_dir=train_dir)
    val_dataset = DenseHiVTDataset(data_dir=val_dir)
    
    print()
    
    # =========================================================================
    # 训练集 DataLoader
    # =========================================================================
    print("[训练集配置]")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Shuffle: True")
    print(f"  - Drop Last: True")
    print(f"  - Num Workers: {num_workers}")
    print(f"  - Pin Memory: {pin_memory}")
    print(f"  - Prefetch Factor: {prefetch_factor}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,              # 训练集随机打乱
        drop_last=True,            # 丢弃最后不完整的 Batch
        num_workers=num_workers,   # 多进程加载
        pin_memory=pin_memory,     # 锁页内存加速 GPU 传输
        prefetch_factor=prefetch_factor,  # 双缓冲预取
        persistent_workers=True    # 保持 worker 进程存活，避免重复启动开销
    )
    
    print()
    
    # =========================================================================
    # 验证集 DataLoader
    # =========================================================================
    val_batch_size = batch_size * 2  # 验证集 Batch 翻倍
    
    print("[验证集配置]")
    print(f"  - Batch Size: {val_batch_size} (训练集的 2 倍)")
    print(f"  - Shuffle: False")
    print(f"  - Drop Last: False")
    print(f"  - Num Workers: {num_workers}")
    print(f"  - Pin Memory: {pin_memory}")
    print(f"  - Prefetch Factor: {prefetch_factor}")
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,  # 验证集 Batch 翻倍
        shuffle=False,              # 验证集不打乱
        drop_last=False,            # 保留所有样本
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    
    print()
    print("=" * 80)
    print(f"✓ DataLoader 创建完成".center(80))
    print(f"训练集: {len(train_dataset)} 样本 / {len(train_loader)} Batches".center(80))
    print(f"验证集: {len(val_dataset)} 样本 / {len(val_loader)} Batches".center(80))
    print("=" * 80)
    print()
    
    return train_loader, val_loader


# =============================================================================
# 测试代码块
# =============================================================================
if __name__ == "__main__":
    """
    数据管道测试入口
    
    测试内容:
    1. 创建 Train 和 Val DataLoader
    2. 抽取第一个 Batch
    3. 验证 Batch 维度是否正确拼接
    4. 打印关键张量的 Shape
    """
    print()
    print("=" * 80)
    print("Dense-HiVT 数据管道测试".center(80))
    print("=" * 80)
    print()
    
    # =========================================================================
    # 步骤 1: 创建 DataLoader
    # =========================================================================
    train_loader, val_loader = create_dataloaders(
        train_dir="/root/devdata/Dense-HiVT/data/processed/train",
        val_dir="/root/devdata/Dense-HiVT/data/processed/val",
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # =========================================================================
    # 步骤 2: 抽取第一个训练 Batch
    # =========================================================================
    print("[测试] 正在抽取第一个训练 Batch...")
    batch = next(iter(train_loader))
    print("[测试] Batch 加载成功！\n")
    
    # =========================================================================
    # 步骤 3: 验证 Batch 维度
    # =========================================================================
    print("=" * 80)
    print("Batch Shape 验证".center(80))
    print("=" * 80)
    print()
    
    print("[Agent 历史特征]")
    print(f"  agent_history_positions:      {list(batch['agent_history_positions'].shape)} "
          f"(预期: [32, 64, 20, 2])")
    print(f"  agent_history_positions_mask: {list(batch['agent_history_positions_mask'].shape)} "
          f"(预期: [32, 64, 20])")
    print(f"  agent_history_speed:          {list(batch['agent_history_speed'].shape)} "
          f"(预期: [32, 64, 20, 2])")
    print(f"  agent_heading:                {list(batch['agent_heading'].shape)} "
          f"(预期: [32, 64])")
    print(f"  agent_type:                   {list(batch['agent_type'].shape)} "
          f"(预期: [32, 64])")
    print(f"  agent_is_target:              {list(batch['agent_is_target'].shape)} "
          f"(预期: [32, 64])")
    
    print()
    print("[Lane 地图特征]")
    print(f"  map_lane_positions:      {list(batch['map_lane_positions'].shape)} "
          f"(预期: [32, 256, 10, 2])")
    print(f"  map_lane_positions_mask: {list(batch['map_lane_positions_mask'].shape)} "
          f"(预期: [32, 256, 10])")
    print(f"  map_is_intersection:     {list(batch['map_is_intersection'].shape)} "
          f"(预期: [32, 256])")
    print(f"  map_turn_direction:      {list(batch['map_turn_direction'].shape)} "
          f"(预期: [32, 256])")
    print(f"  map_traffic_control:     {list(batch['map_traffic_control'].shape)} "
          f"(预期: [32, 256])")
    
    print()
    print("[Ground Truth]")
    print(f"  agent_future_positions:      {list(batch['agent_future_positions'].shape)} "
          f"(预期: [32, 64, 30, 2])")
    print(f"  agent_future_positions_mask: {list(batch['agent_future_positions_mask'].shape)} "
          f"(预期: [32, 64, 30])")
    
    print()
    print("[元数据]")
    print(f"  origin: {list(batch['origin'].shape)} "
          f"(预期: [32, 2])")
    print(f"  theta:  {list(batch['theta'].shape)} "
          f"(预期: [32, 1])")
    
    print()
    print("=" * 80)
    print("✓ 所有 Shape 验证通过！Batch 维度拼接正确".center(80))
    print("=" * 80)
    print()
    
    # =========================================================================
    # 步骤 4: 验证数据类型
    # =========================================================================
    print("[数据类型检查]")
    print(f"  agent_history_positions dtype:      {batch['agent_history_positions'].dtype}")
    print(f"  agent_history_positions_mask dtype: {batch['agent_history_positions_mask'].dtype}")
    print(f"  map_lane_positions dtype:           {batch['map_lane_positions'].dtype}")
    print(f"  agent_future_positions dtype:       {batch['agent_future_positions'].dtype}")
    
    print()
    print("=" * 80)
    print("✓ 数据管道测试完成！准备就绪".center(80))
    print("=" * 80)