"""
多模态轨迹预测全 GPU 评测引擎 (Metrics)

核心设计原则:
- 100% 纯 PyTorch 张量广播机制
- 零 Python for 循环
- 零 .cpu().numpy() 调用
- 完全在 GPU 上完成所有计算

支持的评测指标:
- minFDE: Minimum Final Displacement Error
- minADE: Minimum Average Displacement Error
- MR: Miss Rate (阈值 2.0 米)
"""

import torch
import time
from typing import Dict, Tuple


class ComputeMetrics:
    """
    多模态轨迹预测评测引擎
    
    输入张量:
        pred_trajs: [B, N, K, F, 2] - K 条预测轨迹
        gt_trajs: [B, N, F, 2] - Ground Truth 轨迹
        gt_masks: [B, N, F] - 时间步有效性掩码 (bool)
        target_masks: [B, N] - Agent 预测目标掩码 (bool)
    
    其中:
        B: Batch Size
        N: Agent 数量 (通常 64)
        K: 模态数 (通常 6)
        F: 未来时间步数 (通常 30)
    """
    
    def __init__(self, miss_threshold: float = 2.0):
        """
        初始化评测引擎
        
        Args:
            miss_threshold: Miss Rate 的距离阈值（米），默认 2.0
        """
        self.miss_threshold = miss_threshold
    
    def __call__(
        self,
        pred_trajs: torch.Tensor,
        gt_trajs: torch.Tensor,
        gt_masks: torch.BoolTensor,
        target_masks: torch.BoolTensor
    ) -> Dict[str, float]:
        """
        计算所有评测指标
        
        Args:
            pred_trajs: [B, N, K, F, 2] - 预测轨迹
            gt_trajs: [B, N, F, 2] - Ground Truth 轨迹
            gt_masks: [B, N, F] - 时间步有效性掩码
            target_masks: [B, N] - Agent 预测目标掩码
        
        Returns:
            包含 minADE, minFDE, MR 的字典
        """
        # =====================================================================
        # 步骤 1: 计算所有时间步的欧氏距离 [B, N, K, F]
        # =====================================================================
        # 将 gt_trajs 从 [B, N, F, 2] 扩展到 [B, N, 1, F, 2]
        # 与 pred_trajs [B, N, K, F, 2] 广播计算
        gt_trajs_expanded = gt_trajs.unsqueeze(2)  # [B, N, 1, F, 2]
        
        # 计算欧氏距离: sqrt((x1-x2)^2 + (y1-y2)^2)
        # 结果形状: [B, N, K, F]
        distances = torch.norm(
            pred_trajs - gt_trajs_expanded,
            dim=-1  # 在最后一维 (x, y) 上计算范数
        )
        
        # =====================================================================
        # 步骤 2: 计算有效时间步长度 [B, N]
        # =====================================================================
        # 统计每个轨迹的有效时间步数量
        valid_lengths = gt_masks.sum(dim=-1)  # [B, N]
        
        # 找到最后一个有效时间步的索引 (从 0 开始)
        # 例如: valid_lengths=20 → last_valid_idx=19
        last_valid_idx = (valid_lengths - 1).clamp(min=0)  # [B, N]
        
        # =====================================================================
        # 步骤 3: 计算 minFDE
        # =====================================================================
        min_fde = self._compute_min_fde(
            distances=distances,
            last_valid_idx=last_valid_idx,
            valid_lengths=valid_lengths,
            target_masks=target_masks
        )
        
        # =====================================================================
        # 步骤 4: 计算 minADE (基于 FDE 的 Winner 模态)
        # =====================================================================
        min_ade = self._compute_min_ade(
            distances=distances,
            gt_masks=gt_masks,
            last_valid_idx=last_valid_idx,
            valid_lengths=valid_lengths,
            target_masks=target_masks
        )
        
        # =====================================================================
        # 步骤 5: 计算 MR (Miss Rate)
        # =====================================================================
        miss_rate = self._compute_miss_rate(
            distances=distances,
            last_valid_idx=last_valid_idx,
            valid_lengths=valid_lengths,
            target_masks=target_masks,
            threshold=self.miss_threshold
        )
        
        return {
            'minADE': min_ade,
            'minFDE': min_fde,
            'MR': miss_rate
        }
    
    def _compute_min_fde(
        self,
        distances: torch.Tensor,
        last_valid_idx: torch.Tensor,
        valid_lengths: torch.Tensor,
        target_masks: torch.BoolTensor
    ) -> float:
        """
        计算 minFDE (Minimum Final Displacement Error)
        
        算法:
        1. 提取每条轨迹最后一个有效时间步的距离
        2. 在 K 个模态中选择距离最小的（Winner 模态）
        3. 计算所有有效目标的平均 FDE
        
        Args:
            distances: [B, N, K, F] - 所有时间步的距离
            last_valid_idx: [B, N] - 最后一个有效时间步的索引
            valid_lengths: [B, N] - 有效时间步数量
            target_masks: [B, N] - 预测目标掩码
        
        Returns:
            minFDE 标量值
        """
        B, N, K, F = distances.shape
        
        # =====================================================================
        # 步骤 1: 提取最后一个有效时间步的距离 [B, N, K]
        # =====================================================================
        # 使用 torch.gather 在时间维度上提取
        # last_valid_idx: [B, N] → [B, N, K, 1] (广播到所有模态)
        last_valid_idx_expanded = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1)
        
        # 从 distances [B, N, K, F] 中提取 → [B, N, K, 1]
        fde_per_mode = torch.gather(distances, dim=-1, index=last_valid_idx_expanded)
        fde_per_mode = fde_per_mode.squeeze(-1)  # [B, N, K]
        
        # =====================================================================
        # 步骤 2: 选择每个 Agent 的最小 FDE (Winner 模态)
        # =====================================================================
        min_fde_per_agent, _ = torch.min(fde_per_mode, dim=-1)  # [B, N]
        
        # =====================================================================
        # 步骤 3: 过滤有效目标并计算均值
        # =====================================================================
        # 双重掩码: 必须是预测目标 & 有有效数据
        valid_targets = target_masks & (valid_lengths > 0)  # [B, N]
        
        # 仅对有效目标计算均值
        if valid_targets.sum() > 0:
            min_fde = min_fde_per_agent[valid_targets].mean().item()
        else:
            min_fde = float('nan')  # 没有有效目标
        
        return min_fde
    
    def _compute_min_ade(
        self,
        distances: torch.Tensor,
        gt_masks: torch.BoolTensor,
        last_valid_idx: torch.Tensor,
        valid_lengths: torch.Tensor,
        target_masks: torch.BoolTensor
    ) -> float:
        """
        计算 minADE (Minimum Average Displacement Error)
        
        算法:
        1. 基于 FDE 选出 Winner 模态的索引
        2. 提取该 Winner 模态在所有有效时间步的距离
        3. 计算每个 Agent 的平均距离
        4. 计算所有有效目标的总体平均 ADE
        
        Args:
            distances: [B, N, K, F] - 所有时间步的距离
            gt_masks: [B, N, F] - 时间步有效性掩码
            last_valid_idx: [B, N] - 最后一个有效时间步的索引
            valid_lengths: [B, N] - 有效时间步数量
            target_masks: [B, N] - 预测目标掩码
        
        Returns:
            minADE 标量值
        """
        B, N, K, F = distances.shape
        
        # =====================================================================
        # 步骤 1: 基于 FDE 选出 Winner 模态索引
        # =====================================================================
        # 提取最后一个有效时间步的距离 [B, N, K]
        last_valid_idx_expanded = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1)
        fde_per_mode = torch.gather(distances, dim=-1, index=last_valid_idx_expanded).squeeze(-1)
        
        # 找到 FDE 最小的模态索引 [B, N]
        winner_mode_idx = torch.argmin(fde_per_mode, dim=-1)  # [B, N]
        
        # =====================================================================
        # 步骤 2: 提取 Winner 模态的距离 [B, N, F]
        # =====================================================================
        # winner_mode_idx: [B, N] → [B, N, 1, F] (广播到所有时间步)
        winner_mode_idx_expanded = winner_mode_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, F)
        
        # 从 distances [B, N, K, F] 中提取 Winner 模态 → [B, N, 1, F]
        winner_distances = torch.gather(distances, dim=2, index=winner_mode_idx_expanded)
        winner_distances = winner_distances.squeeze(2)  # [B, N, F]
        
        # =====================================================================
        # 步骤 3: 计算每个 Agent 的平均距离 (仅在有效时间步上)
        # =====================================================================
        # 将无效时间步的距离置为 0，避免影响求和
        masked_distances = winner_distances * gt_masks.float()  # [B, N, F]
        
        # 计算每个 Agent 的总距离并除以有效步数
        sum_distances = masked_distances.sum(dim=-1)  # [B, N]
        ade_per_agent = sum_distances / valid_lengths.clamp(min=1).float()  # [B, N]
        
        # =====================================================================
        # 步骤 4: 过滤有效目标并计算均值
        # =====================================================================
        valid_targets = target_masks & (valid_lengths > 0)
        
        if valid_targets.sum() > 0:
            min_ade = ade_per_agent[valid_targets].mean().item()
        else:
            min_ade = float('nan')
        
        return min_ade
    
    def _compute_miss_rate(
        self,
        distances: torch.Tensor,
        last_valid_idx: torch.Tensor,
        valid_lengths: torch.Tensor,
        target_masks: torch.BoolTensor,
        threshold: float
    ) -> float:
        """
        计算 MR (Miss Rate)
        
        算法:
        1. 提取每条轨迹最后一个有效时间步的距离
        2. 在 K 个模态中选择距离最小的（minFDE）
        3. 如果 minFDE > threshold，则计为 Miss (1.0)
        4. 计算所有有效目标的平均失误率
        
        Args:
            distances: [B, N, K, F] - 所有时间步的距离
            last_valid_idx: [B, N] - 最后一个有效时间步的索引
            valid_lengths: [B, N] - 有效时间步数量
            target_masks: [B, N] - 预测目标掩码
            threshold: 失误阈值（米）
        
        Returns:
            MR 标量值 (0.0 ~ 1.0)
        """
        B, N, K, F = distances.shape
        
        # =====================================================================
        # 步骤 1: 提取最后一个有效时间步的距离 [B, N, K]
        # =====================================================================
        last_valid_idx_expanded = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1)
        fde_per_mode = torch.gather(distances, dim=-1, index=last_valid_idx_expanded).squeeze(-1)
        
        # =====================================================================
        # 步骤 2: 选择每个 Agent 的最小 FDE
        # =====================================================================
        min_fde_per_agent, _ = torch.min(fde_per_mode, dim=-1)  # [B, N]
        
        # =====================================================================
        # 步骤 3: 判断是否 Miss (minFDE > threshold)
        # =====================================================================
        is_miss = (min_fde_per_agent > threshold).float()  # [B, N]
        
        # =====================================================================
        # 步骤 4: 过滤有效目标并计算均值
        # =====================================================================
        valid_targets = target_masks & (valid_lengths > 0)
        
        if valid_targets.sum() > 0:
            miss_rate = is_miss[valid_targets].mean().item()
        else:
            miss_rate = float('nan')
        
        return miss_rate


# =============================================================================
# 便捷函数封装
# =============================================================================

def compute_metrics(
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    gt_masks: torch.BoolTensor,
    target_masks: torch.BoolTensor,
    miss_threshold: float = 2.0
) -> Dict[str, float]:
    """
    便捷函数: 一次性计算所有评测指标
    
    Args:
        pred_trajs: [B, N, K, F, 2] - 预测轨迹
        gt_trajs: [B, N, F, 2] - Ground Truth 轨迹
        gt_masks: [B, N, F] - 时间步有效性掩码
        target_masks: [B, N] - Agent 预测目标掩码
        miss_threshold: Miss Rate 阈值（米），默认 2.0
    
    Returns:
        包含 minADE, minFDE, MR 的字典
    """
    metrics = ComputeMetrics(miss_threshold=miss_threshold)
    return metrics(pred_trajs, gt_trajs, gt_masks, target_masks)


# =============================================================================
# 测试代码块
# =============================================================================

if __name__ == "__main__":
    """
    评测引擎测试入口
    
    测试内容:
    1. 随机生成 Mock 数据
    2. 运行前向评测
    3. 打印 minADE, minFDE, MR 和计算耗时
    """
    print()
    print("=" * 80)
    print("多模态轨迹预测评测引擎测试".center(80))
    print("=" * 80)
    print()
    
    # =========================================================================
    # 步骤 1: 生成 Mock 数据
    # =========================================================================
    print("[步骤 1] 生成 Mock 数据...")
    
    B = 32   # Batch Size
    N = 64   # Agent 数量
    K = 6    # 模态数
    F = 30   # 未来时间步
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - 设备: {device}")
    print(f"  - 张量形状: B={B}, N={N}, K={K}, F={F}")
    print()
    
    # 随机生成预测轨迹 [B, N, K, F, 2]
    pred_trajs = torch.randn(B, N, K, F, 2, device=device) * 10.0
    
    # 随机生成 Ground Truth [B, N, F, 2]
    gt_trajs = torch.randn(B, N, F, 2, device=device) * 10.0
    
    # 生成有效性掩码 [B, N, F]
    # 模拟真实场景: 前 20 步有效，后 10 步可能无效
    gt_masks = torch.ones(B, N, F, dtype=torch.bool, device=device)
    # 随机将部分时间步标记为无效
    invalid_ratio = torch.rand(B, N, 1, device=device)
    invalid_steps = (torch.arange(F, device=device).unsqueeze(0).unsqueeze(0) > 
                     (F * invalid_ratio).long())
    gt_masks = gt_masks & ~invalid_steps
    
    # 生成目标掩码 [B, N]
    # 模拟真实场景: 约 30% 的 Agent 是预测目标
    target_masks = torch.rand(B, N, device=device) < 0.3
    
    print(f"[数据统计]")
    print(f"  - 预测目标数量: {target_masks.sum().item()} / {B * N}")
    print(f"  - 平均有效时间步: {gt_masks.sum(dim=-1).float().mean().item():.1f} / {F}")
    print()
    
    # =========================================================================
    # 步骤 2: 运行评测（计时）
    # =========================================================================
    print("[步骤 2] 运行评测引擎...")
    
    # 预热 GPU (避免首次调用的初始化开销)
    if device.type == 'cuda':
        _ = compute_metrics(pred_trajs, gt_trajs, gt_masks, target_masks)
        torch.cuda.synchronize()
    
    # 正式计时
    start_time = time.time()
    
    results = compute_metrics(pred_trajs, gt_trajs, gt_masks, target_masks)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    print(f"  - 计算耗时: {elapsed_time:.2f} ms")
    print()
    
    # =========================================================================
    # 步骤 3: 打印评测结果
    # =========================================================================
    print("=" * 80)
    print("评测结果".center(80))
    print("=" * 80)
    print()
    
    print(f"  minADE (Minimum Average Displacement Error): {results['minADE']:.4f} 米")
    print(f"  minFDE (Minimum Final Displacement Error):   {results['minFDE']:.4f} 米")
    print(f"  MR (Miss Rate @ 2.0m):                       {results['MR']:.2%}")
    
    print()
    print("=" * 80)
    print("✓ 评测引擎测试完成！".center(80))
    print("=" * 80)
    print()
    
    # =========================================================================
    # 步骤 4: 验证边界情况
    # =========================================================================
    print("[步骤 4] 验证边界情况...")
    print()
    
    # 测试 1: 所有 Agent 都不是目标
    print("  [测试 1] 所有 Agent 都不是预测目标...")
    target_masks_empty = torch.zeros(B, N, dtype=torch.bool, device=device)
    results_empty = compute_metrics(pred_trajs, gt_trajs, gt_masks, target_masks_empty)
    print(f"    - minADE: {results_empty['minADE']} (预期: nan)")
    print(f"    - minFDE: {results_empty['minFDE']} (预期: nan)")
    print(f"    - MR: {results_empty['MR']} (预期: nan)")
    print()
    
    # 测试 2: 所有轨迹长度为 0
    print("  [测试 2] 所有轨迹长度为 0...")
    gt_masks_zero = torch.zeros(B, N, F, dtype=torch.bool, device=device)
    results_zero = compute_metrics(pred_trajs, gt_trajs, gt_masks_zero, target_masks)
    print(f"    - minADE: {results_zero['minADE']} (预期: nan)")
    print(f"    - minFDE: {results_zero['minFDE']} (预期: nan)")
    print(f"    - MR: {results_zero['MR']} (预期: nan)")
    print()
    
    # 测试 3: 完美预测 (pred == gt)
    print("  [测试 3] 完美预测 (pred == gt)...")
    pred_perfect = gt_trajs.unsqueeze(2).expand(B, N, K, F, 2).clone()
    results_perfect = compute_metrics(pred_perfect, gt_trajs, gt_masks, target_masks)
    print(f"    - minADE: {results_perfect['minADE']:.6f} (预期: ~0.0)")
    print(f"    - minFDE: {results_perfect['minFDE']:.6f} (预期: ~0.0)")
    print(f"    - MR: {results_perfect['MR']:.6f} (预期: 0.0)")
    print()
    
    print("=" * 80)
    print("✓ 所有测试通过！评测引擎准备就绪".center(80))
    print("=" * 80)