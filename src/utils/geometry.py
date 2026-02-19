"""
Dense-HiVT 几何变换工具模块
实现全流程稠密张量的几何变换，移除 PyG 稀疏图依赖
"""

import torch
from typing import Tuple


def get_local_geometry_and_mask(
    center_pos: torch.Tensor,      # [B, N, 2] 中心 Agent 的全局坐标
    center_heading: torch.Tensor,  # [B, N] 中心 Agent 的朝向角（弧度制）
    target_pos: torch.Tensor,      # [B, M, 2] 目标节点的全局坐标
    target_mask: torch.Tensor,     # [B, M] Bool 张量，True=有效节点，False=Padding 幽灵节点
    radius: float = 50.0           # 注意力截断半径（米）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    核心几何变换算子：通过张量广播机制完成全局坐标到局部坐标的旋转变换，并生成注意力掩码
    
    功能：
    1. 将目标节点的全局坐标转换为相对于中心 Agent 的局部坐标系
    2. 生成基于距离和有效性的双重硬截断注意力掩码
    
    Args:
        center_pos: 中心 Agent 的全局坐标，Shape [B, N, 2]
        center_heading: 中心 Agent 的朝向角（弧度制），Shape [B, N]
        target_pos: 目标节点的全局坐标，Shape [B, M, 2]
                   - 如果是 AA 交互（Agent-Agent），M=N=64
                   - 如果是 AL 交互（Agent-Lane），M=L=256
        target_mask: 目标节点的有效性掩码，Shape [B, M]
                    True 表示有效节点，False 表示因上限而 Padding 的幽灵节点
        radius: 注意力截断半径（米），默认 50.0
    
    Returns:
        local_rel_pos: 局部相对坐标，Shape [B, N, M, 2]
        attn_mask: 注意力掩码，Shape [B, N, M]
                  有效位置为 0.0，无效位置为 -inf（会被 Softmax 归零）
    
    数学原理：
        对于中心 Agent i 和目标节点 j：
        1. 全局相对坐标：delta_global = target_pos[j] - center_pos[i]
        2. 旋转矩阵：R^T = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
        3. 局部相对坐标：delta_local = R^T @ delta_global
    """
    
    # =====================================================================
    # 步骤 1: 张量广播求全局相对坐标
    # =====================================================================
    # 扩展维度以支持广播机制
    center_pos_expanded = center_pos.unsqueeze(2)  # [B, N, 1, 2]
    target_pos_expanded = target_pos.unsqueeze(1)  # [B, 1, M, 2]
    
    # 通过广播机制计算全局相对坐标
    global_rel_pos = target_pos_expanded - center_pos_expanded  # [B, N, M, 2]
    
    
    # =====================================================================
    # 步骤 2: 计算欧式距离与生成 50m 掩码
    # =====================================================================
    # 计算欧式距离（L2 范数）
    dist = torch.norm(global_rel_pos, p=2, dim=-1)  # [B, N, M]
    
    # 初始化注意力掩码（全 0）
    attn_mask = torch.zeros_like(dist)  # [B, N, M]
    
    # 双重硬截断：
    # 条件 1: 距离超过 radius 的位置
    distance_invalid = dist > radius  # [B, N, M]
    
    # 条件 2: target_mask 为 False 的位置（Padding 幽灵节点）
    # 需要广播到 [B, N, M] 的形状
    target_mask_expanded = target_mask.unsqueeze(1)  # [B, 1, M]
    padding_invalid = ~target_mask_expanded  # [B, 1, M]，会自动广播到 [B, N, M]
    
    # 合并两个无效条件
    invalid_mask = distance_invalid | padding_invalid  # [B, N, M]
    
    # 将无效位置填充为 -inf（Softmax 后会变为 0）
    attn_mask = attn_mask.masked_fill(invalid_mask, float('-inf'))  # [B, N, M]
    
    
    # =====================================================================
    # 步骤 3: 动态构建旋转矩阵
    # =====================================================================
    # 计算旋转矩阵的元素
    cos_theta = torch.cos(center_heading)  # [B, N]
    sin_theta = torch.sin(center_heading)  # [B, N]
    
    # 构建旋转矩阵 R^T = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
    # 注意：这是转置后的旋转矩阵，用于将全局坐标系转换为局部坐标系
    rot_matrix = torch.stack([
        torch.stack([cos_theta, sin_theta], dim=-1),      # 第一行: [cos(θ), sin(θ)]
        torch.stack([-sin_theta, cos_theta], dim=-1)      # 第二行: [-sin(θ), cos(θ)]
    ], dim=-2)  # [B, N, 2, 2]
    
    
    # =====================================================================
    # 步骤 4: 批量矩阵乘法（Batched Matmul）
    # =====================================================================
    # 扩展 rot_matrix 以支持批量乘法
    rot_matrix_expanded = rot_matrix.unsqueeze(2)  # [B, N, 1, 2, 2]
    
    # 将 global_rel_pos 转换为列向量形式
    global_rel_pos_col = global_rel_pos.unsqueeze(-1)  # [B, N, M, 2, 1]
    
    # 批量矩阵乘法：R^T @ delta_global
    local_rel_pos_col = torch.matmul(rot_matrix_expanded, global_rel_pos_col)  # [B, N, M, 2, 1]
    
    # 去除最后一个维度，得到最终的局部相对坐标
    local_rel_pos = local_rel_pos_col.squeeze(-1)  # [B, N, M, 2]
    
    
    return local_rel_pos, attn_mask
