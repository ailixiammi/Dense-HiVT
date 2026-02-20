"""
Dense-HiVT 损失函数模块
实现基于 Winner-Takes-All (WTA) 机制的 Laplace NLL 损失和软目标交叉熵损失
包含严谨的坐标系转换和掩码处理，防止梯度污染
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def laplace_nll_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Laplace 负对数似然损失
    
    公式: NLL = |y - μ| / b + log(2b)
    
    Args:
        pred: 预测参数，Shape [..., 4]，最后一维为 [μ_x, μ_y, b_x, b_y]
        target: 真实坐标，Shape [..., 2]，最后一维为 [x, y]
    
    Returns:
        损失值，Shape [...] (在最后一维求和后的结果)
    """
    # 提取均值和尺度参数
    mu = pred[..., :2]  # [..., 2]
    b = pred[..., 2:]   # [..., 2]
    
    # 强制截断尺度参数，防止过小导致梯度爆炸
    b = torch.clamp(b, min=1e-3)
    
    # 除法和 log 都加入安全垫，防止数值不稳定
    nll = torch.abs(target - mu) / (b + 1e-5) + torch.log(2 * b + 1e-6)  # [..., 2]
    
    # 在最后一维求和（x 和 y 方向的损失相加）
    nll = nll.sum(dim=-1)  # [...]
    
    return nll


def soft_target_cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    软目标交叉熵损失
    
    公式: Loss = -Σ(target * log_softmax(pred))
    
    Args:
        pred: 预测 Logits（未归一化），Shape [B, N, 6]
        target: 软标签（已归一化），Shape [B, N, 6]
    
    Returns:
        损失值，Shape [B, N]
    """
    # 计算 log_softmax
    log_softmax_pred = F.log_softmax(pred, dim=-1)  # [B, N, 6]
    
    # 计算交叉熵: -(target * log_softmax(pred))
    loss = -(target * log_softmax_pred).sum(dim=-1)  # [B, N]
    
    return loss


class DenseHiVTLoss(nn.Module):
    """
    Dense-HiVT 顶层损失模块
    
    功能：
    1. 将世界坐标系下的未来真值转换到局部坐标系
    2. 基于 FDE 挑选 Winner 模态（Winner-Takes-All）
    3. 计算分类损失（软目标交叉熵）
    4. 计算回归损失（Laplace NLL）
    5. 严格处理各种掩码，防止梯度污染
    """
    
    def __init__(self):
        """
        初始化损失模块
        """
        super(DenseHiVTLoss, self).__init__()
    
    def forward(
        self,
        pi: torch.Tensor,                      # [B, N, 6]
        loc: torch.Tensor,                     # [B, N, 6, 30, 4]
        y: torch.Tensor,                       # [B, N, 30, 2]
        agent_current_pos: torch.Tensor,       # [B, N, 2]
        agent_current_heading: torch.Tensor,   # [B, N]
        reg_mask: torch.Tensor,                # [B, N, 30]
        valid_mask: torch.Tensor               # [B, N]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pi: 预测的模态 Logits（未归一化），Shape [B, N, 6]
            loc: 预测的轨迹参数，Shape [B, N, 6, 30, 4]，最后一维为 [μ_x, μ_y, b_x, b_y]
            y: 真实的未来轨迹（世界坐标系），Shape [B, N, 30, 2]
            agent_current_pos: 当前时刻位置（t=19），Shape [B, N, 2]
            agent_current_heading: 当前时刻朝向（t=19），Shape [B, N]
            reg_mask: 未来轨迹有效掩码，Shape [B, N, 30] (Bool, True=有效)
            valid_mask: 有效 Agent 掩码，Shape [B, N] (Bool, True=真实车辆)
        
        Returns:
            包含 reg_loss, cls_loss, total_loss 的字典
        """
        B, N, K, F_steps, _ = loc.shape  # B, N, 6, 30, 4
        
        # =====================================================================
        # 注意：y 已经在预处理时转换到局部坐标系，无需再次转换！
        # 局部坐标系定义：以 AV 在 t=19 的位置为原点，朝向为 x 轴正方向
        # =====================================================================
        
        # =====================================================================
        # 步骤 2: 挑选 Winner 模态（Winner-Takes-All）
        # =====================================================================
        # 2.1 提取预测位置（忽略尺度参数）
        loc_pos = loc[..., :2]  # [B, N, 6, 30, 2]
        
        # 2.2 计算 L2 距离
        # 扩展 y 维度以支持广播: [B, N, 30, 2] -> [B, N, 1, 30, 2]
        y_expanded = y.unsqueeze(2)  # [B, N, 1, 30, 2]
        
        # 手动展开 L2 范数计算并加入安全垫，防止求导时除零
        # 原始的 torch.norm 在 (0,0) 点求导会导致梯度爆炸
        diff = loc_pos - y_expanded  # [B, N, 6, 30, 2]
        l2_norm = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)  # [B, N, 6, 30]
        
        # 2.3 提取 FDE（最后一帧 t=29 的误差）
        fde = l2_norm[..., -1]  # [B, N, 6]
        
        # 2.4 找出 FDE 最小的模态索引
        best_mode = fde.argmin(dim=-1)  # [B, N]
        
        # =====================================================================
        # 步骤 3: 计算分类损失
        # =====================================================================        
        # 计算 Hard Label 交叉熵
        # F.cross_entropy 要求输入格式：
        #   - input: [B, C, *] 或 [B, C] where C = num_classes
        #   - target: [B, *] or [B] containing class indices
        # 当前 pi shape: [B, N, 6]，需要转置为 [B, 6, N]
        # 当前 best_mode shape: [B, N]，无需修改
        pi_transposed = pi.transpose(1, 2)  # [B, 6, N]
        
        # 计算标准交叉熵（内部自动执行 log_softmax，数值稳定性最优）
        cls_loss_unmasked = F.cross_entropy(
            pi_transposed,      # [B, 6, N]
            best_mode,          # [B, N]
            reduction='none'    # 返回 [B, N]，保留每个 Agent 的独立损失
        )  # [B, N]
        
        # =====================================================================
        # 步骤 4: 计算回归损失
        # =====================================================================
        # 4.1 提取 Winner 模态的预测结果
        # 使用 gather 提取最优模态的轨迹
        # best_mode: [B, N] -> [B, N, 1, 1, 1] 以匹配 loc 的维度
        best_mode_expanded = best_mode.view(B, N, 1, 1, 1).expand(B, N, 1, F_steps, 4)  # [B, N, 1, 30, 4]
        
        # 从 loc 中提取 Winner 轨迹
        y_hat_winner = torch.gather(loc, dim=2, index=best_mode_expanded).squeeze(2)  # [B, N, 30, 4]
        
        # 4.2 计算 Laplace NLL Loss（包含均值和方差预测）
        # 安全保护已在 laplace_nll_loss 函数内部实现
        reg_loss_per_frame = laplace_nll_loss(y_hat_winner, y)  # [B, N, 30]
        
        # 4.3 应用时间掩码并计算平均损失
        # 将 reg_mask 转换为 float 类型
        reg_mask_float = reg_mask.float()  # [B, N, 30]
        
        # 计算有效帧的损失总和
        reg_loss_sum = (reg_loss_per_frame * reg_mask_float).sum(dim=-1)  # [B, N]
        
        # 计算有效帧数，加入 epsilon 防止除零
        valid_frames = reg_mask_float.sum(dim=-1) + 1e-5  # [B, N]
        
        # 计算每个 Agent 的平均损失
        reg_loss_unmasked = reg_loss_sum / valid_frames  # [B, N]
        
        # =====================================================================
        # 步骤 5: 全局有效性过滤
        # =====================================================================
        # 5.1 将 valid_mask 转换为 float 类型
        valid_mask_float = valid_mask.float()  # [B, N]
        
        # 计算有效 Agent 的数量，加入 epsilon 防止除零
        num_valid_agents = valid_mask_float.sum() + 1e-5  # 标量
        
        # 5.3 过滤并计算最终的回归损失
        reg_loss = (reg_loss_unmasked * valid_mask_float).sum() / num_valid_agents
        
        # 5.4 过滤并计算最终的分类损失
        cls_loss = (cls_loss_unmasked * valid_mask_float).sum() / num_valid_agents
        
        # =====================================================================
        # 步骤 6: 计算总损失
        # =====================================================================
        total_loss = reg_loss + cls_loss
        
        # 返回损失字典
        return {
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'total_loss': total_loss
        }