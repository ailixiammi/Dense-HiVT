"""
Dense-HiVT 多模态未来解码器模块
接收全局特征，注入智能体类别先验，输出 K=6 条未来预测轨迹及模态概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultimodalDecoder(nn.Module):
    """
    多模态解码器
    
    功能：
    1. 将智能体类别信息（One-Hot）与全局特征融合
    2. 输出 K=6 个模态的分类评分（未归一化 Logits）
    3. 输出 K=6 个模态的轨迹预测，每条轨迹包含 F=30 帧的 Laplace 分布参数
       - 每帧预测 4 个值：[μ_x, μ_y, b_x, b_y]
       - μ_x, μ_y: 位置均值
       - b_x, b_y: Laplace 分布尺度参数（必须 > 0）
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_modes: int = 6,
        future_steps: int = 30,
        num_classes: int = 3
    ):
        """
        Args:
            embed_dim: 特征维度，默认 128
            num_modes: 模态数 K，默认 6
            future_steps: 未来预测帧数 F，默认 30
            num_classes: 智能体类别数，默认 3（0=AV, 1=Agent, 2=Others）
        """
        super(MultimodalDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.num_classes = num_classes
        
        # =====================================================================
        # 1. 特征聚合层：将 (embed_dim + num_classes) 维拼接特征降维回 embed_dim
        # =====================================================================
        self.agg_embed = nn.Sequential(
            nn.Linear(embed_dim + num_classes, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # =====================================================================
        # 2. 分类评分头：输出 K 个模态的未归一化 Logits
        # =====================================================================
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_modes)
        )
        
        # =====================================================================
        # 3. 轨迹回归头：输出 K * F * 4 维向量
        #    每个模态预测 F 帧，每帧 4 个值 [μ_x, μ_y, b_x, b_y]
        # =====================================================================
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_modes * future_steps * 4)
        )
    
    def forward(
        self,
        global_embed: torch.Tensor,   # [B, N, 128]
        agent_type: torch.Tensor      # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            global_embed: 全局交互特征，Shape [B, N, 128]
            agent_type: 智能体类别，Shape [B, N] (Long 类型)
        
        Returns:
            pi: 模态分类 Logits（未归一化），Shape [B, N, 6]
            loc: 轨迹预测参数，Shape [B, N, 6, 30, 4]
                 最后一维：[μ_x, μ_y, b_x, b_y]
        """
        B, N, _ = global_embed.shape
        
        # =====================================================================
        # 步骤 1: 类别注入（Late Fusion）
        # =====================================================================
        # 将 agent_type 钳制到有效范围 [0, 2]，处理 padding 位置的 -1 值
        agent_type_safe = torch.clamp(agent_type, min=0, max=self.num_classes - 1)
        
        # 将 agent_type 转换为 One-Hot 编码
        # 注意：需要确保 agent_type 是 Long 类型
        agent_type_onehot = F.one_hot(
            agent_type_safe.long(), 
            num_classes=self.num_classes
        ).float()  # [B, N, 3]
        
        # =====================================================================
        # 步骤 2: 特征拼接与聚合
        # =====================================================================
        # 拼接全局特征和类别 One-Hot
        x = torch.cat([global_embed, agent_type_onehot], dim=-1)  # [B, N, 131]
        
        # 通过聚合层降维回 128
        x = self.agg_embed(x)  # [B, N, 128]
        
        # =====================================================================
        # 步骤 3: 分类推理（模态概率）
        # =====================================================================
        # 输出未归一化的 Logits（在损失函数中会使用 Softmax）
        pi = self.cls_head(x)  # [B, N, 6]
        
        # =====================================================================
        # 步骤 4: 回归推理（轨迹参数）
        # =====================================================================
        # 输出原始回归值
        loc = self.reg_head(x)  # [B, N, 720] (6 * 30 * 4 = 720)
        
        # 重塑为 [B, N, K, F, 4]
        loc = loc.view(B, N, self.num_modes, self.future_steps, 4)  # [B, N, 6, 30, 4]
        
        # =====================================================================
        # 步骤 5: 【防 NaN 装甲】确保 Laplace 尺度参数 b > 0
        # =====================================================================
        # 对最后一维的后两个元素（b_x, b_y）应用 Softplus + 小偏置
        # loc[..., :2] 保持不变（μ_x, μ_y 可以是任意实数）
        # loc[..., 2:] 是尺度参数，必须为正
        
        # 拆分位置均值 (mu) 和尺度参数 (b)
        loc_mu = loc[..., :2]  # [B, N, 6, 30, 2], 允许为负数
        loc_b = loc[..., 2:]   # [B, N, 6, 30, 2], 需要激活为正数
        
        # 对尺度参数应用 Softplus + 小偏置
        loc_b = F.softplus(loc_b) + 1e-4
        
        # 重新拼接为一个新的张量，保护反向传播计算图
        loc = torch.cat([loc_mu, loc_b], dim=-1)  # [B, N, 6, 30, 4]
        
        return pi, loc