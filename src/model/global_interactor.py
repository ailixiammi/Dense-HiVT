"""
Dense-HiVT 全局交互模块
打破局部 50m 空间限制，在全量有效 Agent 之间建立全连接的稠密注意力交互
零 PyG 依赖，全部基于稠密张量
"""

import torch
import torch.nn as nn
from typing import Tuple

from utils.geometry import get_local_geometry_and_mask
from model.embedding import SingleInputEmbedding
from model.local_encoder import CustomMaskedMHA


class GlobalRelativePositionEmbedding(nn.Module):
    """
    全局相对几何编码器
    
    将相对坐标和相对朝向拼接为 4 维几何向量，并映射为高维位置编码
    
    功能：
    1. 计算 Agent 之间的相对朝向（target_heading - center_heading）
    2. 将相对坐标 [dx, dy] 与相对朝向的三角函数 [cos(Δθ), sin(Δθ)] 拼接
    3. 将 4D 几何向量映射到 128 维空间
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Args:
            embed_dim: 嵌入维度，默认 128
        """
        super(GlobalRelativePositionEmbedding, self).__init__()
        
        # 4D 几何向量 -> 128D 位置编码
        self.geometry_emb = SingleInputEmbedding(in_channel=4, out_channel=embed_dim)
    
    def forward(
        self,
        local_rel_pos: torch.Tensor,   # [B, N, N, 2]
        agent_heading: torch.Tensor    # [B, N]
    ) -> torch.Tensor:
        """
        Args:
            local_rel_pos: 局部相对坐标，Shape [B, N, N, 2]
                          （已由 geometry 算子旋转到目标局部视角）
            agent_heading: 全局朝向角（弧度制），Shape [B, N]
        
        Returns:
            global_rel_pe: 全局相对位置编码，Shape [B, N, N, 128]
        """
        # =====================================================================
        # 步骤 1: 张量广播计算相对朝向
        # =====================================================================
        # rel_theta = target_heading - center_heading
        # 扩展维度以支持广播：[B, N] -> [B, 1, N] 和 [B, N, 1]
        center_heading = agent_heading.unsqueeze(2)  # [B, N, 1]
        target_heading = agent_heading.unsqueeze(1)  # [B, 1, N]
        
        # 计算相对朝向
        rel_theta = target_heading - center_heading  # [B, N, N]
        
        # =====================================================================
        # 步骤 2: 计算三角函数
        # =====================================================================
        # 计算 cos 和 sin，并扩展最后一个维度
        cos_theta = torch.cos(rel_theta).unsqueeze(-1)  # [B, N, N, 1]
        sin_theta = torch.sin(rel_theta).unsqueeze(-1)  # [B, N, N, 1]
        
        # =====================================================================
        # 步骤 3: 特征拼接
        # =====================================================================
        # 拼接相对坐标和相对朝向的三角函数
        # [dx, dy, cos(Δθ), sin(Δθ)]
        geometry_features = torch.cat([local_rel_pos, cos_theta, sin_theta], dim=-1)  # [B, N, N, 4]
        
        # =====================================================================
        # 步骤 4: 升维映射
        # =====================================================================
        # 将 4D 几何向量映射到 128D 空间
        global_rel_pe = self.geometry_emb(geometry_features)  # [B, N, N, 128]
        
        return global_rel_pe


class GlobalInteractorLayer(nn.Module):
    """
    单层全局交互网络
    
    使用 Pre-LN Transformer 架构：
    1. Multi-Head Self-Attention（带相对位置编码）
    2. Feed-Forward Network
    每个子层都使用 Pre-LN + 残差连接
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 特征维度，默认 128
            num_heads: 注意力头数，默认 8
            dropout: Dropout 概率，默认 0.1
        """
        super(GlobalInteractorLayer, self).__init__()
        
        # 多头注意力层 (Pre-LN)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CustomMaskedMHA(embed_dim, num_heads, dropout)
        
        # 前馈网络 (Pre-LN)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,                # [B, N, 128]
        global_rel_pe: torch.Tensor,    # [B, N, N, 128]
        attn_mask: torch.Tensor         # [B, N, N]
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征，Shape [B, N, 128]
            global_rel_pe: 全局相对位置编码，Shape [B, N, N, 128]
            attn_mask: 注意力掩码，Shape [B, N, N] (无效位置为 -inf)
        
        Returns:
            输出特征，Shape [B, N, 128]
        """
        # =====================================================================
        # 步骤 1: Multi-Head Self-Attention (Pre-LN + 残差)
        # =====================================================================
        # Pre-LN
        x_norm = self.norm1(x)  # [B, N, 128]
        
        # Self-Attention
        attn_out = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            rel_pe=global_rel_pe,
            attn_mask=attn_mask
        )  # [B, N, 128]
        
        # 残差连接
        x = x + attn_out  # [B, N, 128]
        
        # =====================================================================
        # 步骤 2: Feed-Forward Network (Pre-LN + 残差)
        # =====================================================================
        # Pre-LN
        x_norm = self.norm2(x)  # [B, N, 128]
        
        # FFN
        ffn_out = self.ffn(x_norm)  # [B, N, 128]
        
        # 残差连接
        x = x + ffn_out  # [B, N, 128]
        
        return x


class GlobalInteractor(nn.Module):
    """
    全局交互模块顶层
    
    打破局部 50m 空间限制，在全量有效 Agent 之间建立全连接交互
    
    核心策略：
    1. 使用超大半径（100000m）调用 geometry 算子，绕过距离截断
    2. 仅保留对 Padding 幽灵车辆的掩码（-inf），真实车辆之间全连接
    3. 多层 Transformer 提取全局长程依赖
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 特征维度，默认 128
            num_heads: 注意力头数，默认 8
            num_layers: Transformer 层数，默认 3
            dropout: Dropout 概率，默认 0.1
        """
        super(GlobalInteractor, self).__init__()
        
        # 全局相对位置编码器
        self.rel_pe_encoder = GlobalRelativePositionEmbedding(embed_dim=embed_dim)
        
        # 多层 Transformer
        self.layers = nn.ModuleList([
            GlobalInteractorLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        local_embed: torch.Tensor,      # [B, N, 128]
        agent_positions: torch.Tensor,  # [B, N, 2]
        agent_heading: torch.Tensor,    # [B, N]
        agent_mask: torch.Tensor        # [B, N]
    ) -> torch.Tensor:
        """
        Args:
            local_embed: 局部编码特征，Shape [B, N, 128]
            agent_positions: 智能体当前时刻位置（t=19），Shape [B, N, 2]
            agent_heading: 智能体当前时刻朝向（t=19），Shape [B, N]
            agent_mask: 智能体有效掩码，Shape [B, N] (Bool, True=真实车辆, False=Padding)
        
        Returns:
            global_embed: 全局交互后的特征，Shape [B, N, 128]
        """
        # =====================================================================
        # 步骤 1: 巧妙提取全图掩码与旋转坐标
        # =====================================================================
        # 【极其关键】使用超大半径（100000m）绕过距离截断
        # 这样 attn_mask 仅保留对 Padding 无效车辆的 -inf 死亡掩码
        # 真实车辆之间全部为 0（允许注意力流动）
        local_rel_pos, attn_mask = get_local_geometry_and_mask(
            center_pos=agent_positions,      # [B, N, 2]
            center_heading=agent_heading,    # [B, N]
            target_pos=agent_positions,      # [B, N, 2]
            target_mask=agent_mask,          # [B, N]
            radius=100000.0                  # 10 万米超大半径！
        )
        # local_rel_pos: [B, N, N, 2] - 旋转到局部视角的相对坐标
        # attn_mask: [B, N, N] - 仅 Padding 位置为 -inf，其余为 0
        
        # =====================================================================
        # 步骤 2: 生成全局相对位置编码
        # =====================================================================
        # 将相对坐标和相对朝向融合为 4D 几何向量，并映射到 128D
        global_rel_pe = self.rel_pe_encoder(local_rel_pos, agent_heading)  # [B, N, N, 128]
        
        # =====================================================================
        # 步骤 3: 循环多层 Transformer 进行全局交互
        # =====================================================================
        # 初始化为局部编码特征
        global_embed = local_embed  # [B, N, 128]
        
        # 逐层更新特征
        for layer in self.layers:
            global_embed = layer(
                x=global_embed,
                global_rel_pe=global_rel_pe,
                attn_mask=attn_mask
            )  # [B, N, 128]
        
        return global_embed
