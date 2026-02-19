"""
Dense-HiVT 局部编码器模块
实现无图依赖的纯稠密张量交互：逐帧 AA交互 -> 序列压缩 -> AL交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from .embedding import AgentNodeEmbedding, LaneNodeEmbedding, RelativePositionEmbedding
from ..utils.geometry import get_local_geometry_and_mask


class CustomMaskedMHA(nn.Module):
    """
    带相对位置编码的自研多头注意力模块
    
    使用 einsum 实现高效的多头注意力计算，避免生成 5D 张量
    支持相对位置编码的加法式融合
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
        super(CustomMaskedMHA, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 相对位置编码投影层
        self.pe_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,      # [B, N, 128]
        key: torch.Tensor,        # [B, M, 128]
        value: torch.Tensor,      # [B, M, 128]
        rel_pe: torch.Tensor,     # [B, N, M, 128]
        attn_mask: torch.Tensor   # [B, N, M]
    ) -> torch.Tensor:
        """
        Args:
            query: Query 特征，Shape [B, N, 128]
            key: Key 特征，Shape [B, M, 128]
            value: Value 特征，Shape [B, M, 128]
            rel_pe: 相对位置编码，Shape [B, N, M, 128]
            attn_mask: 注意力掩码，Shape [B, N, M] (无效位置为 -inf)
        
        Returns:
            输出特征，Shape [B, N, 128]
        """
        B, N, _ = query.shape
        M = key.shape[1]
        H = self.num_heads
        D = self.head_dim
        
        # =====================================================================
        # 步骤 1: 投影并拆分多头
        # =====================================================================
        # 投影 Q, K, V
        q = self.q_proj(query)  # [B, N, 128]
        k = self.k_proj(key)    # [B, M, 128]
        v = self.v_proj(value)  # [B, M, 128]
        
        # 投影相对位置编码
        pe = self.pe_proj(rel_pe)  # [B, N, M, 128]
        
        # 拆分多头：[B, N, 128] -> [B, N, H, D]
        q = q.reshape(B, N, H, D)  # [B, N, 8, 16]
        k = k.reshape(B, M, H, D)  # [B, M, 8, 16]
        v = v.reshape(B, M, H, D)  # [B, M, 8, 16]
        pe = pe.reshape(B, N, M, H, D)  # [B, N, M, 8, 16]
        
        # =====================================================================
        # 步骤 2: 计算 Attention Score (使用 einsum 避免 5D 张量)
        # =====================================================================
        # Score = Q @ K^T + Q @ PE^T
        score_qk = torch.einsum('bnhd,bmhd->bhnm', q, k)      # [B, 8, N, M]
        score_qpe = torch.einsum('bnhd,bnmhd->bhnm', q, pe)   # [B, 8, N, M]
        score = score_qk + score_qpe  # [B, 8, N, M]
        
        # =====================================================================
        # 步骤 3: 缩放
        # =====================================================================
        score = score / self.scale  # [B, 8, N, M]
        
        # =====================================================================
        # 步骤 4: 应用注意力掩码
        # =====================================================================
        # 扩展掩码维度以匹配多头：[B, N, M] -> [B, 1, N, M]
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N, M]
        score = score + attn_mask  # [B, 8, N, M]
        
        # =====================================================================
        # 步骤 5: Softmax 激活
        # =====================================================================
        attn_weight = F.softmax(score, dim=-1)  # [B, 8, N, M]
        
        # =====================================================================
        # 步骤 6: 防 NaN 机制
        # =====================================================================
        # 当某一行全是 -inf 时，softmax 会产生 NaN，需要替换为 0
        attn_weight = torch.nan_to_num(attn_weight, nan=0.0)  # [B, 8, N, M]
        
        # =====================================================================
        # 步骤 7: Dropout
        # =====================================================================
        attn_weight = self.dropout(attn_weight)  # [B, 8, N, M]
        
        # =====================================================================
        # 步骤 8: 计算输出 (使用 einsum)
        # =====================================================================
        # Output = Attention @ V + Attention @ PE
        out_v = torch.einsum('bhnm,bmhd->bnhd', attn_weight, v)     # [B, N, 8, 16]
        out_pe = torch.einsum('bhnm,bnmhd->bnhd', attn_weight, pe)  # [B, N, 8, 16]
        out = out_v + out_pe  # [B, N, 8, 16]
        
        # =====================================================================
        # 步骤 9: 拼接多头并投影
        # =====================================================================
        # 拼接多头：[B, N, 8, 16] -> [B, N, 128]
        out = out.contiguous().reshape(B, N, self.embed_dim)  # [B, N, 128]
        
        # 输出投影
        out = self.out_proj(out)  # [B, N, 128]
        
        return out


class DenseAAEncoder(nn.Module):
    """
    时间维度折叠的 Agent-Agent 交互模块
    
    处理流程：
    1. 折叠时间维度到 Batch 维度
    2. 在超大 Batch 上进行 AA 交互
    3. 恢复时间维度
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
        super(DenseAAEncoder, self).__init__()
        
        # 相对位置编码器
        self.rel_pe = RelativePositionEmbedding(embed_dim=embed_dim)
        
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
        agent_history_positions: torch.Tensor,  # [B, N, 20, 2]
        agent_history_heading: torch.Tensor,    # [B, N, 20]
        agent_history_mask: torch.Tensor,       # [B, N, 20]
        agent_features: torch.Tensor            # [B, N, 20, 128]
    ) -> torch.Tensor:
        """
        Args:
            agent_history_positions: 智能体历史位置，Shape [B, N, 20, 2]
            agent_history_heading: 智能体历史朝向，Shape [B, N, 20]
            agent_history_mask: 智能体历史有效掩码，Shape [B, N, 20] (Bool)
            agent_features: 智能体特征，Shape [B, N, 20, 128]
        
        Returns:
            交互后的特征，Shape [B, N, 20, 128]
        """
        B, N, T, _ = agent_history_positions.shape
        
        # =====================================================================
        # 步骤 1: 折叠时间维度到 Batch 维度
        # =====================================================================
        # 将 [B, N, 20, ...] 变为 [B*20, N, ...]
        pos_flat = agent_history_positions.transpose(1, 2).reshape(B * T, N, 2)  # [B*20, N, 2]
        heading_flat = agent_history_heading.transpose(1, 2).reshape(B * T, N)    # [B*20, N]
        mask_flat = agent_history_mask.transpose(1, 2).reshape(B * T, N)          # [B*20, N]
        features_flat = agent_features.transpose(1, 2).reshape(B * T, N, 128)     # [B*20, N, 128]
        
        # =====================================================================
        # 步骤 2: 计算局部几何和掩码
        # =====================================================================
        # AA 交互：center 和 target 都是 Agent 自己
        local_rel_pos, attn_mask = get_local_geometry_and_mask(
            center_pos=pos_flat,      # [B*20, N, 2]
            center_heading=heading_flat,  # [B*20, N]
            target_pos=pos_flat,      # [B*20, N, 2]
            target_mask=mask_flat,    # [B*20, N]
            radius=50.0
        )
        # local_rel_pos: [B*20, N, N, 2]
        # attn_mask: [B*20, N, N]
        
        # =====================================================================
        # 步骤 3: 生成相对位置编码
        # =====================================================================
        rel_pe = self.rel_pe(local_rel_pos)  # [B*20, N, N, 128]
        
        # =====================================================================
        # 步骤 4: Multi-Head Attention (Pre-LN + 残差)
        # =====================================================================
        # Pre-LN
        features_norm = self.norm1(features_flat)  # [B*20, N, 128]
        
        # Self-Attention
        attn_out = self.attn(
            query=features_norm,
            key=features_norm,
            value=features_norm,
            rel_pe=rel_pe,
            attn_mask=attn_mask
        )  # [B*20, N, 128]
        
        # 残差连接
        features_flat = features_flat + attn_out  # [B*20, N, 128]
        
        # =====================================================================
        # 步骤 5: Feed-Forward Network (Pre-LN + 残差)
        # =====================================================================
        # Pre-LN
        features_norm = self.norm2(features_flat)  # [B*20, N, 128]
        
        # FFN
        ffn_out = self.ffn(features_norm)  # [B*20, N, 128]
        
        # 残差连接
        features_flat = features_flat + ffn_out  # [B*20, N, 128]
        
        # =====================================================================
        # 步骤 6: 恢复时间维度
        # =====================================================================
        # 将 [B*20, N, 128] 变回 [B, N, 20, 128]
        features_out = features_flat.reshape(B, T, N, 128).transpose(1, 2)  # [B, N, 20, 128]
        
        return features_out


class DenseTemporalEncoder(nn.Module):
    """
    时序压缩模块
    
    使用标准 Transformer Encoder 对时间序列进行编码，并提取最后时刻的状态
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
        super(DenseTemporalEncoder, self).__init__()
        
        # 标准 Transformer Encoder Layer
        self.temporal_encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(
        self,
        agent_features: torch.Tensor,      # [B, N, 20, 128]
        agent_history_mask: torch.Tensor   # [B, N, 20]
    ) -> torch.Tensor:
        """
        Args:
            agent_features: 智能体特征，Shape [B, N, 20, 128]
            agent_history_mask: 智能体历史有效掩码，Shape [B, N, 20] (Bool, True=有效)
        
        Returns:
            压缩后的特征（最后时刻状态），Shape [B, N, 128]
        """
        B, N, T, D = agent_features.shape
        
        # =====================================================================
        # 步骤 1: 重塑为 [B*N, 20, 128]
        # =====================================================================
        features_flat = agent_features.reshape(B * N, T, D)  # [B*N, 20, 128]
        mask_flat = agent_history_mask.reshape(B * N, T)     # [B*N, 20]
        
        # =====================================================================
        # 步骤 2: 准备 Padding Mask
        # =====================================================================
        # PyTorch TransformerEncoder 要求无效位置为 True
        # 我们的 mask 是 True=有效，所以需要取反
        padding_mask = ~mask_flat  # [B*N, 20]
        
        # =====================================================================
        # 步骤 3: 传入 Transformer Encoder
        # =====================================================================
        encoded_features = self.temporal_encoder(
            features_flat,
            src_key_padding_mask=padding_mask
        )  # [B*N, 20, 128]
        
        # =====================================================================
        # 步骤 4: 提取最后时刻的状态
        # =====================================================================
        # 取 T=19 时刻（索引为 -1）的输出
        last_state = encoded_features[:, -1, :]  # [B*N, 128]
        
        # =====================================================================
        # 步骤 5: 重塑回 [B, N, 128]
        # =====================================================================
        output = last_state.reshape(B, N, D)  # [B, N, 128]
        
        return output


class DenseALEncoder(nn.Module):
    """
    Agent-Lane 交互模块
    
    使用 Cross-Attention 实现 Agent 对 Lane 特征的查询和融合
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
        super(DenseALEncoder, self).__init__()
        
        # 相对位置编码器
        self.rel_pe = RelativePositionEmbedding(embed_dim=embed_dim)
        
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
        agent_features: torch.Tensor,           # [B, N, 128]
        agent_positions: torch.Tensor,          # [B, N, 2]
        agent_heading: torch.Tensor,            # [B, N]
        lane_features: torch.Tensor,            # [B, L, 128]
        lane_positions: torch.Tensor,           # [B, L, 2]
        lane_mask: torch.Tensor                 # [B, L]
    ) -> torch.Tensor:
        """
        Args:
            agent_features: 智能体特征，Shape [B, N, 128]
            agent_positions: 智能体位置，Shape [B, N, 2]
            agent_heading: 智能体朝向，Shape [B, N]
            lane_features: 车道特征，Shape [B, L, 128]
            lane_positions: 车道位置（中心点或起始点），Shape [B, L, 2]
            lane_mask: 车道有效掩码，Shape [B, L] (Bool, True=有效)
        
        Returns:
            融合后的智能体特征，Shape [B, N, 128]
        """
        # =====================================================================
        # 步骤 1: 计算局部几何和掩码
        # =====================================================================
        # AL 交互：center=Agent, target=Lane
        local_rel_pos, attn_mask = get_local_geometry_and_mask(
            center_pos=agent_positions,    # [B, N, 2]
            center_heading=agent_heading,  # [B, N]
            target_pos=lane_positions,     # [B, L, 2]
            target_mask=lane_mask,         # [B, L]
            radius=50.0
        )
        # local_rel_pos: [B, N, L, 2]
        # attn_mask: [B, N, L]
        
        # =====================================================================
        # 步骤 2: 生成相对位置编码
        # =====================================================================
        rel_pe = self.rel_pe(local_rel_pos)  # [B, N, L, 128]
        
        # =====================================================================
        # 步骤 3: Cross-Attention (Pre-LN + 残差)
        # =====================================================================
        # Pre-LN
        agent_features_norm = self.norm1(agent_features)  # [B, N, 128]
        
        # Cross-Attention: Query=Agent, Key=Value=Lane
        attn_out = self.attn(
            query=agent_features_norm,
            key=lane_features,
            value=lane_features,
            rel_pe=rel_pe,
            attn_mask=attn_mask
        )  # [B, N, 128]
        
        # 残差连接
        agent_features = agent_features + attn_out  # [B, N, 128]
        
        # =====================================================================
        # 步骤 4: Feed-Forward Network (Pre-LN + 残差)
        # =====================================================================
        # Pre-LN
        agent_features_norm = self.norm2(agent_features)  # [B, N, 128]
        
        # FFN
        ffn_out = self.ffn(agent_features_norm)  # [B, N, 128]
        
        # 残差连接
        agent_features = agent_features + ffn_out  # [B, N, 128]
        
        return agent_features


class LocalEncoder(nn.Module):
    """
    局部编码器顶层模块
    
    组装完整的局部编码流水线：
    1. 节点特征初始化（Agent & Lane）
    2. AA 交互（时间维度折叠）
    3. 时序压缩
    4. AL 交互
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
        super(LocalEncoder, self).__init__()
        
        # 节点特征初始化模块
        self.agent_emb = AgentNodeEmbedding(embed_dim=embed_dim)
        self.lane_emb = LaneNodeEmbedding(embed_dim=embed_dim)
        
        # AA 交互模块
        self.aa_encoder = DenseAAEncoder(embed_dim, num_heads, dropout)
        
        # 时序压缩模块
        self.temporal_encoder = DenseTemporalEncoder(embed_dim, num_heads, dropout)
        
        # AL 交互模块
        self.al_encoder = DenseALEncoder(embed_dim, num_heads, dropout)
    
    def forward(
        self,
        # Agent 相关输入
        agent_history_speed: torch.Tensor,       # [B, N, 20, 2]
        agent_history_positions: torch.Tensor,   # [B, N, 20, 2]
        agent_history_heading: torch.Tensor,     # [B, N, 20]
        agent_history_mask: torch.Tensor,        # [B, N, 20]
        agent_type: torch.Tensor,                # [B, N]
        # Lane 相关输入
        map_lane_positions: torch.Tensor,        # [B, L, 10, 2]
        map_is_intersection: torch.Tensor,       # [B, L]
        map_turn_direction: torch.Tensor,        # [B, L]
        map_traffic_control: torch.Tensor,       # [B, L]
        map_lane_mask: torch.Tensor              # [B, L]
    ) -> torch.Tensor:
        """
        Args:
            agent_history_speed: 智能体历史速度，Shape [B, N, 20, 2]
            agent_history_positions: 智能体历史位置，Shape [B, N, 20, 2]
            agent_history_heading: 智能体历史朝向，Shape [B, N, 20]
            agent_history_mask: 智能体历史有效掩码，Shape [B, N, 20] (Bool)
            map_lane_positions: 车道线点坐标，Shape [B, L, 10, 2]
            map_is_intersection: 交叉路口标志，Shape [B, L]
            map_turn_direction: 转向方向，Shape [B, L]
            map_traffic_control: 交通管制，Shape [B, L]
            map_lane_mask: 车道有效掩码，Shape [B, L] (Bool)
        
        Returns:
            local_embed: 局部编码特征，Shape [B, N, 128]
        """
        # =====================================================================
        # 步骤 1: 节点特征初始化
        # =====================================================================
        # Agent 节点初始化
        agent_features = self.agent_emb(agent_history_speed)  # [B, N, 20, 128]
        
        # Lane 节点初始化
        lane_features = self.lane_emb(
            map_lane_positions,
            map_is_intersection,
            map_turn_direction,
            map_traffic_control
        )  # [B, L, 128]
        
        # =====================================================================
        # 步骤 2: AA 交互（时间维度折叠）
        # =====================================================================
        agent_features = self.aa_encoder(
            agent_history_positions,
            agent_history_heading,
            agent_history_mask,
            agent_features
        )  # [B, N, 20, 128]
        
        # =====================================================================
        # 步骤 3: 时序压缩
        # =====================================================================
        agent_features = self.temporal_encoder(
            agent_features,
            agent_history_mask
        )  # [B, N, 128]
        
        # =====================================================================
        # 步骤 4: AL 交互
        # =====================================================================
        # 提取当前时刻（T=19）的 Agent 位置和朝向
        agent_current_pos = agent_history_positions[:, :, -1, :]  # [B, N, 2]
        agent_current_heading = agent_history_heading[:, :, -1]   # [B, N]
        
        # 计算车道中心点位置（取 10 个点的平均）
        lane_center_pos = map_lane_positions.mean(dim=2)  # [B, L, 2]
        
        # AL 交互
        local_embed = self.al_encoder(
            agent_features,
            agent_current_pos,
            agent_current_heading,
            lane_features,
            lane_center_pos,
            map_lane_mask
        )  # [B, N, 128]
        
        return local_embed
