"""
Dense-HiVT 特征映射模块
将原始运动学和地图特征映射到高维连续空间，零 PyG 依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SingleInputEmbedding(nn.Module):
    """
    基础映射块：将任意维度的输入特征映射到目标维度
    结构：Linear(in_channel, 64) -> LayerNorm(64) -> ReLU -> Linear(64, out_channel)
    
    支持任意维度的输入张量，nn.Linear 会自动对最后一个维度进行映射
    例如：输入 [B, N, 20, 2] 会被映射为 [B, N, 20, out_channel]
    """
    
    def __init__(self, in_channel: int, out_channel: int):
        """
        Args:
            in_channel: 输入特征维度
            out_channel: 输出特征维度（通常为 128）
        """
        super(SingleInputEmbedding, self).__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，Shape [..., in_channel]
        
        Returns:
            输出张量，Shape [..., out_channel]
        """
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    """
    多特征融合模块：处理连续特征和离散特征的混合输入
    
    策略：
    1. 为每个连续特征构建独立的 SingleInputEmbedding
    2. 为每个离散特征构建独立的 nn.Embedding
    3. 将所有映射后的张量进行 Element-wise Sum（逐元素相加）
    """
    
    def __init__(
        self,
        in_channels: List[int],           # 连续特征的维度列表
        num_classes: List[int],           # 离散特征的类别数列表
        out_channel: int                  # 输出特征维度
    ):
        """
        Args:
            in_channels: 连续特征的输入维度列表，例如 [2] 表示一个 2D 向量
            num_classes: 离散特征的类别数列表，例如 [3, 3, 2] 表示三个离散特征
            out_channel: 输出特征维度（通常为 128）
        """
        super(MultipleInputEmbedding, self).__init__()
        
        # 为每个连续特征构建独立的映射层
        self.continuous_embs = nn.ModuleList([
            SingleInputEmbedding(in_channel, out_channel) 
            for in_channel in in_channels
        ])
        
        # 为每个离散特征构建独立的嵌入层
        self.discrete_embs = nn.ModuleList([
            nn.Embedding(num_class, out_channel) 
            for num_class in num_classes
        ])
    
    def forward(
        self, 
        continuous_features: List[torch.Tensor], 
        discrete_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            continuous_features: 连续特征列表，每个元素 Shape [..., in_channel_i]
            discrete_features: 离散特征列表，每个元素 Shape [...] (需要是整数索引)
        
        Returns:
            融合后的特征，Shape [..., out_channel]
        """
        # 处理连续特征
        continuous_embeds = [
            emb(feat) for emb, feat in zip(self.continuous_embs, continuous_features)
        ]  # 每个元素: [..., out_channel]
        
        # 处理离散特征（确保类型为 torch.long）
        discrete_embeds = [
            emb(feat.long()) for emb, feat in zip(self.discrete_embs, discrete_features)
        ]  # 每个元素: [..., out_channel]
        
        # Element-wise Sum：逐元素相加所有特征
        all_embeds = continuous_embeds + discrete_embeds
        output = torch.stack(all_embeds, dim=0).sum(dim=0)  # [..., out_channel]
        
        return output


class AgentNodeEmbedding(nn.Module):
    """
    智能体节点初始化模块：仅处理历史速度，完全忽略类别信息
    
    输入：agent_history_speed [B, N, 20, 2]
    输出：agent_base_features [B, N, 20, 128]
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Args:
            embed_dim: 嵌入维度，默认 128
        """
        super(AgentNodeEmbedding, self).__init__()
        
        # 速度向量映射：2D -> embed_dim
        self.speed_emb = SingleInputEmbedding(in_channel=2, out_channel=embed_dim)
    
    def forward(self, agent_history_speed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_history_speed: 智能体历史速度，Shape [B, N, 20, 2]
        
        Returns:
            agent_base_features: 智能体基础特征，Shape [B, N, 20, 128]
        """
        # 直接映射速度向量到高维空间
        agent_base_features = self.speed_emb(agent_history_speed)  # [B, N, 20, 128]
        
        return agent_base_features


class LaneNodeEmbedding(nn.Module):
    """
    车道节点初始化模块：融合几何向量和离散属性
    
    输入：
        - map_lane_positions: [B, L, 10, 2] (10 个离散点)
        - map_is_intersection: [B, L] (交叉路口标志)
        - map_turn_direction: [B, L] (转向方向)
        - map_traffic_control: [B, L] (交通管制)
    
    输出：
        - lane_base_features: [B, L, 128]
    
    处理流程：
        1. 计算方向向量：相邻点差分 -> [B, L, 9, 2]
        2. L2 归一化
        3. 离散特征广播到序列维度
        4. 多特征融合（MultipleInputEmbedding）
        5. Max Pooling 降维到 [B, L, 128]
    """
    
    def __init__(
        self, 
        embed_dim: int = 128,
        num_is_intersection: int = 3,
        num_turn_direction: int = 3,
        num_traffic_control: int = 2
    ):
        """
        Args:
            embed_dim: 嵌入维度，默认 128
            num_is_intersection: 交叉路口类别数，默认 3
            num_turn_direction: 转向方向类别数，默认 3
            num_traffic_control: 交通管制类别数，默认 2
        """
        super(LaneNodeEmbedding, self).__init__()
        
        # 多特征融合模块：1 个连续特征（方向向量）+ 3 个离散特征
        self.lane_emb = MultipleInputEmbedding(
            in_channels=[2],  # 方向向量 2D
            num_classes=[num_is_intersection, num_turn_direction, num_traffic_control],
            out_channel=embed_dim
        )
    
    def forward(
        self,
        map_lane_positions: torch.Tensor,      # [B, L, 10, 2]
        map_is_intersection: torch.Tensor,     # [B, L]
        map_turn_direction: torch.Tensor,      # [B, L]
        map_traffic_control: torch.Tensor      # [B, L]
    ) -> torch.Tensor:
        """
        Args:
            map_lane_positions: 车道线点坐标，Shape [B, L, 10, 2]
            map_is_intersection: 交叉路口标志，Shape [B, L] (Long 索引)
            map_turn_direction: 转向方向，Shape [B, L] (Long 索引)
            map_traffic_control: 交通管制，Shape [B, L] (Long 索引)
        
        Returns:
            lane_base_features: 车道基础特征，Shape [B, L, 128]
        """
        # =====================================================================
        # 步骤 1: 计算方向向量（相邻点差分）
        # =====================================================================
        # 计算相邻点的差分向量
        direction_vectors = map_lane_positions[:, :, 1:, :] - map_lane_positions[:, :, :-1, :]  
        # [B, L, 9, 2]
        
        # =====================================================================
        # 步骤 2: L2 归一化
        # =====================================================================
        # 计算每个向量的 L2 范数（展开并加安全垫，防止 sqrt(0) 导致梯度 NaN）
        norms = torch.sqrt(torch.sum(direction_vectors**2, dim=-1, keepdim=True) + 1e-6)  # [B, L, 9, 1]
        
        # 归一化（norms 已经有安全垫，无需再加）
        direction_vectors_normalized = direction_vectors / norms  # [B, L, 9, 2]
        
        # =====================================================================
        # 步骤 3: 离散特征广播到序列维度（确保类型为 torch.long）
        # =====================================================================
        # 将离散特征从 [B, L] 扩展为 [B, L, 9] 以对齐点序列维度
        # 注意：nn.Embedding 要求输入必须是 torch.long 类型
        map_is_intersection_expanded = map_is_intersection.long().unsqueeze(-1).expand(-1, -1, 9)  # [B, L, 9]
        map_turn_direction_expanded = map_turn_direction.long().unsqueeze(-1).expand(-1, -1, 9)    # [B, L, 9]
        map_traffic_control_expanded = map_traffic_control.long().unsqueeze(-1).expand(-1, -1, 9)  # [B, L, 9]
        
        # =====================================================================
        # 步骤 4: 多特征融合
        # =====================================================================
        # 连续特征列表
        continuous_features = [direction_vectors_normalized]  # [B, L, 9, 2]
        
        # 离散特征列表
        discrete_features = [
            map_is_intersection_expanded,   # [B, L, 9]
            map_turn_direction_expanded,    # [B, L, 9]
            map_traffic_control_expanded    # [B, L, 9]
        ]
        
        # 融合所有特征
        lane_features_seq = self.lane_emb(continuous_features, discrete_features)  # [B, L, 9, 128]
        
        # =====================================================================
        # 步骤 5: Max Pooling 降维
        # =====================================================================
        # 在序列维度（S=9）上执行 Max Pooling
        lane_base_features = lane_features_seq.max(dim=2)[0]  # [B, L, 128]
        
        return lane_base_features


class RelativePositionEmbedding(nn.Module):
    """
    相对位置编码器：将局部相对坐标映射为位置编码
    
    用于后续的 Attention 交互，将旋转好的局部相对坐标转换为高维特征
    
    输入：local_rel_pos [B, N, M, 2]
    输出：position_embedding [B, N, M, 128]
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Args:
            embed_dim: 嵌入维度，默认 128
        """
        super(RelativePositionEmbedding, self).__init__()
        
        # 相对坐标映射：2D -> embed_dim
        self.pos_emb = SingleInputEmbedding(in_channel=2, out_channel=embed_dim)
    
    def forward(self, local_rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_rel_pos: 局部相对坐标，Shape [B, N, M, 2]
        
        Returns:
            position_embedding: 位置编码，Shape [B, N, M, 128]
        """
        # 直接映射相对坐标到高维空间
        position_embedding = self.pos_emb(local_rel_pos)  # [B, N, M, 128]
        
        return position_embedding