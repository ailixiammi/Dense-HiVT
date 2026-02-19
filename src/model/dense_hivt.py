"""
Dense-HiVT 顶层模型封装
将局部编码器、全局交互器和多模态解码器优雅地串联，提供统一的前向传播接口
"""

import torch
import torch.nn as nn
from typing import Dict

from .local_encoder import LocalEncoder
from .global_interactor import GlobalInteractor
from .decoder import MultimodalDecoder


class DenseHiVT(nn.Module):
    """
    Dense-HiVT 完整模型
    
    架构流程:
    1. Local Encoder: 提取 Agent-Lane 局部时序特征
    2. Global Interactor: 建模 Agent 之间的全局交互
    3. Multimodal Decoder: 输出多模态未来轨迹预测
    
    特点:
    - 高度解耦的模块化设计
    - 统一的字典输入输出接口
    - 清晰的数据流传递
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        global_layers: int = 3,
        num_modes: int = 6,
        future_steps: int = 30,
        dropout: float = 0.1
    ):
        """
        初始化 Dense-HiVT 模型
        
        Args:
            embed_dim: 特征嵌入维度，默认 128
            num_heads: 多头注意力头数，默认 8
            global_layers: 全局交互 Transformer 层数，默认 3
            num_modes: 多模态预测数量 K，默认 6
            future_steps: 未来预测帧数 F，默认 30
            dropout: Dropout 比率，默认 0.1
        """
        super(DenseHiVT, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.global_layers = global_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.dropout = dropout
        
        # =====================================================================
        # 组件 1: 局部编码器（Local Encoder）
        # =====================================================================
        self.local_encoder = LocalEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # =====================================================================
        # 组件 2: 全局交互器（Global Interactor）
        # =====================================================================
        self.global_interactor = GlobalInteractor(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=global_layers,
            dropout=dropout
        )
        
        # =====================================================================
        # 组件 3: 多模态解码器（Multimodal Decoder）
        # =====================================================================
        self.decoder = MultimodalDecoder(
            embed_dim=embed_dim,
            num_modes=num_modes,
            future_steps=future_steps
        )
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Dense-HiVT 前向传播
        
        设计理念:
        - 采用字典输入输出接口，提升代码的可扩展性和可维护性
        - 数据流清晰，每个阶段的输出形状都有明确标注
        - 适配未来复杂的训练/推理流程
        
        Args:
            data: 包含所有输入数据的字典，必须包含以下键:
                # Agent 历史特征
                - 'agent_history_positions': [B, N, 20, 2] - 历史位置轨迹
                - 'agent_history_speed': [B, N, 20, 2] - 历史速度 (vx, vy)
                - 'agent_history_heading': [B, N, 20] - 历史朝向角度 (弧度)
                - 'agent_history_mask': [B, N, 20] - 历史有效性掩码 (Bool)
                - 'agent_type': [B, N] - Agent 类别 (0=AV, 1=Agent, 2=Others)
                
                # Lane 地图特征
                - 'map_lane_positions': [B, L, 10, 2] - Lane 采样点位置
                - 'map_is_intersection': [B, L] - 是否交叉路口 (0/1/2)
                - 'map_turn_direction': [B, L] - 转向方向 (0/1/2)
                - 'map_traffic_control': [B, L] - 交通控制 (0/1)
                - 'map_lane_mask': [B, L] - Lane 有效性掩码 (Bool)
        
        Returns:
            包含预测结果的字典:
                - 'pi': [B, N, 6] - 模态分类 Logits (未归一化)
                - 'loc': [B, N, 6, 30, 4] - 轨迹预测参数 [μ_x, μ_y, b_x, b_y]
        """
        # =====================================================================
        # 步骤 1: 提取输入特征
        # =====================================================================
        # Agent 历史特征
        agent_history_positions = data['agent_history_positions']  # [B, N, 20, 2]
        agent_history_speed = data['agent_history_speed']          # [B, N, 20, 2]
        
        # 从当前朝向扩展为历史序列（假设恒定朝向）
        agent_current_heading_single = data['agent_heading']  # [B, N]
        agent_history_heading = agent_current_heading_single.unsqueeze(-1).expand(-1, -1, 20)  # [B, N, 20]
        
        agent_history_mask = data['agent_history_positions_mask']  # [B, N, 20]
        agent_type = data['agent_type']                            # [B, N]
        
        # Lane 地图特征
        map_lane_positions = data['map_lane_positions']            # [B, L, 10, 2]
        map_is_intersection = data['map_is_intersection']          # [B, L]
        map_turn_direction = data['map_turn_direction']            # [B, L]
        map_traffic_control = data['map_traffic_control']          # [B, L]
        map_lane_mask = data['map_lane_positions_mask'].any(dim=-1)# [B, L]
        
        # =====================================================================
        # 步骤 2: Local Encoder - 提取局部时序特征
        # =====================================================================
        local_embed = self.local_encoder(
            agent_history_positions=agent_history_positions,
            agent_history_speed=agent_history_speed,
            agent_history_heading=agent_history_heading,
            agent_history_mask=agent_history_mask,
            agent_type=agent_type,
            map_lane_positions=map_lane_positions,
            map_is_intersection=map_is_intersection,
            map_turn_direction=map_turn_direction,
            map_traffic_control=map_traffic_control,
            map_lane_mask=map_lane_mask
        )  # Shape: [B, N, 128]

        # =====================================================================
        # 步骤 3: 提取当前时刻状态（t=19，即最后一个历史帧）
        # =====================================================================
        # 全局交互器需要当前时刻的位置、朝向和掩码来计算相对位置编码
        agent_current_pos = agent_history_positions[:, :, -1, :]  # Shape: [B, N, 2]
        agent_current_heading = agent_current_heading_single       # Shape: [B, N] (直接使用原始朝向)
        agent_current_mask = agent_history_mask[:, :, -1]          # Shape: [B, N]
        
        # =====================================================================
        # 步骤 4: Global Interactor - 建模全局交互
        # =====================================================================
        global_embed = self.global_interactor(
            local_embed=local_embed,
            agent_positions=agent_current_pos,
            agent_heading=agent_current_heading,
            agent_mask=agent_current_mask
        )  # Shape: [B, N, 128]
        
        # =====================================================================
        # 步骤 5: Multimodal Decoder - 多模态未来解码
        # =====================================================================
        pi, loc = self.decoder(
            global_embed=global_embed,
            agent_type=agent_type
        )  # pi: [B, N, 6], loc: [B, N, 6, 30, 4]
        
        # =====================================================================
        # 步骤 6: 返回预测结果
        # =====================================================================
        return {
            'pi': pi,    # 模态分类 Logits, Shape: [B, N, 6]
            'loc': loc   # 轨迹预测参数, Shape: [B, N, 6, 30, 4]
        }