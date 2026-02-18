#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argoverse 1.1 Offline Preprocessing Script (Dense Tensor Version)
生成包含完整语义特征的稠密 Tensor，并保存为 .pt 文件
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Argoverse Map API
try:
    from argoverse.map_representation.map_api import ArgoverseMap
except ImportError:
    print("[ERROR] 无法导入 argoverse 库")
    print("请安装: pip install argoverse")
    sys.exit(1)


# ==================== 全局配置 ====================
class GlobalConfig:
    """全局配置常量"""
    MAX_AGENTS = 64
    MAX_LANES = 256
    MAX_POINTS_PER_LANE = 10
    SEARCH_RADIUS = 100.0
    
    # Agent 类型映射
    AGENT_TYPE_MAP = {
        'AV': 0,
        'AGENT': 1,
        'OTHERS': 2,
        '<PAD>': -1
    }
    
    # Lane 转向映射
    TURN_DIRECTION_MAP = {
        'NONE': 0,
        'LEFT': 1,
        'RIGHT': 2
    }
    
    # 时间步配置
    HISTORY_STEPS = 20  # 历史轨迹: 0-19
    FUTURE_STEPS = 30   # 未来轨迹: 20-49
    TOTAL_STEPS = 50
    OBS_TIMESTEP = 19   # 观测时刻（第20帧，索引19）


# ==================== 坐标转换模块 ====================
class CoordinateTransform:
    """坐标转换工具类"""
    
    @staticmethod
    def compute_rotation_matrix(theta: float) -> torch.Tensor:
        """
        计算旋转矩阵
        
        Args:
            theta: 旋转角度(弧度)
            
        Returns:
            [2, 2] 旋转矩阵
        """
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        rotate_mat = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], dtype=torch.float32)
        
        return rotate_mat
    
    @staticmethod
    def transform_to_local(positions: torch.Tensor, 
                          origin: torch.Tensor, 
                          rotate_mat: torch.Tensor) -> torch.Tensor:
        """
        将全局坐标转换到局部坐标系
        
        Args:
            positions: [..., 2] 全局坐标
            origin: [2] 原点坐标
            rotate_mat: [2, 2] 旋转矩阵
            
        Returns:
            [..., 2] 局部坐标
        """
        # 平移到原点
        centered = positions - origin
        
        # 旋转
        local = torch.matmul(centered, rotate_mat)
        
        return local
    
    @staticmethod
    def compute_speed_vectors(positions: torch.Tensor, 
                             mask: torch.Tensor,
                             rotate_mat: torch.Tensor) -> torch.Tensor:
        """
        计算速度向量（局部坐标系）
        
        Args:
            positions: [N, T, 2] 位置（已转换到局部坐标系）
            mask: [N, T] 有效性mask
            rotate_mat: [2, 2] 旋转矩阵（用于速度）
            
        Returns:
            [N, T, 2] 速度向量（局部坐标系）
        """
        N, T, _ = positions.shape
        speed = torch.zeros(N, T, 2, dtype=torch.float32)
        
        # 计算差分: v_t = p_t - p_{t-1}
        for t in range(1, T):
            # 只计算两个时刻都有效的速度
            valid = mask[:, t-1] & mask[:, t]
            speed[:, t] = torch.where(
                valid.unsqueeze(-1),
                positions[:, t] - positions[:, t-1],
                torch.zeros(N, 2)
            )
        
        # t=0 的速度设为0
        speed[:, 0] = 0.0
        
        return speed


# ==================== Agent 特征提取模块 ====================
class AgentFeatureExtractor:
    """Agent 特征提取器"""
    
    def __init__(self, config: GlobalConfig, am: ArgoverseMap):
        self.config = config
        self.am = am
    
    def extract_features(self, 
                        df: pd.DataFrame,
                        timestamps: List[int],
                        origin: torch.Tensor,
                        rotate_mat: torch.Tensor,
                        theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 Agent 的所有特征（使用双锚点筛选策略）
        
        Args:
            df: 场景的完整 DataFrame
            timestamps: 时间戳列表（已排序）
            origin: [2] 原点坐标（全局，AV在t=19的位置）
            rotate_mat: [2, 2] 旋转矩阵
            theta: AV 在 t=19 的朝向（弧度）
            
        Returns:
            包含所有 Agent 特征的字典
        """
        # 获取 t=19 时刻的数据用于筛选
        obs_timestamp = timestamps[self.config.OBS_TIMESTEP]
        df_obs = df[df['TIMESTAMP'] == obs_timestamp]
        
        # 获取历史时刻的有效 Agent（必须在前20帧至少出现过）
        historical_timestamps = timestamps[:self.config.HISTORY_STEPS]
        historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
        candidate_ids = list(historical_df['TRACK_ID'].unique())
        
        # 只保留在 t=19 时刻存在的 Agent
        obs_ids = set(df_obs['TRACK_ID'].unique())
        actor_ids = [aid for aid in candidate_ids if aid in obs_ids]
        
        if len(actor_ids) == 0:
            return self._create_empty_agent_features()
        
        # 应用双锚点筛选策略
        actor_ids = self._apply_dual_anchor_filtering(df, df_obs, actor_ids, origin)
        num_agents = len(actor_ids)
        
        # 获取城市名称
        city = df_obs['CITY_NAME'].iloc[0]
        
        # 初始化特征容器
        positions_global = torch.zeros(num_agents, self.config.TOTAL_STEPS, 2, dtype=torch.float32)
        history_mask = torch.zeros(num_agents, self.config.HISTORY_STEPS, dtype=torch.bool)
        future_mask = torch.zeros(num_agents, self.config.FUTURE_STEPS, dtype=torch.bool)
        agent_types = torch.full((num_agents,), -1, dtype=torch.long)
        is_target = torch.zeros(num_agents, dtype=torch.bool)
        agent_headings = []  # 【新增】朝向列表
        
        # 遍历每个 Agent
        for idx, actor_id in enumerate(actor_ids):
            actor_df = df[df['TRACK_ID'] == actor_id]
            
            # 获取类型
            object_type = actor_df['OBJECT_TYPE'].iloc[0]
            agent_types[idx] = self.config.AGENT_TYPE_MAP.get(object_type, 2)
            
            # 判断是否为目标 Agent
            if object_type == 'AGENT':
                is_target[idx] = True
            
            # 提取轨迹
            for _, row in actor_df.iterrows():
                timestamp = row['TIMESTAMP']
                if timestamp not in timestamps:
                    continue
                
                t = timestamps.index(timestamp)
                positions_global[idx, t] = torch.tensor([row['X'], row['Y']], dtype=torch.float32)
                
                if t < self.config.HISTORY_STEPS:
                    history_mask[idx, t] = True
                else:
                    future_mask[idx, t - self.config.HISTORY_STEPS] = True
            
            # 【优化】AV 直接复用外部传入的 robust theta
            if object_type == 'AV':
                agent_headings.append(theta.item())
                continue
            
            # 【新增】计算当前 Agent 的朝向（在循环内部）
            heading = self._compute_agent_heading(
                actor_df=actor_df,
                timestamps=timestamps,
                positions_global=positions_global[idx],
                history_mask=history_mask[idx],
                city=city,
                av_heading=theta
            )
            agent_headings.append(heading)
        
        # 转换朝向列表为 Tensor
        agent_headings = torch.tensor(agent_headings, dtype=torch.float32)  # [N]
        
        # 坐标转换到局部坐标系
        positions_local = CoordinateTransform.transform_to_local(
            positions_global, origin, rotate_mat
        )
        
        # 提取历史和未来位置
        history_positions = positions_local[:, :self.config.HISTORY_STEPS]  # [N, 20, 2]
        future_positions = positions_local[:, self.config.HISTORY_STEPS:]  # [N, 30, 2]
        
        # 计算速度（历史部分）
        all_mask = torch.cat([history_mask, future_mask], dim=1)  # [N, 50]
        history_speed = CoordinateTransform.compute_speed_vectors(
            positions_local, all_mask, rotate_mat
        )[:, :self.config.HISTORY_STEPS]  # [N, 20, 2]
        
        # 对于不可见的 Agent，将未来位置 mask 设为 False
        for idx in range(num_agents):
            if not history_mask[idx, self.config.OBS_TIMESTEP]:
                future_mask[idx, :] = False
        
        return {
            'agent_history_positions': history_positions,
            'agent_history_positions_mask': history_mask,
            'agent_history_speed': history_speed,
            'agent_future_positions': future_positions,
            'agent_future_positions_mask': future_mask,
            'agent_type': agent_types,
            'agent_is_target': is_target,
            'agent_heading': agent_headings,  # 【新增】
            'num_valid_agents': num_agents
        }
    
    def _compute_agent_heading(self,
                              actor_df: pd.DataFrame,
                              timestamps: List[int],
                              positions_global: torch.Tensor,
                              history_mask: torch.Tensor,
                              city: str,
                              av_heading: torch.Tensor) -> float:
        """
        计算单个 Agent 在 T=19 时刻的朝向
        
        策略：
        1. 检查历史 20 帧的移动距离
        2. 如果移动 > 0.1m: 使用几何差分
        3. 如果静止: 查询最近车道切线方向
        
        Args:
            actor_df: 该 Agent 的 DataFrame
            timestamps: 时间戳列表
            positions_global: [50, 2] 该 Agent 的全局轨迹
            history_mask: [20] 该 Agent 的历史有效性 mask
            city: 城市名称
            av_heading: AV 的朝向（用于 fallback）
            
        Returns:
            朝向角度（弧度）
        """
        import math
        
        EPSILON = 0.1  # 移动阈值（米）
        OBS_TIMESTEP = self.config.OBS_TIMESTEP
        
        # Step 1: 提取历史轨迹（T=0~19）
        history_positions = positions_global[:self.config.HISTORY_STEPS]  # [20, 2]
        valid_positions = history_positions[history_mask]  # 只取有效点
        
        if len(valid_positions) < 2:
            # 没有足够的点，返回 AV 朝向
            return av_heading.item()
        
        # Step 2: 计算最大位移（轨迹包络）
        max_displacement = (valid_positions.max(dim=0)[0] - 
                           valid_positions.min(dim=0)[0]).norm().item()
        
        # Step 3: 判断运动 vs 静止
        if max_displacement > EPSILON:
            # 【运动物体】使用几何差分
            # 尝试从 T=19 往前找最近的有效差分
            for t in range(OBS_TIMESTEP, 0, -1):
                if history_mask[t] and history_mask[t-1]:
                    dx = positions_global[t, 0] - positions_global[t-1, 0]
                    dy = positions_global[t, 1] - positions_global[t-1, 1]
                    displacement = torch.sqrt(dx**2 + dy**2).item()
                    if displacement > 1e-3:  # 避免除零
                        return math.atan2(dy.item(), dx.item())
            
            # 如果 T=19 附近都静止，沿用更早的移动方向
            for t in range(OBS_TIMESTEP-1, 0, -1):
                if history_mask[t] and history_mask[t-1]:
                    dx = positions_global[t, 0] - positions_global[t-1, 0]
                    dy = positions_global[t, 1] - positions_global[t-1, 1]
                    displacement = torch.sqrt(dx**2 + dy**2).item()
                    if displacement > 1e-3:
                        return math.atan2(dy.item(), dx.item())
        
        # 【静止物体】查询地图
        # 获取 T=19 时刻的位置
        if not history_mask[OBS_TIMESTEP]:
            return av_heading.item()
        
        agent_x = positions_global[OBS_TIMESTEP, 0].item()
        agent_y = positions_global[OBS_TIMESTEP, 1].item()
        
        try:
            # 调用地图 API 查询附近车道
            lane_ids = self.am.get_lane_ids_in_xy_bbox(
                agent_x, agent_y, city, query_search_range_manhattan=5.0
            )
            
            if len(lane_ids) > 0:
                # 获取最近车道的中心线
                centerline = self.am.get_lane_segment_centerline(
                    lane_ids[0], city
                )
                
                if centerline is not None and len(centerline) >= 2:
                    # 找到最近点
                    dists = np.linalg.norm(
                        centerline[:, :2] - np.array([agent_x, agent_y]), 
                        axis=1
                    )
                    nearest_idx = np.argmin(dists)
                    
                    # 计算切线方向（顺着车流）
                    if nearest_idx < len(centerline) - 1:
                        v = centerline[nearest_idx + 1, :2] - centerline[nearest_idx, :2]
                    else:
                        v = centerline[nearest_idx, :2] - centerline[nearest_idx - 1, :2]
                    
                    return math.atan2(v[1], v[0])
        
        except Exception:
            pass
        
        # Fallback: 返回 AV 朝向
        return av_heading.item()
    
    def _apply_dual_anchor_filtering(self,
                                     df: pd.DataFrame,
                                     df_obs: pd.DataFrame,
                                     candidate_ids: List[int],
                                     av_position: torch.Tensor) -> List[int]:
        """
        应用双锚点筛选策略：基于 AV 和 Target Agent 的相关性评分
        
        策略：
        1. AV (Type 0) 必须在 Index 0
        2. Target Agent (Type 1) 必须在 Index 1
        3. 其他 Agent 按相关性评分排序：score = dist_to_target + 0.5 * dist_to_av
        
        Args:
            df: 完整 DataFrame
            df_obs: t=19 时刻的 DataFrame
            candidate_ids: 候选 Agent ID 列表
            av_position: AV 在 t=19 的位置 [2]
            
        Returns:
            排序后的 Agent ID 列表（最多 MAX_AGENTS 个）
        """
        # 分离 AV、Target 和 Others
        av_id = None
        target_id = None
        other_ids = []
        
        for actor_id in candidate_ids:
            actor_row = df_obs[df_obs['TRACK_ID'] == actor_id].iloc[0]
            obj_type = actor_row['OBJECT_TYPE']
            
            if obj_type == 'AV':
                av_id = actor_id
            elif obj_type == 'AGENT':
                if target_id is None:  # 只取第一个 Target
                    target_id = actor_id
                else:
                    other_ids.append(actor_id)  # 多余的 AGENT 视为 OTHERS
            else:
                other_ids.append(actor_id)
        
        # 如果没有 AV，返回空（异常情况）
        if av_id is None:
            return []
        
        # 如果没有 Target Agent，降级为单锚点筛选（仅基于 AV）
        if target_id is None:
            return self._fallback_av_only_filtering(df_obs, av_id, other_ids, av_position)
        
        # 获取 Target Agent 的位置
        target_row = df_obs[df_obs['TRACK_ID'] == target_id].iloc[0]
        target_position = torch.tensor([target_row['X'], target_row['Y']], dtype=torch.float32)
        
        # 计算其他 Agent 的相关性评分
        scored_others = []
        for actor_id in other_ids:
            actor_row = df_obs[df_obs['TRACK_ID'] == actor_id].iloc[0]
            pos = torch.tensor([actor_row['X'], actor_row['Y']], dtype=torch.float32)
            
            dist_to_av = torch.norm(pos - av_position).item()
            dist_to_target = torch.norm(pos - target_position).item()
            
            # 相关性评分：离 Target 或 AV 越近越重要
            score = dist_to_target + 0.5 * dist_to_av
            scored_others.append((actor_id, score))
        
        # 按评分排序（分数越小越重要）
        scored_others.sort(key=lambda x: x[1])
        
        # 选取前 MAX_AGENTS - 2 个
        max_others = self.config.MAX_AGENTS - 2
        selected_others = [aid for aid, _ in scored_others[:max_others]]
        
        # 组合：[AV, Target, Selected_Others...]
        return [av_id, target_id] + selected_others
    
    def _fallback_av_only_filtering(self,
                                     df_obs: pd.DataFrame,
                                     av_id: int,
                                     other_ids: List[int],
                                     av_position: torch.Tensor) -> List[int]:
        """
        降级筛选策略：仅基于到 AV 的距离（无 Target Agent 时使用）
        
        Args:
            df_obs: t=19 时刻的 DataFrame
            av_id: AV 的 Track ID
            other_ids: 其他 Agent ID 列表
            av_position: AV 位置 [2]
            
        Returns:
            排序后的 Agent ID 列表
        """
        scored_others = []
        for actor_id in other_ids:
            actor_row = df_obs[df_obs['TRACK_ID'] == actor_id].iloc[0]
            pos = torch.tensor([actor_row['X'], actor_row['Y']], dtype=torch.float32)
            dist = torch.norm(pos - av_position).item()
            scored_others.append((actor_id, dist))
        
        scored_others.sort(key=lambda x: x[1])
        max_others = self.config.MAX_AGENTS - 1
        selected_others = [aid for aid, _ in scored_others[:max_others]]
        
        return [av_id] + selected_others
    
    def _create_empty_agent_features(self) -> Dict[str, torch.Tensor]:
        """创建空的 Agent 特征"""
        return {
            'agent_history_positions': torch.zeros(0, self.config.HISTORY_STEPS, 2, dtype=torch.float32),
            'agent_history_positions_mask': torch.zeros(0, self.config.HISTORY_STEPS, dtype=torch.bool),
            'agent_history_speed': torch.zeros(0, self.config.HISTORY_STEPS, 2, dtype=torch.float32),
            'agent_future_positions': torch.zeros(0, self.config.FUTURE_STEPS, 2, dtype=torch.float32),
            'agent_future_positions_mask': torch.zeros(0, self.config.FUTURE_STEPS, dtype=torch.bool),
            'agent_type': torch.zeros(0, dtype=torch.long),
            'agent_is_target': torch.zeros(0, dtype=torch.bool),
            'agent_heading': torch.zeros(0, dtype=torch.float32),  # 【新增】
            'num_valid_agents': 0
        }
    
    def pad_to_max_agents(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        将 Agent 特征填充到 MAX_AGENTS
        
        Args:
            features: 原始特征字典
            
        Returns:
            填充后的特征字典
        """
        num_valid = features['num_valid_agents']
        max_agents = self.config.MAX_AGENTS
        
        if num_valid >= max_agents:
            # 截断到 MAX_AGENTS
            for key in features:
                if key != 'num_valid_agents' and isinstance(features[key], torch.Tensor):
                    features[key] = features[key][:max_agents]
        else:
            # 填充到 MAX_AGENTS
            pad_size = max_agents - num_valid
            
            # 填充位置
            features['agent_history_positions'] = torch.cat([
                features['agent_history_positions'],
                torch.zeros(pad_size, self.config.HISTORY_STEPS, 2, dtype=torch.float32)
            ], dim=0)
            
            features['agent_future_positions'] = torch.cat([
                features['agent_future_positions'],
                torch.zeros(pad_size, self.config.FUTURE_STEPS, 2, dtype=torch.float32)
            ], dim=0)
            
            # 填充速度
            features['agent_history_speed'] = torch.cat([
                features['agent_history_speed'],
                torch.zeros(pad_size, self.config.HISTORY_STEPS, 2, dtype=torch.float32)
            ], dim=0)
            
            # 填充 mask（False）
            features['agent_history_positions_mask'] = torch.cat([
                features['agent_history_positions_mask'],
                torch.zeros(pad_size, self.config.HISTORY_STEPS, dtype=torch.bool)
            ], dim=0)
            
            features['agent_future_positions_mask'] = torch.cat([
                features['agent_future_positions_mask'],
                torch.zeros(pad_size, self.config.FUTURE_STEPS, dtype=torch.bool)
            ], dim=0)
            
            # 截断朝向
            if 'agent_heading' in features:
                features['agent_heading'] = features['agent_heading'][:max_agents]
            
            # 填充类型（-1 表示 padding）
            features['agent_type'] = torch.cat([
                features['agent_type'],
                torch.full((pad_size,), -1, dtype=torch.long)
            ], dim=0)
            
            # 填充 target flag（False）
            features['agent_is_target'] = torch.cat([
                features['agent_is_target'],
                torch.zeros(pad_size, dtype=torch.bool)
            ], dim=0)
            
            # 【新增】填充朝向（使用 0.0）
            features['agent_heading'] = torch.cat([
                features['agent_heading'],
                torch.zeros(pad_size, dtype=torch.float32)
            ], dim=0)
        
        # 移除辅助键
        features.pop('num_valid_agents', None)
        
        return features


# ==================== Map 特征提取模块 ====================
class MapFeatureExtractor:
    """Map 特征提取器"""
    
    def __init__(self, config: GlobalConfig, am: ArgoverseMap):
        self.config = config
        self.am = am
    
    def extract_features(self,
                        df: pd.DataFrame,
                        timestamps: List[int],
                        origin: torch.Tensor,
                        rotate_mat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取 Map 的所有特征（使用双核心筛选策略）
        
        Args:
            df: 场景的完整 DataFrame
            timestamps: 时间戳列表
            origin: [2] 原点坐标（全局，AV在t=19的位置）
            rotate_mat: [2, 2] 旋转矩阵
            
        Returns:
            包含所有 Map 特征的字典
        """
        # 获取 t=19 时刻的所有 Agent 位置
        obs_timestamp = timestamps[self.config.OBS_TIMESTEP]
        df_obs = df[df['TIMESTAMP'] == obs_timestamp]
        
        if len(df_obs) == 0:
            return self._create_empty_map_features()
        
        city = df_obs['CITY_NAME'].iloc[0]
        
        # 获取 AV 和 Target Agent 的位置（用于双核心筛选）
        av_position = origin  # AV 已经是 origin
        target_position = None
        
        # 查找 Target Agent 的位置
        target_row = df_obs[df_obs['OBJECT_TYPE'] == 'AGENT']
        if len(target_row) > 0:
            target_row = target_row.iloc[0]
            target_position = torch.tensor([target_row['X'], target_row['Y']], dtype=torch.float32)
        
        # 查询附近的车道
        lane_ids = set()
        for _, row in df_obs.iterrows():
            x, y = row['X'], row['Y']
            nearby_lanes = self.am.get_lane_ids_in_xy_bbox(
                x, y, city, self.config.SEARCH_RADIUS
            )
            lane_ids.update(nearby_lanes)
        
        if len(lane_ids) == 0:
            return self._create_empty_map_features()
        
        # 收集车道信息（全局坐标系下计算几何中心）
        lane_data_list = []
        
        for lane_id in lane_ids:
            try:
                # 获取 centerline（全局坐标）
                centerline = self.am.get_lane_segment_centerline(lane_id, city)
                if centerline is None or len(centerline) == 0:
                    continue
                
                # 转换为 torch tensor（只取 x, y）
                centerline_tensor = torch.from_numpy(centerline[:, :2]).float()
                
                # 计算几何中心（全局坐标系）
                lane_center = centerline_tensor.mean(dim=0)
                
                # 提取语义特征
                is_inter = self.am.lane_is_in_intersection(lane_id, city)
                turn_dir = self.am.get_lane_turn_direction(lane_id, city)
                has_control = self.am.lane_has_traffic_control_measure(lane_id, city)
                
                lane_data_list.append({
                    'lane_id': lane_id,
                    'centerline_global': centerline_tensor,
                    'lane_center': lane_center,
                    'is_intersection': is_inter,
                    'turn_direction': self.config.TURN_DIRECTION_MAP.get(turn_dir, 0),
                    'traffic_control': has_control
                })
                
            except Exception:
                # 单条车道处理失败，跳过
                continue
        
        if len(lane_data_list) == 0:
            return self._create_empty_map_features()
        
        # 应用双核心筛选策略
        filtered_lane_data = self._apply_dual_core_filtering(
            lane_data_list, av_position, target_position
        )
        
        if len(filtered_lane_data) == 0:
            return self._create_empty_map_features()
        
        # 转换到局部坐标系并构建最终 Tensor
        lane_positions_list = []
        lane_masks_list = []
        is_intersections = []
        turn_directions = []
        traffic_controls = []
        
        for lane_data in filtered_lane_data:
            # 转换到局部坐标系（筛选后才转换）
            centerline_local = CoordinateTransform.transform_to_local(
                lane_data['centerline_global'], origin, rotate_mat
            )
            
            # 采样或填充到 MAX_POINTS_PER_LANE
            lane_points, lane_mask = self._process_lane_points(centerline_local)
            
            lane_positions_list.append(lane_points)
            lane_masks_list.append(lane_mask)
            is_intersections.append(lane_data['is_intersection'])
            turn_directions.append(lane_data['turn_direction'])
            traffic_controls.append(lane_data['traffic_control'])
        
        # 堆叠为 Tensor
        map_lane_positions = torch.stack(lane_positions_list, dim=0)  # [L, S, 2]
        map_lane_masks = torch.stack(lane_masks_list, dim=0)  # [L, S]
        map_is_intersection = torch.tensor(is_intersections, dtype=torch.bool)  # [L]
        map_turn_direction = torch.tensor(turn_directions, dtype=torch.long)  # [L]
        map_traffic_control = torch.tensor(traffic_controls, dtype=torch.bool)  # [L]
        
        return {
            'map_lane_positions': map_lane_positions,
            'map_lane_positions_mask': map_lane_masks,
            'map_is_intersection': map_is_intersection,
            'map_turn_direction': map_turn_direction,
            'map_traffic_control': map_traffic_control,
            'num_valid_lanes': len(lane_positions_list)
        }
    
    def _apply_dual_core_filtering(self,
                                   lane_data_list: List[Dict],
                                   av_position: torch.Tensor,
                                   target_position: Optional[torch.Tensor]) -> List[Dict]:
        """
        应用双核心筛选策略：基于到 AV 和 Target 的距离排序
        
        策略：
        1. 计算每条 Lane 的几何中心到 AV 和 Target 的距离
        2. 使用 D_metric = min(d_av, d_target) 作为排序依据
        3. 选取前 MAX_LANES 条车道
        
        Args:
            lane_data_list: 候选 Lane 数据列表
            av_position: AV 位置 [2]（全局坐标）
            target_position: Target Agent 位置 [2]（全局坐标），可能为 None
            
        Returns:
            筛选并排序后的 Lane 数据列表
        """
        scored_lanes = []
        
        for lane_data in lane_data_list:
            lane_center = lane_data['lane_center']
            
            # 计算到 AV 的距离
            dist_to_av = torch.norm(lane_center - av_position).item()
            
            if target_position is not None:
                # 计算到 Target 的距离
                dist_to_target = torch.norm(lane_center - target_position).item()
                
                # D_metric: 取较小距离（更接近核心 Actor 的车道更重要）
                d_metric = min(dist_to_av, dist_to_target)
            else:
                # 降级策略：只有 AV
                d_metric = dist_to_av
            
            scored_lanes.append((lane_data, d_metric))
        
        # 按 D_metric 升序排序
        scored_lanes.sort(key=lambda x: x[1])
        
        # 取前 MAX_LANES
        selected_lanes = [lane_data for lane_data, _ in scored_lanes[:self.config.MAX_LANES]]
        
        return selected_lanes
    
    def _process_lane_points(self, centerline: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理 Lane 的点：采样或填充到 MAX_POINTS_PER_LANE
        
        Args:
            centerline: [N, 2] 局部坐标系下的中心线点
            
        Returns:
            points: [MAX_POINTS_PER_LANE, 2] 处理后的点
            mask: [MAX_POINTS_PER_LANE] 有效性 mask
        """
        num_points = len(centerline)
        max_points = self.config.MAX_POINTS_PER_LANE
        
        points = torch.zeros(max_points, 2, dtype=torch.float32)
        mask = torch.zeros(max_points, dtype=torch.bool)
        
        if num_points <= max_points:
            # 直接填充
            points[:num_points] = centerline
            mask[:num_points] = True
        else:
            # 均匀采样
            indices = torch.linspace(0, num_points - 1, max_points).long()
            points = centerline[indices]
            mask[:] = True
        
        return points, mask
    
    def _create_empty_map_features(self) -> Dict[str, torch.Tensor]:
        """创建空的 Map 特征（用于没有车道的场景）"""
        return {
            'map_lane_positions': torch.zeros(0, self.config.MAX_POINTS_PER_LANE, 2, dtype=torch.float32),
            'map_lane_positions_mask': torch.zeros(0, self.config.MAX_POINTS_PER_LANE, dtype=torch.bool),
            'map_is_intersection': torch.zeros(0, dtype=torch.bool),
            'map_turn_direction': torch.zeros(0, dtype=torch.long),
            'map_traffic_control': torch.zeros(0, dtype=torch.bool),
            'num_valid_lanes': 0
        }
    
    def pad_to_max_lanes(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        将 Map 特征填充到 MAX_LANES
        
        Args:
            features: 原始特征字典
            
        Returns:
            填充后的特征字典
        """
        num_valid = features['num_valid_lanes']
        max_lanes = self.config.MAX_LANES
        
        if num_valid >= max_lanes:
            # 截断到 MAX_LANES
            for key in features:
                if key != 'num_valid_lanes' and isinstance(features[key], torch.Tensor):
                    features[key] = features[key][:max_lanes]
        else:
            # 填充到 MAX_LANES
            pad_size = max_lanes - num_valid
            
            # 填充位置
            features['map_lane_positions'] = torch.cat([
                features['map_lane_positions'],
                torch.zeros(pad_size, self.config.MAX_POINTS_PER_LANE, 2, dtype=torch.float32)
            ], dim=0)
            
            # 填充 mask（False）
            features['map_lane_positions_mask'] = torch.cat([
                features['map_lane_positions_mask'],
                torch.zeros(pad_size, self.config.MAX_POINTS_PER_LANE, dtype=torch.bool)
            ], dim=0)
            
            # 填充语义特征
            features['map_is_intersection'] = torch.cat([
                features['map_is_intersection'],
                torch.zeros(pad_size, dtype=torch.bool)
            ], dim=0)
            
            features['map_turn_direction'] = torch.cat([
                features['map_turn_direction'],
                torch.zeros(pad_size, dtype=torch.long)
            ], dim=0)
            
            features['map_traffic_control'] = torch.cat([
                features['map_traffic_control'],
                torch.zeros(pad_size, dtype=torch.bool)
            ], dim=0)
        
        # 移除辅助键
        features.pop('num_valid_lanes', None)
        
        return features


# ==================== 场景处理主类 ====================
class SceneProcessor:
    """场景处理器"""
    
    def __init__(self, config: GlobalConfig, am: ArgoverseMap):
        self.config = config
        self.am = am
        self.agent_extractor = AgentFeatureExtractor(config, am)
        self.map_extractor = MapFeatureExtractor(config, am)
    
    def process_single_scene(self, csv_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        """
        处理单个场景 - 逻辑重构版
        
        Args:
            csv_path: CSV 文件路径
            
        Returns:
            包含所有特征的字典，失败返回 None
        """
        try:
            df = pd.read_csv(csv_path)
            
            # 数据验证
            required_columns = ['TRACK_ID', 'OBJECT_TYPE', 'TIMESTAMP', 'X', 'Y', 'CITY_NAME']
            if not all(col in df.columns for col in required_columns):
                return None
            
            # 获取时间戳列表
            timestamps = sorted(df['TIMESTAMP'].unique())
            if len(timestamps) < self.config.TOTAL_STEPS:
                return None
            
            # 获取城市名称 (关键：计算 AV 朝向需要用到地图)
            city = df['CITY_NAME'].iloc[0]
            
            av_info = self._get_robust_av_info(df, timestamps, city)
            if av_info is None:
                return None
            
            origin, theta = av_info  # theta 已经是经过地图校准的正确角度
            
            # 使用正确的 theta 构建旋转矩阵
            rotate_mat = CoordinateTransform.compute_rotation_matrix(theta)
            
            # 提取 Agent 特征
            agent_features = self.agent_extractor.extract_features(
                df, timestamps, origin, rotate_mat, theta
            )
            agent_features = self.agent_extractor.pad_to_max_agents(agent_features)
            
            # 提取 Map 特征
            map_features = self.map_extractor.extract_features(
                df, timestamps, origin, rotate_mat
            )
            map_features = self.map_extractor.pad_to_max_lanes(map_features)
            
            # 合并所有特征
            sample = {
                **agent_features,
                **map_features,
                'origin': origin,
                'theta': theta.unsqueeze(0)
            }
            
            return sample
            
        except Exception as e:
            logging.warning(f"处理 {csv_path.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_robust_av_info(self, 
                            df: pd.DataFrame, 
                            timestamps: List[int],
                            city: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取 AV 在 t=19 的位置和【鲁棒】朝向
        包含：运动检测 + 地图回退策略
        
        Args:
            df: 场景 DataFrame
            timestamps: 时间戳列表
            city: 城市名称
            
        Returns:
            (origin, theta) 或 None
        """
        import math
        
        obs_timestamp = timestamps[self.config.OBS_TIMESTEP]
        av_df = df[df['OBJECT_TYPE'] == 'AV']
        
        # 1. 获取当前位置 (Origin)
        av_curr = av_df[av_df['TIMESTAMP'] == obs_timestamp]
        if len(av_curr) == 0:
            return None
        av_curr = av_curr.iloc[0]
        origin = torch.tensor([av_curr['X'], av_curr['Y']], dtype=torch.float32)
        
        # 2. 计算鲁棒朝向 (Robust Theta)
        # 提取历史轨迹 (0~19)
        hist_timestamps = timestamps[:self.config.OBS_TIMESTEP+1]
        av_hist = av_df[av_df['TIMESTAMP'].isin(hist_timestamps)]
        
        valid_pos_list = []
        for ts in hist_timestamps:
            row = av_hist[av_hist['TIMESTAMP'] == ts]
            if len(row) > 0:
                valid_pos_list.append([row.iloc[0]['X'], row.iloc[0]['Y']])
        
        if len(valid_pos_list) < 2:
            return None  # 只有一帧数据，无法计算方向
            
        valid_pos = np.array(valid_pos_list)  # [T, 2]
        
        # --- 策略 A: 运动检测 ---
        EPSILON = 0.1
        # 计算最大位移
        displacement = np.linalg.norm(valid_pos.max(axis=0) - valid_pos.min(axis=0))
        
        theta = 0.0
        
        if displacement > EPSILON:
            # 只要动过，就用几何差分
            # 倒序寻找最近的有效移动向量
            found_moving = False
            for i in range(len(valid_pos)-1, 0, -1):
                p_curr = valid_pos[i]
                p_prev = valid_pos[i-1]
                dist = np.linalg.norm(p_curr - p_prev)
                if dist > 1e-2:  # 1cm 的微动
                    theta = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
                    found_moving = True
                    break
            
            if not found_moving:
                # 极其罕见的情况：累计位移大，但帧间位移都极小（漂移？）
                # 这种情况下使用首尾差分
                delta = valid_pos[-1] - valid_pos[0]
                theta = math.atan2(delta[1], delta[0])

        else:
            # --- 策略 B: 地图回退 (Map Fallback) ---
            # AV 完全静止，查询车道方向
            try:
                # 查询 Search Radius 内的车道
                lane_ids = self.am.get_lane_ids_in_xy_bbox(
                    origin[0].item(), origin[1].item(), city, query_search_range_manhattan=5.0
                )
                
                if len(lane_ids) > 0:
                    # 获取最近车道
                    centerline = self.am.get_lane_segment_centerline(lane_ids[0], city)
                    
                    if centerline is not None and len(centerline) >= 2:
                        # 找到最近点索引
                        dists = np.linalg.norm(centerline[:, :2] - valid_pos[-1], axis=1)
                        nearest_idx = np.argmin(dists)
                        
                        # 计算切线向量 (保证顺着索引方向)
                        if nearest_idx < len(centerline) - 1:
                            v = centerline[nearest_idx + 1, :2] - centerline[nearest_idx, :2]
                        else:
                            v = centerline[nearest_idx, :2] - centerline[nearest_idx - 1, :2]
                        
                        theta = math.atan2(v[1], v[0])
                    else:
                        theta = 0.0
                else:
                    # Off-road 且静止，给个默认值 0
                    theta = 0.0
                    
            except Exception as e:
                logging.debug(f"Map query failed for AV heading: {e}")
                theta = 0.0

        return origin, torch.tensor(theta, dtype=torch.float32)
    
    def save_to_pt(self, sample: Dict[str, torch.Tensor], output_path: Path) -> None:
        """保存为 .pt 文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample, output_path)


# ==================== 主函数 ====================
def setup_logging(log_file: Optional[str] = None):
    """设置日志"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description='Argoverse 1.1 离线预处理脚本 (Dense Tensor 版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 快速测试 10 个样本（使用默认路径）
  python preprocess_offline.py --sample 10
  
  # 处理训练集（使用默认路径）
  python preprocess_offline.py
  
  # 处理验证集
  python preprocess_offline.py --data_dir /root/vc/data/val/data --output_dir /root/vc/data/val/processed_dense
  
  # 自定义参数 + 采样测试
  python preprocess_offline.py --data_dir /path/to/data --output_dir /path/to/output --sample 5 --radius 150.0
        """
    )
    
    parser.add_argument('--data_dir', type=str, 
                        default='/root/vc/data/train/data',
                        help='CSV 文件所在目录 (默认: /root/vc/data/train/data)')
    parser.add_argument('--output_dir', type=str, 
                        default='/root/vc/data/train/processed_dense',
                        help='输出 .pt 文件的目录 (默认: /root/vc/data/train/processed_dense)')
    parser.add_argument('--map_dir', type=str, default=None,
                        help='地图文件目录 (可选，默认使用 Argoverse 环境变量)')
    parser.add_argument('--radius', type=float, default=100.0,
                        help='地图查询半径(米) (默认: 100.0)')
    parser.add_argument('--max_agents', type=int, default=64,
                        help='最大 Agent 数量 (默认: 64)')
    parser.add_argument('--max_lanes', type=int, default=256,
                        help='最大 Lane 数量 (默认: 256)')
    parser.add_argument('--max_points', type=int, default=10,
                        help='每条 Lane 的最大点数 (默认: 10)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='日志文件路径 (可选)')
    parser.add_argument('--sample', type=int, default=None,
                        help='随机抽取 N 个样本进行测试 (可选，用于快速验证脚本)')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_file)
    
    logging.info("=" * 80)
    logging.info("Argoverse 1.1 离线预处理 (Dense Tensor 版本)".center(80))
    logging.info("=" * 80)
    logging.info("")
    logging.info("配置:")
    logging.info(f"  数据目录: {args.data_dir}")
    logging.info(f"  输出目录: {args.output_dir}")
    logging.info(f"  查询半径: {args.radius}m")
    logging.info(f"  MAX_AGENTS: {args.max_agents}")
    logging.info(f"  MAX_LANES: {args.max_lanes}")
    logging.info(f"  MAX_POINTS_PER_LANE: {args.max_points}")
    logging.info("")
    
    # 更新配置
    config = GlobalConfig()
    config.MAX_AGENTS = args.max_agents
    config.MAX_LANES = args.max_lanes
    config.MAX_POINTS_PER_LANE = args.max_points
    config.SEARCH_RADIUS = args.radius
    
    # 初始化 Map API
    logging.info("初始化 Argoverse Map API...")
    if args.map_dir:
        os.environ['ARGOVERSE_DATA_DIR'] = str(Path(args.map_dir).parent)
    
    try:
        am = ArgoverseMap()
        logging.info("✓ Map API 初始化成功")
    except Exception as e:
        logging.error(f"✗ Map API 初始化失败: {e}")
        sys.exit(1)
    
    # 获取所有 CSV 文件
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logging.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)
    
    csv_files = sorted(data_dir.glob("*.csv"))
    if len(csv_files) == 0:
        logging.error(f"未找到 CSV 文件: {data_dir}")
        sys.exit(1)
    
    total_files = len(csv_files)
    logging.info(f"找到 {total_files} 个 CSV 文件")
    
    # 如果指定了采样数量，随机抽取样本
    if args.sample:
        sample_size = min(args.sample, total_files)
        csv_files = random.sample(csv_files, sample_size)
        logging.info(f"✓ 采样模式: 随机抽取 {sample_size} 个样本进行测试")
    
    logging.info("")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建处理器
    processor = SceneProcessor(config, am)
    
    # 处理所有场景
    logging.info("开始处理场景...")
    success_count = 0
    failed_count = 0
    
    for csv_path in tqdm(csv_files, desc="处理进度", unit="场景"):
        seq_id = csv_path.stem
        output_path = output_dir / f"{seq_id}.pt"
        
        # 如果已存在，跳过
        if output_path.exists():
            success_count += 1
            continue
        
        # 处理场景
        sample = processor.process_single_scene(csv_path)
        
        if sample is not None:
            processor.save_to_pt(sample, output_path)
            success_count += 1
        else:
            failed_count += 1
    
    # 打印总结
    logging.info("")
    logging.info("=" * 80)
    logging.info("处理完成!".center(80))
    logging.info("=" * 80)
    logging.info(f"  成功: {success_count} 个场景")
    logging.info(f"  失败: {failed_count} 个场景")
    logging.info(f"  输出目录: {output_dir}")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()