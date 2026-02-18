#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Point-Based Data Viewer for Argoverse 1.1 Processed Data
使用 Streamlit + Plotly 可视化预处理后的 .pt 文件
"""

import streamlit as st
import torch
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


# ==================== 配置 ====================
PROCESSED_DATA_DIR = Path("/root/vc/data/train/processed_dense/")  # 默认数据目录
HISTORY_STEPS = 20
FUTURE_STEPS = 30

# 颜色配置
COLORS = {
    'av': 'red',
    'target': 'blue',
    'others': 'green',
    'map': 'lightgrey'
}


# ==================== 数据加载 ====================
@st.cache(allow_output_mutation=True)
def load_pt_file(file_path: str) -> Dict[str, torch.Tensor]:
    """
    加载 .pt 文件（带缓存）
    
    Args:
        file_path: 文件路径
        
    Returns:
        数据字典
    """
    data = torch.load(file_path, map_location='cpu')
    return data


@st.cache(allow_output_mutation=True)
def get_available_files(data_dir: str) -> List[str]:
    """
    获取可用的 .pt 文件列表
    
    Args:
        data_dir: 数据目录
        
    Returns:
        文件名列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    pt_files = sorted([f.name for f in data_path.glob("*.pt")])
    return pt_files


# ==================== 数据准备 ====================
def prepare_agent_points(data: Dict[str, torch.Tensor],
                        show_av: bool,
                        show_target: bool,
                        show_others: bool,
                        show_history: bool,
                        show_future: bool) -> List[Dict]:
    """
    准备 Agent 点数据（根据过滤条件）
    
    Args:
        data: 数据字典
        show_av: 是否显示 AV
        show_target: 是否显示 Target
        show_others: 是否显示 Others
        show_history: 是否显示历史点
        show_future: 是否显示未来点
        
    Returns:
        点数据列表，每个元素包含 {x, y, agent_id, time_step, agent_type, is_future}
    """
    # 提取数据
    history_pos = data['agent_history_positions']  # [N, 20, 2]
    history_mask = data['agent_history_positions_mask']  # [N, 20]
    future_pos = data['agent_future_positions']  # [N, 30, 2]
    future_mask = data['agent_future_positions_mask']  # [N, 30]
    agent_type = data['agent_type']  # [N]
    
    N = history_pos.shape[0]
    
    points = []
    
    for agent_id in range(N):
        # 获取 Agent 类型
        atype = agent_type[agent_id].item()
        
        # 跳过 padding agent
        if atype == -1:
            continue
        
        # 判断是否显示该 Agent
        if atype == 0 and not show_av:
            continue
        if atype == 1 and not show_target:
            continue
        if atype == 2 and not show_others:
            continue
        
        # 处理历史点
        if show_history:
            for t in range(HISTORY_STEPS):
                if history_mask[agent_id, t]:
                    x, y = history_pos[agent_id, t].tolist()
                    points.append({
                        'x': x,
                        'y': y,
                        'agent_id': agent_id,
                        'time_step': t,
                        'agent_type': atype,
                        'is_future': False
                    })
        
        # 处理未来点
        if show_future:
            for t in range(FUTURE_STEPS):
                if future_mask[agent_id, t]:
                    x, y = future_pos[agent_id, t].tolist()
                    points.append({
                        'x': x,
                        'y': y,
                        'agent_id': agent_id,
                        'time_step': t + HISTORY_STEPS,
                        'agent_type': atype,
                        'is_future': True
                    })
    
    return points


def prepare_map_lanes(data: Dict[str, torch.Tensor]) -> List[Dict]:
    """
    准备 Map Lane 数据
    
    Args:
        data: 数据字典
        
    Returns:
        Lane 数据列表，每个元素包含 {x_coords, y_coords}
    """
    lane_pos = data['map_lane_positions']  # [L, S, 2]
    lane_mask = data['map_lane_positions_mask']  # [L, S]
    
    L = lane_pos.shape[0]
    
    lanes = []
    
    for lane_id in range(L):
        # 提取有效点
        valid_indices = lane_mask[lane_id].nonzero(as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            continue
        
        x_coords = lane_pos[lane_id, valid_indices, 0].tolist()
        y_coords = lane_pos[lane_id, valid_indices, 1].tolist()
        
        lanes.append({
            'x_coords': x_coords,
            'y_coords': y_coords
        })
    
    return lanes


# ==================== 可视化 ====================
def create_plot(agent_points: List[Dict],
                map_lanes: List[Dict],
                show_map: bool) -> go.Figure:
    """
    创建 Plotly 交互式图表
    
    Args:
        agent_points: Agent 点数据
        map_lanes: Map Lane 数据
        show_map: 是否显示地图
        
    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()
    
    # 添加 Map Lanes（如果启用）
    if show_map:
        for lane in map_lanes:
            fig.add_trace(go.Scatter(
                x=lane['x_coords'],
                y=lane['y_coords'],
                mode='lines',
                line=dict(color=COLORS['map'], width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 按类型和时间分组绘制 Agent 点
    agent_groups = {
        (0, False): {'name': 'AV (History)', 'color': COLORS['av'], 'opacity': 1.0, 'symbol': 'circle'},
        (0, True): {'name': 'AV (Future)', 'color': COLORS['av'], 'opacity': 0.6, 'symbol': 'x'},
        (1, False): {'name': 'Target (History)', 'color': COLORS['target'], 'opacity': 1.0, 'symbol': 'circle'},
        (1, True): {'name': 'Target (Future)', 'color': COLORS['target'], 'opacity': 0.6, 'symbol': 'x'},
        (2, False): {'name': 'Others (History)', 'color': COLORS['others'], 'opacity': 1.0, 'symbol': 'circle'},
        (2, True): {'name': 'Others (Future)', 'color': COLORS['others'], 'opacity': 0.6, 'symbol': 'x'},
    }
    
    # 分组点
    grouped_points = {key: [] for key in agent_groups.keys()}
    
    for point in agent_points:
        key = (point['agent_type'], point['is_future'])
        grouped_points[key].append(point)
    
    # 绘制每组点
    for key, group_info in agent_groups.items():
        points = grouped_points[key]
        
        if len(points) == 0:
            continue
        
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]
        hover_texts = [f"Agent {p['agent_id']}, t={p['time_step']}" for p in points]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=group_info['color'],
                opacity=group_info['opacity'],
                symbol=group_info['symbol']
            ),
            name=group_info['name'],
            text=hover_texts,
            hovertemplate='%{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
    
    # 设置布局（1:1 比例）
    fig.update_layout(
        title="Argoverse Data Viewer (Point-Based)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        width=800,
        height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1),  # 1:1 比例
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


# ==================== 主程序 ====================
def main():
    st.set_page_config(
        page_title="Argoverse Data Viewer",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Interactive Argoverse Data Viewer")
    st.markdown("**Point-Based Visualization** for Processed `.pt` Files")
    
    # 侧边栏控制
    st.sidebar.header("📊 Controls")
    
    # 1. 文件选择器
    st.sidebar.subheader("1️⃣ File Selector")
    
    # 数据目录输入
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=str(PROCESSED_DATA_DIR),
        help="Path to the directory containing .pt files"
    )
    
    available_files = get_available_files(data_dir)
    
    if len(available_files) == 0:
        st.error(f"❌ No .pt files found in `{data_dir}`")
        st.info("Please check the data directory path.")
        return
    
    selected_file = st.sidebar.selectbox(
        "Select a .pt file",
        available_files,
        index=0
    )
    
    # 加载数据
    file_path = Path(data_dir) / selected_file
    
    try:
        data = load_pt_file(str(file_path))
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return
    
    # 2. Agent 过滤器
    st.sidebar.subheader("2️⃣ Agent Filter")
    show_av = st.sidebar.checkbox("Show AV (Self-driving car)", value=True)
    show_target = st.sidebar.checkbox("Show Target Agent", value=True)
    show_others = st.sidebar.checkbox("Show Others (Context)", value=True)
    
    # 3. 时间过滤器
    st.sidebar.subheader("3️⃣ Temporal Filter")
    show_history = st.sidebar.checkbox("Show History Points (t ≤ 19)", value=True)
    show_future = st.sidebar.checkbox("Show Future Points (t > 19)", value=True)
    
    # 4. Map Toggle
    st.sidebar.subheader("4️⃣ Map Toggle")
    show_map = st.sidebar.checkbox("Show Lane Centerlines", value=True)
    
    # 准备数据
    agent_points = prepare_agent_points(
        data, show_av, show_target, show_others, show_history, show_future
    )
    
    map_lanes = prepare_map_lanes(data)
    
    # 创建图表
    fig = create_plot(agent_points, map_lanes, show_map)
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示文件信息
    st.markdown("---")
    st.subheader("📄 File Info")
    
    origin = data.get('origin', torch.tensor([0.0, 0.0]))
    origin_x, origin_y = origin.tolist()
    
    # 尝试从文件名推断城市（假设文件名包含城市信息）
    city = "Unknown"
    if 'PIT' in selected_file.upper():
        city = "Pittsburgh (PIT)"
    elif 'MIA' in selected_file.upper():
        city = "Miami (MIA)"
    
    st.markdown(f"""
    - **Filename**: `{selected_file}`
    - **City**: {city}
    - **Origin**: ({origin_x:.2f}, {origin_y:.2f})
    - **Total Agents**: {(data['agent_type'] != -1).sum().item()}
    - **Total Lanes**: {(data['map_lane_positions_mask'].any(dim=1)).sum().item()}
    """)
    
    # 额外信息
    with st.expander("🔍 Advanced Info"):
        st.write("**Agent Type Distribution:**")
        agent_type = data['agent_type']
        num_av = (agent_type == 0).sum().item()
        num_target = (agent_type == 1).sum().item()
        num_others = (agent_type == 2).sum().item()
        
        st.write(f"- AV: {num_av}")
        st.write(f"- Target: {num_target}")
        st.write(f"- Others: {num_others}")
        
        st.write("**Data Shapes:**")
        st.write(f"- agent_history_positions: {list(data['agent_history_positions'].shape)}")
        st.write(f"- agent_future_positions: {list(data['agent_future_positions'].shape)}")
        st.write(f"- map_lane_positions: {list(data['map_lane_positions'].shape)}")


if __name__ == '__main__':
    main()