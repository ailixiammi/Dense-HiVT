#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Jitter Diagnostic Tool
诊断工具：对比原始 CSV 数据与预处理后的 Tensor 数据，判断预处理是否引入噪音
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


# ==================== 配置 ====================
class DiagnosticConfig:
    """诊断工具配置"""
    # 数据路径（默认值）
    RAW_DATA_DIR = Path("/root/vc/data/train/data")
    PROCESSED_DATA_DIR = Path("/root/vc/data/train/processed_dense")
    
    # Agent 类型映射
    AGENT_TYPE_MAP = {
        'AV': 0,
        'AGENT': 1
    }
    
    # 误差阈值（米）
    # 如果最大误差超过此值，认为预处理引入了噪音
    MAX_ERROR_THRESHOLD = 0.01  # 1cm
    
    # 时间参数
    HISTORY_STEPS = 20
    FUTURE_STEPS = 30
    TOTAL_STEPS = 50
    DT = 0.1  # 时间间隔（秒）


# ==================== 数据加载 ====================
def load_raw_csv(csv_path: Path, agent_type: str) -> Optional[pd.DataFrame]:
    """加载原始 CSV 数据并筛选指定类型的 Agent"""
    try:
        df = pd.read_csv(csv_path)
        
        # 验证必需列
        required_cols = ['TRACK_ID', 'OBJECT_TYPE', 'TIMESTAMP', 'X', 'Y']
        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] CSV 文件缺少必需列: {required_cols}")
            return None
        
        # 筛选指定类型的 Agent
        agent_df = df[df['OBJECT_TYPE'] == agent_type].copy()
        
        if len(agent_df) == 0:
            print(f"[ERROR] 未找到类型为 '{agent_type}' 的 Agent")
            return None
        
        # 按时间戳排序
        agent_df = agent_df.sort_values('TIMESTAMP').reset_index(drop=True)
        
        return agent_df
        
    except Exception as e:
        print(f"[ERROR] 加载 CSV 失败: {e}")
        return None


def load_processed_pt(pt_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    """加载预处理后的 .pt 数据"""
    try:
        data = torch.load(pt_path, map_location='cpu')
        
        # 验证必需键
        required_keys = [
            'origin', 'theta',
            'agent_history_positions', 'agent_history_positions_mask',
            'agent_future_positions', 'agent_future_positions_mask'
        ]
        
        if not all(key in data for key in required_keys):
            print(f"[ERROR] .pt 文件缺少必需键: {required_keys}")
            return None
        
        return data
        
    except Exception as e:
        print(f"[ERROR] 加载 .pt 文件失败: {e}")
        return None


# ==================== 坐标转换 ====================
def inverse_transform(local_positions: torch.Tensor,
                     origin: torch.Tensor,
                     theta: torch.Tensor) -> torch.Tensor:
    """
    将局部坐标逆变换回全局坐标
    
    逆变换公式: P_global = P_local @ R.T + P_origin
    
    Args:
        local_positions: [..., 2] 局部坐标
        origin: [2] 原点坐标
        theta: 标量，旋转角度（弧度）
        
    Returns:
        [..., 2] 全局坐标
    """
    # 构建旋转矩阵
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    R = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], dtype=torch.float32)
    
    # 逆旋转：P_local @ R.T
    rotated = torch.matmul(local_positions, R.T)
    
    # 逆平移：+ origin
    global_positions = rotated + origin
    
    return global_positions


# ==================== 主要诊断逻辑 ====================
def diagnose(sample_id: str,
            agent_type: str,
            raw_dir: Path,
            processed_dir: Path,
            config: DiagnosticConfig) -> None:
    """
    诊断预处理是否引入噪音
    
    Args:
        sample_id: 样本ID
        agent_type: Agent类型 ('AV' 或 'AGENT')
        raw_dir: 原始数据目录
        processed_dir: 处理后数据目录
        config: 配置对象
    """
    # 文件路径
    csv_path = raw_dir / f"{sample_id}.csv"
    pt_path = processed_dir / f"{sample_id}.pt"
    
    # 验证文件存在
    if not csv_path.exists():
        print(f"[ERROR] CSV 文件不存在: {csv_path}")
        sys.exit(1)
    
    if not pt_path.exists():
        print(f"[ERROR] .pt 文件不存在: {pt_path}")
        sys.exit(1)
    
    # 加载数据
    print(f"\n{'='*80}")
    print(f"轨迹一致性诊断".center(80))
    print(f"{'='*80}\n")
    print(f"样本ID: {sample_id}")
    print(f"Agent类型: {agent_type}")
    print(f"\n加载原始数据: {csv_path}")
    
    raw_df = load_raw_csv(csv_path, agent_type)
    if raw_df is None:
        sys.exit(1)
    
    print(f"加载处理数据: {pt_path}")
    processed_data = load_processed_pt(pt_path)
    if processed_data is None:
        sys.exit(1)
    
    # 提取原始轨迹
    raw_positions = raw_df[['X', 'Y']].values.astype(np.float32)
    
    # 提取处理后的轨迹并逆变换到全局坐标
    agent_idx = config.AGENT_TYPE_MAP[agent_type]
    
    history_local = processed_data['agent_history_positions'][agent_idx]  # [20, 2]
    future_local = processed_data['agent_future_positions'][agent_idx]  # [30, 2]
    local_positions = torch.cat([history_local, future_local], dim=0)  # [50, 2]
    
    history_mask = processed_data['agent_history_positions_mask'][agent_idx]  # [20]
    future_mask = processed_data['agent_future_positions_mask'][agent_idx]  # [30]
    masks = torch.cat([history_mask, future_mask], dim=0)  # [50]
    
    origin = processed_data['origin']  # [2]
    theta = processed_data['theta'].squeeze()  # scalar
    
    restored_positions = inverse_transform(local_positions, origin, theta)
    
    # 计算误差
    print(f"\n计算坐标误差...")
    errors = np.zeros(len(raw_positions))
    
    for i in range(len(raw_positions)):
        if masks[i]:
            raw_pos = raw_positions[i]
            restored_pos = restored_positions[i].numpy()
            errors[i] = np.linalg.norm(raw_pos - restored_pos)
        else:
            errors[i] = np.nan
    
    # 过滤有效误差
    valid_errors = errors[~np.isnan(errors)]
    
    if len(valid_errors) == 0:
        print(f"[ERROR] 没有有效的数据帧可以对比")
        sys.exit(1)
    
    # 统计信息
    max_error = valid_errors.max()
    mean_error = valid_errors.mean()
    median_error = np.median(valid_errors)
    
    print(f"\n{'='*80}")
    print(f"诊断结果".center(80))
    print(f"{'='*80}\n")
    
    print(f"有效帧数: {len(valid_errors)} / {len(errors)}")
    print(f"\n坐标误差统计:")
    print(f"  最大值: {max_error:.6f} 米")
    print(f"  平均值: {mean_error:.6f} 米")
    print(f"  中位数: {median_error:.6f} 米")
    
    # 显示前5帧和后5帧的详细对比
    print(f"\n详细对比 (显示前5帧和后5帧):\n")
    print(f"{'帧':<6} {'原始X':<12} {'原始Y':<12} {'恢复X':<12} {'恢复Y':<12} {'误差(m)':<10}")
    print(f"{'-'*70}")
    
    total = len(raw_positions)
    rows_to_show = list(range(min(5, total))) + list(range(max(0, total - 5), total))
    
    prev_idx = -1
    for i in rows_to_show:
        if i - prev_idx > 1:
            print("  ...")
        
        if masks[i]:
            print(f"{i:<6} {raw_positions[i, 0]:<12.4f} {raw_positions[i, 1]:<12.4f} "
                  f"{restored_positions[i, 0].item():<12.4f} {restored_positions[i, 1].item():<12.4f} "
                  f"{errors[i]:<10.6f}")
        else:
            print(f"{i:<6} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
        
        prev_idx = i
    
    # 最终结论
    print(f"\n{'='*80}")
    print(f"结论".center(80))
    print(f"{'='*80}\n")
    
    if max_error < config.MAX_ERROR_THRESHOLD:
        print(f"✅ 预处理未引入明显噪音")
        print(f"   最大误差 {max_error:.6f}m < 阈值 {config.MAX_ERROR_THRESHOLD}m")
        print(f"\n   说明: 预处理过程（坐标变换、旋转、平移）是可逆的，")
        print(f"         处理后的数据与原始数据高度一致。")
    else:
        print(f"❌ 预处理引入了噪音")
        print(f"   最大误差 {max_error:.6f}m >= 阈值 {config.MAX_ERROR_THRESHOLD}m")
        print(f"\n   说明: 预处理过程引入了不可接受的误差，")
        print(f"         需要检查 preprocess_offline.py 中的坐标变换逻辑。")
    
    print(f"\n{'='*80}\n")


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(
        description='轨迹一致性诊断：判断 preprocess_offline.py 是否引入噪音',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 诊断 AV 轨迹
  python scripts/diagnose_jitter.py --sample_id 123456 --agent_type AV
  
  # 诊断 Target Agent 轨迹
  python scripts/diagnose_jitter.py --sample_id 123456 --agent_type AGENT
  
  # 自定义数据路径
  python scripts/diagnose_jitter.py --sample_id 123456 --agent_type AV \\
      --raw_dir /path/to/raw \\
      --processed_dir /path/to/processed
        """
    )
    
    parser.add_argument('--sample_id', type=str, required=True,
                        help='样本ID（文件名，不含扩展名）')
    parser.add_argument('--agent_type', type=str, required=True,
                        choices=['AV', 'AGENT'],
                        help='Agent 类型（AV 或 AGENT）')
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='原始 CSV 数据目录（默认: /root/vc/data/train/data）')
    parser.add_argument('--processed_dir', type=str, default=None,
                        help='处理后数据目录（默认: /root/vc/data/train/processed_dense）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='最大误差阈值（米，默认: 0.01）')
    
    args = parser.parse_args()
    
    # 配置
    config = DiagnosticConfig()
    
    # 自定义阈值
    if args.threshold is not None:
        config.MAX_ERROR_THRESHOLD = args.threshold
    
    # 设置数据路径
    raw_dir = Path(args.raw_dir) if args.raw_dir else config.RAW_DATA_DIR
    processed_dir = Path(args.processed_dir) if args.processed_dir else config.PROCESSED_DATA_DIR
    
    # 执行诊断
    diagnose(args.sample_id, args.agent_type, raw_dir, processed_dir, config)


if __name__ == '__main__':
    main()