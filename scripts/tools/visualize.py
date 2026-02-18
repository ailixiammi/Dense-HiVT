#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证与可视化脚本
读取预处理后的 .pt 文件，生成可视化报告 PDF
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches


class SceneVisualizer:
    """场景可视化器"""
    
    def __init__(self, coord_range: float = 120.0):
        """
        Args:
            coord_range: 坐标范围（米），默认 [-120, 120]
        """
        self.coord_range = coord_range
        
    def plot_scene(self, sample: Dict[str, torch.Tensor], seq_id: str) -> plt.Figure:
        """
        绘制单个场景
        
        Args:
            sample: 样本数据字典
            seq_id: 场景 ID
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 设置坐标范围和纵横比
        ax.set_xlim(-self.coord_range, self.coord_range)
        ax.set_ylim(-self.coord_range, self.coord_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 绘制地图车道
        self._draw_map_lanes(ax, sample)
        
        # 绘制 Agent 轨迹
        self._draw_agents(ax, sample)
        
        # 绘制原点参考
        self._draw_reference(ax)
        
        # 设置标题和标签
        origin = sample['origin'].numpy()
        theta_deg = float(sample['theta'].item() * 180 / np.pi)
        
        # 统计信息
        num_history = sample['agent_history_positions_mask'].any(dim=1).sum().item()
        num_future = sample['agent_future_positions_mask'].any(dim=1).sum().item()
        num_obs = sample['agent_history_positions_mask'][:, -1].sum().item()
        num_lanes = sample['map_lane_positions_mask'][:, 0].sum().item()
        
        title = (
            f"Scene: {seq_id}\n"
            f"Origin: ({origin[0]:.2f}, {origin[1]:.2f}) | "
            f"Theta: {theta_deg:.2f}°\n"
            f"Agents: History={num_history}, Future={num_future}, Obs={num_obs} | "
            f"Lanes: {num_lanes}"
        )
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel('X (meters)', fontsize=11)
        ax.set_ylabel('Y (meters)', fontsize=11)
        
        # 添加图例
        self._add_legend(ax)
        
        plt.tight_layout()
        return fig
    
    def _draw_map_lanes(self, ax: plt.Axes, sample: Dict[str, torch.Tensor]) -> None:
        """绘制地图车道"""
        lane_positions = sample['map_lane_positions']  # [L, S, 2]
        lane_mask = sample['map_lane_positions_mask']  # [L, S]
        is_intersection = sample['map_is_intersection']  # [L]
        
        num_lanes = lane_mask[:, 0].sum().item()
        
        for i in range(int(num_lanes)):
            # 获取有效点
            valid_mask = lane_mask[i].numpy()
            points = lane_positions[i][valid_mask].numpy()
            
            if len(points) < 2:
                continue
            
            # 根据是否为路口选择颜色
            if is_intersection[i].item():
                color = '#FFB6C1'  # 淡红色（路口）
                linewidth = 2.0
                alpha = 0.7
            else:
                color = '#808080'  # 灰色（普通车道）
                linewidth = 1.5
                alpha = 0.5
            
            # 绘制车道中心线
            ax.plot(points[:, 0], points[:, 1], 
                   color=color, linewidth=linewidth, alpha=alpha, zorder=1)
    
    def _draw_agents(self, ax: plt.Axes, sample: Dict[str, torch.Tensor]) -> None:
        """绘制 Agent 轨迹"""
        history_positions = sample['agent_history_positions']  # [N, 20, 2]
        history_mask = sample['agent_history_positions_mask']  # [N, 20]
        future_positions = sample['agent_future_positions']  # [N, 30, 2]
        future_mask = sample['agent_future_positions_mask']  # [N, 30]
        agent_type = sample['agent_type']  # [N]
        is_target = sample['agent_is_target']  # [N]
        
        num_agents = history_mask.any(dim=1).sum().item()
        
        for i in range(int(num_agents)):
            # 确定颜色
            if agent_type[i].item() == 0:  # AV
                color = '#2ECC71'  # 绿色
                linewidth = 2.5
                label = 'AV'
            elif is_target[i].item():  # 目标 Agent
                color = '#3498DB'  # 蓝色
                linewidth = 2.0
                label = 'Target'
            else:  # 其他 Agent
                color = '#95A5A6'  # 灰色
                linewidth = 1.5
                label = 'Others'
            
            # 绘制历史轨迹（实线）
            history_valid = history_mask[i].numpy()
            history_points = history_positions[i][history_valid].numpy()
            
            if len(history_points) >= 2:
                ax.plot(history_points[:, 0], history_points[:, 1],
                       color=color, linewidth=linewidth, linestyle='-',
                       alpha=0.8, zorder=3)
            
            # 绘制未来轨迹（虚线）
            future_valid = future_mask[i].numpy()
            future_points = future_positions[i][future_valid].numpy()
            
            if len(future_points) >= 2:
                ax.plot(future_points[:, 0], future_points[:, 1],
                       color=color, linewidth=linewidth, linestyle='--',
                       alpha=0.6, zorder=2)
            
            # 在 t=19 位置标记箭头（当前位置和朝向）
            if history_mask[i, 19].item():  # t=19 可见
                curr_pos = history_positions[i, 19].numpy()
                
                # 计算朝向（从 t=18 到 t=19）
                if history_mask[i, 18].item():
                    prev_pos = history_positions[i, 18].numpy()
                    direction = curr_pos - prev_pos
                    
                    # 归一化方向
                    norm = np.linalg.norm(direction)
                    if norm > 0.01:
                        direction = direction / norm * 3.0  # 箭头长度 3 米
                        
                        # 绘制箭头
                        arrow = FancyArrowPatch(
                            curr_pos, curr_pos + direction,
                            arrowstyle='->', mutation_scale=20,
                            color=color, linewidth=2, zorder=5
                        )
                        ax.add_patch(arrow)
                
                # 在当前位置画圆点
                ax.scatter(curr_pos[0], curr_pos[1], 
                          s=100, c=color, marker='o', 
                          edgecolors='white', linewidths=1.5, zorder=6)
    
    def _draw_reference(self, ax: plt.Axes) -> None:
        """在原点绘制红色十字参考标记"""
        cross_size = 5.0  # 十字大小（米）
        
        # 绘制十字
        ax.plot([-cross_size, cross_size], [0, 0], 
               color='red', linewidth=2.5, zorder=10, label='Origin')
        ax.plot([0, 0], [-cross_size, cross_size], 
               color='red', linewidth=2.5, zorder=10)
        
        # 在原点画圆
        ax.scatter(0, 0, s=150, c='red', marker='x', 
                  linewidths=3, zorder=11)
    
    def _add_legend(self, ax: plt.Axes) -> None:
        """添加图例"""
        legend_elements = [
            mpatches.Patch(color='#2ECC71', label='AV'),
            mpatches.Patch(color='#3498DB', label='Target Agent'),
            mpatches.Patch(color='#95A5A6', label='Other Agents'),
            mpatches.Patch(color='#808080', label='Lane', alpha=0.5),
            mpatches.Patch(color='#FFB6C1', label='Intersection', alpha=0.7),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-', 
                      label='History'),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', 
                      label='Future'),
            mpatches.Patch(color='red', label='Origin (0,0)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=9, framealpha=0.9)


def load_samples(data_dir: Path, num_samples: int = 20) -> List[Tuple[str, Dict]]:
    """
    加载样本数据
    
    Args:
        data_dir: 数据目录
        num_samples: 加载样本数量
        
    Returns:
        (seq_id, sample) 列表
    """
    pt_files = sorted(data_dir.glob("*.pt"))[:num_samples]
    
    if len(pt_files) == 0:
        raise FileNotFoundError(f"未找到 .pt 文件: {data_dir}")
    
    samples = []
    for pt_file in pt_files:
        try:
            sample = torch.load(pt_file)
            seq_id = pt_file.stem
            samples.append((seq_id, sample))
        except Exception as e:
            print(f"[WARNING] 加载 {pt_file.name} 失败: {e}")
            continue
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='数据验证与可视化脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认路径
  python scripts/verify_data.py
  
  # 自定义参数
  python scripts/verify_data.py \
      --data_dir ~/vc/data/train/processed_dense \
      --output my_report.pdf \
      --num_samples 10
        """
    )
    
    parser.add_argument('--data_dir', type=str,
                       default='/root/vc/data/train/processed_dense',
                       help='数据目录 (默认: /root/vc/data/train/processed_dense)')
    parser.add_argument('--output', type=str,
                       default='visualization_report.pdf',
                       help='输出 PDF 文件名 (默认: visualization_report.pdf)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='处理样本数量 (默认: 20)')
    parser.add_argument('--coord_range', type=float, default=120.0,
                       help='坐标范围（米） (默认: 120.0)')
    
    args = parser.parse_args()
    
    # 展开路径
    data_dir = Path(args.data_dir).expanduser()
    output_path = Path(args.output)
    
    print("=" * 80)
    print("数据验证与可视化".center(80))
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"输出文件: {output_path}")
    print(f"样本数量: {args.num_samples}")
    print(f"坐标范围: ±{args.coord_range}m")
    print("")
    
    # 加载样本
    print("加载样本数据...")
    samples = load_samples(data_dir, args.num_samples)
    print(f"✓ 成功加载 {len(samples)} 个样本")
    print("")
    
    if len(samples) == 0:
        print("[ERROR] 没有可用的样本数据")
        sys.exit(1)
    
    # 创建可视化器
    visualizer = SceneVisualizer(coord_range=args.coord_range)
    
    # 生成 PDF 报告
    print("生成可视化报告...")
    with PdfPages(output_path) as pdf:
        for idx, (seq_id, sample) in enumerate(samples, 1):
            print(f"  [{idx}/{len(samples)}] 处理场景 {seq_id}")
            
            try:
                fig = visualizer.plot_scene(sample, seq_id)
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"    [WARNING] 场景 {seq_id} 绘制失败: {e}")
                plt.close('all')
                continue
    
    print("")
    print("=" * 80)
    print("✓ 可视化报告生成完成!".center(80))
    print("=" * 80)
    print(f"输出文件: {output_path.absolute()}")
    print(f"包含场景: {len(samples)} 个")
    print("=" * 80)


if __name__ == '__main__':
    main()
