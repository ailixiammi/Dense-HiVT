#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argoverse 1.1 地图特征统计分析脚本
用于分析AV周围局部地图的车道线密度和几何复杂度
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Argoverse Map API
try:
    from argoverse.map_representation.map_api import ArgoverseMap
except ImportError:
    print("[ERROR] 无法导入 argoverse 库")
    print("请安装: pip install argoverse")
    sys.exit(1)


class ArgoverseMapStatsAnalyzer:
    """Argoverse地图特征统计分析器"""
    
    def __init__(self, 
                 data_dir: str,
                 map_dir: str,
                 sample_size: int = 500,
                 radius: float = 100.0):
        """
        初始化分析器
        
        Args:
            data_dir: CSV文件所在目录路径
            map_dir: 地图文件目录路径
            sample_size: 随机采样的文件数量
            radius: 查询半径(米)
        """
        self.data_dir = Path(data_dir)
        self.map_dir = Path(map_dir)
        self.sample_size = sample_size
        self.radius = radius
        
        # 统计数据容器
        self.lane_counts_per_scene: List[int] = []  # 每场景的车道数量(L)
        self.lane_points_per_lane: List[int] = []   # 每条车道的点数(S)
        self.lane_lengths: List[float] = []         # 每条车道的长度(米)
        
        # 时间步配置
        self.t_obs = 19  # 第20帧 (索引从0开始)
        
        # 初始化Map API
        print(f"[INFO] 初始化Argoverse Map API...")
        print(f"  Map目录: {self.map_dir}")
        
        # 设置环境变量指向map目录
        os.environ['ARGOVERSE_DATA_DIR'] = str(self.map_dir.parent)
        
        try:
            self.am = ArgoverseMap()
            print(f"[INFO] Map API初始化成功")
        except Exception as e:
            print(f"[ERROR] Map API初始化失败: {e}")
            raise
        
        print(f"\n[INFO] 初始化分析器")
        print(f"  数据目录: {self.data_dir}")
        print(f"  采样数量: {self.sample_size}")
        print(f"  查询半径: {self.radius}m")
        print(f"  T_obs: 第{self.t_obs + 1}帧 (索引={self.t_obs})")
        print()
    
    def get_csv_files(self) -> List[Path]:
        """获取所有CSV文件并随机采样"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        all_csv_files = list(self.data_dir.glob("*.csv"))
        
        if len(all_csv_files) == 0:
            raise ValueError(f"未找到CSV文件: {self.data_dir}")
        
        print(f"[INFO] 找到 {len(all_csv_files)} 个CSV文件")
        
        # 随机采样
        sample_size = min(self.sample_size, len(all_csv_files))
        sampled_files = random.sample(all_csv_files, sample_size)
        
        print(f"[INFO] 随机采样 {sample_size} 个文件进行分析")
        print()
        
        return sampled_files
    
    def get_av_info_at_t_obs(self, df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        """
        获取AV在T_obs时刻的信息
        
        Args:
            df: 场景的完整DataFrame
            
        Returns:
            (city_name, x, y) 或 None
        """
        # 获取所有唯一的时间戳并排序
        timestamps = sorted(df['TIMESTAMP'].unique())
        
        if len(timestamps) < self.t_obs + 1:
            return None
        
        t_obs_timestamp = timestamps[self.t_obs]
        
        # 获取AV在T_obs的数据
        av_df = df[(df['OBJECT_TYPE'] == 'AV') & (df['TIMESTAMP'] == t_obs_timestamp)]
        
        if len(av_df) == 0:
            return None
        
        av_row = av_df.iloc[0]
        city_name = av_row['CITY_NAME']
        x = av_row['X']
        y = av_row['Y']
        
        return (city_name, x, y)
    
    def calculate_lane_length(self, centerline: np.ndarray) -> float:
        """
        计算车道中心线的长度
        
        Args:
            centerline: [N, 2] 点数组
            
        Returns:
            长度(米)
        """
        if len(centerline) < 2:
            return 0.0
        
        # 计算相邻点之间的距离
        diffs = np.diff(centerline, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(distances)
        
        return float(total_length)
    
    def process_single_scene(self, csv_path: Path) -> bool:
        """
        处理单个场景的CSV文件
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            是否成功处理
        """
        try:
            df = pd.read_csv(csv_path)
            
            # 数据验证
            required_columns = ['TRACK_ID', 'OBJECT_TYPE', 'TIMESTAMP', 'X', 'Y', 'CITY_NAME']
            if not all(col in df.columns for col in required_columns):
                return False
            
            # 获取AV信息
            av_info = self.get_av_info_at_t_obs(df)
            if av_info is None:
                return False
            
            city_name, av_x, av_y = av_info
            
            # 使用Map API查询车道
            try:
                lane_ids = self.am.get_lane_ids_in_xy_bbox(
                    av_x, av_y, city_name, self.radius
                )
            except Exception:
                # 地图查询失败，跳过该场景
                return False
            
            if not lane_ids:
                # 没有查询到车道，跳过
                return False
            
            # 统计车道数量
            lane_count = len(lane_ids)
            self.lane_counts_per_scene.append(lane_count)
            
            # 遍历每条车道，统计点数和长度
            for lane_id in lane_ids:
                try:
                    # 获取车道中心线
                    centerline = self.am.get_lane_segment_centerline(lane_id, city_name)
                    
                    if centerline is None or len(centerline) == 0:
                        continue
                    
                    # 统计点数
                    num_points = len(centerline)
                    self.lane_points_per_lane.append(num_points)
                    
                    # 计算长度
                    lane_length = self.calculate_lane_length(centerline[:, :2])
                    self.lane_lengths.append(lane_length)
                    
                except Exception:
                    # 单条车道查询失败，跳过
                    continue
            
            return True
            
        except Exception:
            # 静默处理错误，避免中断整个统计过程
            return False
    
    def analyze(self) -> None:
        """执行分析"""
        csv_files = self.get_csv_files()
        
        print(f"[INFO] 开始处理 {len(csv_files)} 个场景...")
        print(f"[INFO] 注意: Map API查询可能较慢，预计需要 1-3 分钟")
        print()
        
        success_count = 0
        
        for csv_file in tqdm(csv_files, desc="处理进度", unit="场景"):
            if self.process_single_scene(csv_file):
                success_count += 1
        
        print()
        print(f"[INFO] 处理完成: {success_count}/{len(csv_files)} 个场景成功")
        print(f"[INFO] 统计到 {len(self.lane_points_per_lane)} 条车道")
        print()
    
    def generate_report(self) -> str:
        """生成统计报告"""
        if len(self.lane_counts_per_scene) == 0:
            return "[ERROR] 没有有效的统计数据"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Argoverse 1.1 地图特征统计报告".center(80))
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据目录: {self.data_dir}")
        report_lines.append(f"地图目录: {self.map_dir}")
        report_lines.append(f"查询半径: {self.radius}m")
        report_lines.append(f"采样场景总数: {len(self.lane_counts_per_scene)}")
        report_lines.append(f"统计车道总数: {len(self.lane_points_per_lane)}")
        report_lines.append("")
        
        # A. 车道数量分布统计 (L维度)
        report_lines.append("-" * 80)
        report_lines.append("A. 车道数量分布 (Lane Count Distribution - L)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        lane_counts_array = np.array(self.lane_counts_per_scene)
        report_lines.append(f"【每场景车道数量 (半径{self.radius}m内)】")
        report_lines.append(f"  最小值:          {np.min(lane_counts_array)}")
        report_lines.append(f"  最大值:          {np.max(lane_counts_array)}")
        report_lines.append(f"  平均值:          {np.mean(lane_counts_array):.2f}")
        report_lines.append(f"  中位数:          {np.median(lane_counts_array):.2f}")
        report_lines.append(f"  标准差:          {np.std(lane_counts_array):.2f}")
        report_lines.append(f"  95%分位数:       {np.percentile(lane_counts_array, 95):.2f}  ← 推荐 Max_Lanes")
        report_lines.append(f"  99%分位数:       {np.percentile(lane_counts_array, 99):.2f}  ← 保守 Max_Lanes")
        report_lines.append("")
        
        # B. 车道点数分布统计 (S维度)
        report_lines.append("-" * 80)
        report_lines.append("B. 车道点数分布 (Lane Points Distribution - S)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        lane_points_array = np.array(self.lane_points_per_lane)
        report_lines.append("【每条车道中心线点数】")
        report_lines.append(f"  最小值:          {np.min(lane_points_array)}")
        report_lines.append(f"  最大值:          {np.max(lane_points_array)}")
        report_lines.append(f"  平均值:          {np.mean(lane_points_array):.2f}")
        report_lines.append(f"  中位数:          {np.median(lane_points_array):.2f}")
        report_lines.append(f"  标准差:          {np.std(lane_points_array):.2f}")
        report_lines.append(f"  95%分位数:       {np.percentile(lane_points_array, 95):.2f}  ← 推荐 Max_Points")
        report_lines.append(f"  99%分位数:       {np.percentile(lane_points_array, 99):.2f}  ← 保守 Max_Points")
        report_lines.append("")
        
        # 点数分布区间统计
        total_lanes = len(lane_points_array)
        range_2_10 = np.sum((lane_points_array >= 2) & (lane_points_array <= 10))
        range_11_20 = np.sum((lane_points_array >= 11) & (lane_points_array <= 20))
        range_21_30 = np.sum((lane_points_array >= 21) & (lane_points_array <= 30))
        range_31_plus = np.sum(lane_points_array > 30)
        
        report_lines.append("【点数区间分布】")
        report_lines.append(f"  2-10点:          {range_2_10:6d} ({range_2_10/total_lanes*100:5.2f}%)")
        report_lines.append(f"  11-20点:         {range_11_20:6d} ({range_11_20/total_lanes*100:5.2f}%)")
        report_lines.append(f"  21-30点:         {range_21_30:6d} ({range_21_30/total_lanes*100:5.2f}%)")
        report_lines.append(f"  >30点:           {range_31_plus:6d} ({range_31_plus/total_lanes*100:5.2f}%)")
        report_lines.append("")
        
        # C. 车道长度分布统计
        report_lines.append("-" * 80)
        report_lines.append("C. 车道长度分布 (Lane Length Distribution)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        lane_lengths_array = np.array(self.lane_lengths)
        report_lines.append("【每条车道长度(米)】")
        report_lines.append(f"  最小值:          {np.min(lane_lengths_array):.2f}m")
        report_lines.append(f"  最大值:          {np.max(lane_lengths_array):.2f}m")
        report_lines.append(f"  平均值:          {np.mean(lane_lengths_array):.2f}m")
        report_lines.append(f"  中位数:          {np.median(lane_lengths_array):.2f}m")
        report_lines.append(f"  标准差:          {np.std(lane_lengths_array):.2f}m")
        report_lines.append("")
        
        # 长度分布区间统计
        range_0_20 = np.sum(lane_lengths_array < 20)
        range_20_50 = np.sum((lane_lengths_array >= 20) & (lane_lengths_array < 50))
        range_50_100 = np.sum((lane_lengths_array >= 50) & (lane_lengths_array < 100))
        range_100_plus = np.sum(lane_lengths_array >= 100)
        
        report_lines.append("【长度区间分布】")
        report_lines.append(f"  < 20m:           {range_0_20:6d} ({range_0_20/total_lanes*100:5.2f}%)")
        report_lines.append(f"  20-50m:          {range_20_50:6d} ({range_20_50/total_lanes*100:5.2f}%)")
        report_lines.append(f"  50-100m:         {range_50_100:6d} ({range_50_100/total_lanes*100:5.2f}%)")
        report_lines.append(f"  > 100m:          {range_100_plus:6d} ({range_100_plus/total_lanes*100:5.2f}%)")
        report_lines.append("")
        
        # 模型设计建议
        report_lines.append("-" * 80)
        report_lines.append("模型设计建议 (Recommendations for Model Design)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        p95_lanes = int(np.ceil(np.percentile(lane_counts_array, 95)))
        p99_lanes = int(np.ceil(np.percentile(lane_counts_array, 99)))
        p95_points = int(np.ceil(np.percentile(lane_points_array, 95)))
        p99_points = int(np.ceil(np.percentile(lane_points_array, 99)))
        
        report_lines.append("【推荐的Tensor维度】")
        report_lines.append("")
        report_lines.append("  基于95%分位数 (覆盖95%场景，推荐使用):")
        report_lines.append(f"    - Lane数量维度 L (Max_Lanes):     {p95_lanes}")
        report_lines.append(f"    - 每条Lane点数 S (Max_Points):    {p95_points}")
        report_lines.append("")
        report_lines.append("  基于99%分位数 (覆盖99%场景，更保守):")
        report_lines.append(f"    - Lane数量维度 L (Max_Lanes):     {p99_lanes}")
        report_lines.append(f"    - 每条Lane点数 S (Max_Points):    {p99_points}")
        report_lines.append("")
        
        report_lines.append("【覆盖率说明】")
        report_lines.append("  - 95%方案: 平衡效率与覆盖率，超出部分可截断")
        report_lines.append("  - 99%方案: 更高覆盖率，但会增加内存和计算开销")
        report_lines.append("")
        
        report_lines.append("【实际应用建议】")
        p95_memory = p95_lanes * p95_points * 2 * 4 / 1024  # 假设float32, 2D坐标
        p99_memory = p99_lanes * p99_points * 2 * 4 / 1024
        report_lines.append(f"  - 95%方案单样本地图Tensor大小: ~{p95_memory:.2f}KB")
        report_lines.append(f"  - 99%方案单样本地图Tensor大小: ~{p99_memory:.2f}KB")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, output_path: Path) -> None:
        """保存报告到文件"""
        report = self.generate_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[INFO] 报告已保存至: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Argoverse 1.1 地图特征统计分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  python analyze_map_stats.py --data_dir /root/vc/data/train/data --map_dir /root/vc/data/maps
  
  # 指定采样数量和查询半径
  python analyze_map_stats.py --data_dir /root/vc/data/train/data --map_dir /root/vc/data/maps --sample_size 1000 --radius 100
  
  # 指定输出文件名
  python analyze_map_stats.py --data_dir /root/vc/data/train/data --map_dir /root/vc/data/maps --output map_stats_report.txt
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/root/vc/data/train/data',
        help='CSV文件所在目录 (默认: /root/vc/data/train/data)'
    )
    
    parser.add_argument(
        '--map_dir',
        type=str,
        default='/root/vc/data/maps',
        help='地图文件目录 (默认: /root/vc/data/maps)'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=500,
        help='随机采样的文件数量 (默认: 500)'
    )
    
    parser.add_argument(
        '--radius',
        type=float,
        default=100.0,
        help='查询半径(米) (默认: 100.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出报告文件名 (默认: map_stats_report_YYYYMMDD_HHMMSS.txt)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # 创建分析器
        analyzer = ArgoverseMapStatsAnalyzer(
            data_dir=args.data_dir,
            map_dir=args.map_dir,
            sample_size=args.sample_size,
            radius=args.radius
        )
        
        # 执行分析
        analyzer.analyze()
        
        # 生成并打印报告
        report = analyzer.generate_report()
        print(report)
        
        # 保存报告
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'map_stats_report_{timestamp}.txt'
            output_path = Path(__file__).parent / output_filename
        
        analyzer.save_report(output_path)
        
    except Exception as e:
        print(f"[ERROR] 分析失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
