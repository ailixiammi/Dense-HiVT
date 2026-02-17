#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argoverse 1.1 数据集统计分析脚本
用于对原始CSV文件进行数据画像(Data Profiling)，帮助确定模型输入Tensor的维度边界
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


class ArgoverseStatsAnalyzer:
    """Argoverse数据集统计分析器"""
    
    def __init__(self, data_dir: str, sample_size: int = 3000):
        """
        初始化分析器
        
        Args:
            data_dir: CSV文件所在目录路径
            sample_size: 随机采样的文件数量
        """
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        
        # 统计数据容器
        self.total_agents_per_scene: List[int] = []
        self.agents_at_t_obs_per_scene: List[int] = []
        self.target_agents_per_scene: List[int] = []
        self.distances_to_av: List[float] = []  # 所有Agent到AV的距离
        self.distances_target_to_av: List[float] = []  # 仅Target Agent到AV的距离
        self.object_type_counts: Dict[str, int] = defaultdict(int)
        
        # 时间步配置
        self.total_timesteps = 50
        self.t_obs = 19  # 第20帧 (索引从0开始)
        self.min_history_frames = 3  # 最少历史帧数要求
        
        print(f"[INFO] 初始化分析器")
        print(f"  数据目录: {self.data_dir}")
        print(f"  采样数量: {self.sample_size}")
        print(f"  T_obs: 第{self.t_obs + 1}帧 (索引={self.t_obs})")
        print(f"  最少历史帧数: {self.min_history_frames}")
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
    
    def filter_valid_agents(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        过滤有效的Agent
        
        过滤规则:
        1. 历史轨迹(前20帧)中至少出现3帧
        2. 在T_obs时刻必须存在
        
        Args:
            df: 场景的完整DataFrame
            
        Returns:
            filtered_df: 过滤后的DataFrame
            valid_track_ids: 有效的TRACK_ID列表
        """
        # 获取所有唯一的时间戳并排序
        timestamps = sorted(df['TIMESTAMP'].unique())
        
        if len(timestamps) < self.t_obs + 1:
            # 如果时间步数不足，返回空
            return pd.DataFrame(), []
        
        t_obs_timestamp = timestamps[self.t_obs]
        historical_timestamps = timestamps[:self.t_obs + 1]
        
        valid_track_ids = []
        
        for track_id in df['TRACK_ID'].unique():
            track_df = df[df['TRACK_ID'] == track_id]
            
            # 检查历史帧数
            historical_df = track_df[track_df['TIMESTAMP'].isin(historical_timestamps)]
            if len(historical_df) < self.min_history_frames:
                continue
            
            # 检查是否在T_obs存在
            if t_obs_timestamp not in track_df['TIMESTAMP'].values:
                continue
            
            valid_track_ids.append(track_id)
        
        filtered_df = df[df['TRACK_ID'].isin(valid_track_ids)]
        return filtered_df, valid_track_ids
    
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
            required_columns = ['TRACK_ID', 'OBJECT_TYPE', 'TIMESTAMP', 'X', 'Y']
            if not all(col in df.columns for col in required_columns):
                return False
            
            # 过滤有效Agent
            filtered_df, valid_track_ids = self.filter_valid_agents(df)
            
            if len(valid_track_ids) == 0:
                return False
            
            # 获取时间戳
            timestamps = sorted(df['TIMESTAMP'].unique())
            if len(timestamps) < self.t_obs + 1:
                return False
            
            t_obs_timestamp = timestamps[self.t_obs]
            
            # A. 数量统计
            total_agents = len(valid_track_ids)
            self.total_agents_per_scene.append(total_agents)
            
            # T_obs时刻的Agent数量
            t_obs_df = filtered_df[filtered_df['TIMESTAMP'] == t_obs_timestamp]
            agents_at_t_obs = len(t_obs_df['TRACK_ID'].unique())
            self.agents_at_t_obs_per_scene.append(agents_at_t_obs)
            
            # AGENT类型数量
            target_agents = len(filtered_df[filtered_df['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].unique())
            self.target_agents_per_scene.append(target_agents)
            
            # C. 类别统计
            for obj_type in filtered_df['OBJECT_TYPE'].unique():
                count = len(filtered_df[filtered_df['OBJECT_TYPE'] == obj_type]['TRACK_ID'].unique())
                self.object_type_counts[obj_type] += count
            
            # B. 空间统计 - 计算到AV的距离
            av_df = t_obs_df[t_obs_df['OBJECT_TYPE'] == 'AV']
            
            if len(av_df) == 0:
                return False
            
            # 获取AV在T_obs的位置
            av_x = av_df.iloc[0]['X']
            av_y = av_df.iloc[0]['Y']
            
            # 计算所有其他Agent到AV的距离
            for _, row in t_obs_df.iterrows():
                if row['OBJECT_TYPE'] == 'AV':
                    continue  # 跳过AV自己
                
                distance = np.sqrt((row['X'] - av_x)**2 + (row['Y'] - av_y)**2)
                self.distances_to_av.append(distance)
                
                # 单独统计Target Agent到AV的距离
                if row['OBJECT_TYPE'] == 'AGENT':
                    self.distances_target_to_av.append(distance)
            
            return True
            
        except Exception as e:
            # 静默处理错误，避免中断整个统计过程
            return False
    
    def analyze(self) -> None:
        """执行分析"""
        csv_files = self.get_csv_files()
        
        print(f"[INFO] 开始处理 {len(csv_files)} 个场景...")
        print()
        
        success_count = 0
        
        for csv_file in tqdm(csv_files, desc="处理进度", unit="场景"):
            if self.process_single_scene(csv_file):
                success_count += 1
        
        print()
        print(f"[INFO] 处理完成: {success_count}/{len(csv_files)} 个场景成功")
        print()
    
    def generate_report(self) -> str:
        """生成统计报告"""
        if len(self.total_agents_per_scene) == 0:
            return "[ERROR] 没有有效的统计数据"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Argoverse 1.1 数据集统计报告".center(80))
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据目录: {self.data_dir}")
        report_lines.append(f"采样场景总数: {len(self.total_agents_per_scene)}")
        report_lines.append("")
        
        # A. 数量分布统计
        report_lines.append("-" * 80)
        report_lines.append("A. 数量分布统计 (Quantity Distribution)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        # 总Agent数量统计
        total_agents_array = np.array(self.total_agents_per_scene)
        report_lines.append("【每场景总Agent数量】")
        report_lines.append(f"  最小值:          {np.min(total_agents_array)}")
        report_lines.append(f"  最大值:          {np.max(total_agents_array)}")
        report_lines.append(f"  平均值:          {np.mean(total_agents_array):.2f}")
        report_lines.append(f"  中位数:          {np.median(total_agents_array):.2f}")
        report_lines.append(f"  95%分位数:       {np.percentile(total_agents_array, 95):.2f}")
        report_lines.append(f"  99%分位数:       {np.percentile(total_agents_array, 99):.2f}")
        report_lines.append("")
        
        # T_obs时刻Agent数量统计
        t_obs_agents_array = np.array(self.agents_at_t_obs_per_scene)
        report_lines.append(f"【T_obs(第{self.t_obs + 1}帧)时刻可见Agent数量】")
        report_lines.append(f"  最小值:          {np.min(t_obs_agents_array)}")
        report_lines.append(f"  最大值:          {np.max(t_obs_agents_array)}")
        report_lines.append(f"  平均值:          {np.mean(t_obs_agents_array):.2f}")
        report_lines.append(f"  中位数:          {np.median(t_obs_agents_array):.2f}")
        report_lines.append(f"  95%分位数:       {np.percentile(t_obs_agents_array, 95):.2f}")
        report_lines.append(f"  99%分位数:       {np.percentile(t_obs_agents_array, 99):.2f}")
        report_lines.append("")
        
        # 目标AGENT数量统计
        target_agents_array = np.array(self.target_agents_per_scene)
        report_lines.append("【每场景AGENT(待预测目标)数量】")
        report_lines.append(f"  最小值:          {np.min(target_agents_array)}")
        report_lines.append(f"  最大值:          {np.max(target_agents_array)}")
        report_lines.append(f"  平均值:          {np.mean(target_agents_array):.2f}")
        report_lines.append(f"  中位数:          {np.median(target_agents_array):.2f}")
        report_lines.append("")
        
        # B. 空间分布统计
        report_lines.append("-" * 80)
        report_lines.append("B. 空间分布统计 (Spatial Distribution)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        if len(self.distances_to_av) > 0:
            distances_array = np.array(self.distances_to_av)
            total_count = len(distances_array)
            
            # 距离区间统计
            range_0_30 = np.sum(distances_array < 30)
            range_30_50 = np.sum((distances_array >= 30) & (distances_array < 50))
            range_50_100 = np.sum((distances_array >= 50) & (distances_array < 100))
            range_100_plus = np.sum(distances_array >= 100)
            
            report_lines.append("【所有Agent到AV的距离分布】(包括AGENT + OTHERS)")
            report_lines.append(f"  < 30m:           {range_0_30:6d} ({range_0_30/total_count*100:5.2f}%)")
            report_lines.append(f"  30-50m:          {range_30_50:6d} ({range_30_50/total_count*100:5.2f}%)")
            report_lines.append(f"  50-100m:         {range_50_100:6d} ({range_50_100/total_count*100:5.2f}%)")
            report_lines.append(f"  > 100m:          {range_100_plus:6d} ({range_100_plus/total_count*100:5.2f}%)")
            report_lines.append(f"  总计:            {total_count:6d}")
            report_lines.append("")
            
            report_lines.append("【所有Agent距离统计量】")
            report_lines.append(f"  最小距离:        {np.min(distances_array):.2f}m")
            report_lines.append(f"  最大距离:        {np.max(distances_array):.2f}m")
            report_lines.append(f"  平均距离:        {np.mean(distances_array):.2f}m")
            report_lines.append(f"  中位数距离:      {np.median(distances_array):.2f}m")
            report_lines.append("")
        
        # 新增：Target Agent到AV的距离统计
        if len(self.distances_target_to_av) > 0:
            target_distances_array = np.array(self.distances_target_to_av)
            total_count = len(target_distances_array)
            
            # 距离区间统计
            range_0_30 = np.sum(target_distances_array < 30)
            range_30_50 = np.sum((target_distances_array >= 30) & (target_distances_array < 50))
            range_50_100 = np.sum((target_distances_array >= 50) & (target_distances_array < 100))
            range_100_plus = np.sum(target_distances_array >= 100)
            
            report_lines.append("【Target Agent(AGENT)到AV的距离分布】(仅AGENT类型)")
            report_lines.append(f"  < 30m:           {range_0_30:6d} ({range_0_30/total_count*100:5.2f}%)")
            report_lines.append(f"  30-50m:          {range_30_50:6d} ({range_30_50/total_count*100:5.2f}%)")
            report_lines.append(f"  50-100m:         {range_50_100:6d} ({range_50_100/total_count*100:5.2f}%)")
            report_lines.append(f"  > 100m:          {range_100_plus:6d} ({range_100_plus/total_count*100:5.2f}%)")
            report_lines.append(f"  总计:            {total_count:6d}")
            report_lines.append("")
            
            report_lines.append("【Target Agent距离统计量】")
            report_lines.append(f"  最小距离:        {np.min(target_distances_array):.2f}m")
            report_lines.append(f"  最大距离:        {np.max(target_distances_array):.2f}m")
            report_lines.append(f"  平均距离:        {np.mean(target_distances_array):.2f}m")
            report_lines.append(f"  中位数距离:      {np.median(target_distances_array):.2f}m")
            report_lines.append(f"  95%分位数:       {np.percentile(target_distances_array, 95):.2f}m")
            report_lines.append(f"  99%分位数:       {np.percentile(target_distances_array, 99):.2f}m")
            report_lines.append("")
        
        # C. 类别分布统计
        report_lines.append("-" * 80)
        report_lines.append("C. 类别分布统计 (Category Distribution)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        if self.object_type_counts:
            total_objects = sum(self.object_type_counts.values())
            report_lines.append("【OBJECT_TYPE分布】")
            
            # 按数量排序
            sorted_types = sorted(self.object_type_counts.items(), key=lambda x: x[1], reverse=True)
            
            for obj_type, count in sorted_types:
                percentage = count / total_objects * 100
                report_lines.append(f"  {obj_type:15s}  {count:6d} ({percentage:5.2f}%)")
            
            report_lines.append(f"  {'总计':15s}  {total_objects:6d}")
            report_lines.append("")
        
        # 模型建议
        report_lines.append("-" * 80)
        report_lines.append("模型维度建议 (Recommendations for Model Design)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        p95_total = np.percentile(total_agents_array, 95)
        p99_total = np.percentile(total_agents_array, 99)
        p95_t_obs = np.percentile(t_obs_agents_array, 95)
        p99_t_obs = np.percentile(t_obs_agents_array, 99)
        
        report_lines.append("【Tensor维度Padding建议】")
        report_lines.append(f"  基于95%分位数 (覆盖95%场景):")
        report_lines.append(f"    - 总Agent数量 N: {int(np.ceil(p95_total))}")
        report_lines.append(f"    - T_obs可见数 N: {int(np.ceil(p95_t_obs))}")
        report_lines.append("")
        report_lines.append(f"  基于99%分位数 (覆盖99%场景):")
        report_lines.append(f"    - 总Agent数量 N: {int(np.ceil(p99_total))}")
        report_lines.append(f"    - T_obs可见数 N: {int(np.ceil(p99_t_obs))}")
        report_lines.append("")
        
        report_lines.append("【空间范围建议】")
        range_50_percent = np.sum(np.array(self.distances_to_av) < 50) / len(self.distances_to_av) * 100
        range_100_percent = np.sum(np.array(self.distances_to_av) < 100) / len(self.distances_to_av) * 100
        report_lines.append(f"  - 50m半径可覆盖 {range_50_percent:.2f}% 的Agent")
        report_lines.append(f"  - 100m半径可覆盖 {range_100_percent:.2f}% 的Agent")
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
        description='Argoverse 1.1 数据集统计分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  python analyze_argoverse_stats.py --data_dir /root/vc/data/train/data
  
  # 指定采样数量
  python analyze_argoverse_stats.py --data_dir /root/vc/data/train/data --sample_size 5000
  
  # 指定输出文件名
  python analyze_argoverse_stats.py --data_dir /root/vc/data/train/data --output stats_report.txt
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/root/vc/data/train/data',
        help='CSV文件所在目录 (默认: /root/vc/data/train/data)'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=3000,
        help='随机采样的文件数量 (默认: 3000)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出报告文件名 (默认: argoverse_stats_report_YYYYMMDD_HHMMSS.txt)'
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
        analyzer = ArgoverseStatsAnalyzer(
            data_dir=args.data_dir,
            sample_size=args.sample_size
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
            output_filename = f'argoverse_stats_report_{timestamp}.txt'
            output_path = Path(__file__).parent / output_filename
        
        analyzer.save_report(output_path)
        
    except Exception as e:
        print(f"[ERROR] 分析失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
