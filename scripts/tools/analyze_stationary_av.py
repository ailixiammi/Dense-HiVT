#!/usr/bin/env python3
"""
统计Argoverse数据集中AV静止状态的分布

统计指标：
1. AV在50个时间步中出现静止状态的占比
2. AV在T=19时刻为静止状态的占比  
3. AV全程50个时间步都为静止状态的概率
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_displacement(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两个位置之间的欧式距离"""
    return np.linalg.norm(pos1 - pos2)


def analyze_single_scene(csv_path: str, 
                         displacement_threshold: float = 0.1) -> Dict:
    """
    分析单个场景中AV的静止状态
    
    Args:
        csv_path: CSV文件路径
        displacement_threshold: 判定为静止的位移阈值（米）
        
    Returns:
        stats: 包含该场景统计信息的字典
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None
    
    # 提取AV数据
    av_df = df[df['OBJECT_TYPE'] == 'AV'].copy()
    
    if len(av_df) == 0:
        print(f"Warning: No AV found in {csv_path}")
        return None
    
    # 按时间戳排序
    av_df = av_df.sort_values('TIMESTAMP')
    timestamps = sorted(df['TIMESTAMP'].unique())
    
    if len(timestamps) < 50:
        print(f"Warning: Less than 50 timestamps in {csv_path}")
        return None
    
    # 提取位置序列
    positions = []
    for t in range(50):
        if t < len(timestamps):
            timestamp = timestamps[t]
            row = av_df[av_df['TIMESTAMP'] == timestamp]
            if len(row) > 0:
                pos = row[['X', 'Y']].values[0]
                positions.append(pos)
            else:
                positions.append(None)
        else:
            positions.append(None)
    
    # 计算每个时间步的位移
    stationary_flags = []
    for t in range(1, 50):
        if positions[t] is not None and positions[t-1] is not None:
            displacement = compute_displacement(positions[t], positions[t-1])
            is_stationary = displacement < displacement_threshold
            stationary_flags.append(is_stationary)
        else:
            # 如果数据缺失，不计入统计
            stationary_flags.append(None)
    
    # 统计指标1：任意时刻出现静止的占比
    valid_flags = [f for f in stationary_flags if f is not None]
    if len(valid_flags) > 0:
        stationary_ratio = sum(valid_flags) / len(valid_flags)
    else:
        stationary_ratio = 0.0
    
    # 统计指标2：T=19时刻是否静止
    # T=19对应的是第19->20的位移（索引18，因为stationary_flags从t=1开始）
    t19_stationary = None
    if 18 < len(stationary_flags) and stationary_flags[18] is not None:
        t19_stationary = stationary_flags[18]
    
    # 统计指标3：全程是否都静止
    all_stationary = False
    if len(valid_flags) > 0 and len(valid_flags) == len(stationary_flags):
        all_stationary = all(valid_flags)
    
    return {
        'csv_path': csv_path,
        'seq_id': Path(csv_path).stem,
        'stationary_ratio': stationary_ratio,
        'num_stationary_steps': sum(valid_flags),
        'num_valid_steps': len(valid_flags),
        't19_stationary': t19_stationary,
        'all_stationary': all_stationary,
        'positions': positions,
        'stationary_flags': stationary_flags
    }


def analyze_dataset(data_dir: str, 
                   num_samples: int = 3000,
                   displacement_threshold: float = 0.1,
                   seed: int = 42) -> List[Dict]:
    """
    分析数据集中的样本
    
    Args:
        data_dir: 数据目录路径
        num_samples: 采样数量
        displacement_threshold: 静止判定阈值（米）
        seed: 随机种子
        
    Returns:
        results: 所有样本的统计结果列表
    """
    # 获取所有CSV文件
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return []
    
    csv_files = list(data_path.glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {data_dir}")
        return []
    
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # 采样
    random.seed(seed)
    if len(csv_files) > num_samples:
        sampled_files = random.sample(csv_files, num_samples)
        print(f"Randomly sampled {num_samples} files")
    else:
        sampled_files = csv_files
        print(f"Using all {len(csv_files)} files")
    
    # 分析每个样本
    results = []
    for csv_file in tqdm(sampled_files, desc="Analyzing scenes"):
        result = analyze_single_scene(str(csv_file), displacement_threshold)
        if result is not None:
            results.append(result)
    
    return results


def print_statistics(results: List[Dict]):
    """打印统计结果"""
    if len(results) == 0:
        print("No valid results to analyze")
        return
    
    print("\n" + "="*80)
    print("统计结果汇总")
    print("="*80)
    
    # 指标1：AV在50个时间步中出现静止状态的占比
    stationary_ratios = [r['stationary_ratio'] for r in results]
    avg_stationary_ratio = np.mean(stationary_ratios)
    
    print(f"\n【指标1】AV在时间序列中出现静止状态的占比:")
    print(f"  - 平均值: {avg_stationary_ratio*100:.2f}%")
    print(f"  - 中位数: {np.median(stationary_ratios)*100:.2f}%")
    print(f"  - 最小值: {np.min(stationary_ratios)*100:.2f}%")
    print(f"  - 最大值: {np.max(stationary_ratios)*100:.2f}%")
    print(f"  - 标准差: {np.std(stationary_ratios)*100:.2f}%")
    
    # 分布统计
    bins = [0, 0.01, 0.05, 0.1, 0.2, 1.0]
    hist, _ = np.histogram(stationary_ratios, bins=bins)
    print(f"\n  分布情况:")
    print(f"    - [0%, 1%):     {hist[0]} 场景 ({hist[0]/len(results)*100:.1f}%)")
    print(f"    - [1%, 5%):     {hist[1]} 场景 ({hist[1]/len(results)*100:.1f}%)")
    print(f"    - [5%, 10%):    {hist[2]} 场景 ({hist[2]/len(results)*100:.1f}%)")
    print(f"    - [10%, 20%):   {hist[3]} 场景 ({hist[3]/len(results)*100:.1f}%)")
    print(f"    - [20%, 100%]:  {hist[4]} 场景 ({hist[4]/len(results)*100:.1f}%)")
    
    # 指标2：T=19时刻为静止状态的占比
    t19_stationary_list = [r['t19_stationary'] for r in results if r['t19_stationary'] is not None]
    if len(t19_stationary_list) > 0:
        t19_stationary_count = sum(t19_stationary_list)
        t19_stationary_ratio = t19_stationary_count / len(t19_stationary_list)
        
        print(f"\n【指标2】AV在T=19时刻为静止状态的场景:")
        print(f"  - 静止场景数: {t19_stationary_count}")
        print(f"  - 总场景数: {len(t19_stationary_list)}")
        print(f"  - 占比: {t19_stationary_ratio*100:.2f}%")
    else:
        print(f"\n【指标2】无有效的T=19时刻数据")
    
    # 指标3：全程50个时间步都为静止状态的概率
    all_stationary_count = sum([r['all_stationary'] for r in results])
    all_stationary_ratio = all_stationary_count / len(results)
    
    print(f"\n【指标3】AV全程50个时间步都为静止状态的场景:")
    print(f"  - 静止场景数: {all_stationary_count}")
    print(f"  - 总场景数: {len(results)}")
    print(f"  - 概率: {all_stationary_ratio*100:.4f}%")
    
    # 额外统计：至少有一个时间步静止的场景
    has_stationary_count = sum([1 for r in results if r['num_stationary_steps'] > 0])
    has_stationary_ratio = has_stationary_count / len(results)
    
    print(f"\n【额外统计】至少有一个时间步静止的场景:")
    print(f"  - 场景数: {has_stationary_count}")
    print(f"  - 占比: {has_stationary_ratio*100:.2f}%")
    
    # 高静止占比场景（>10%）
    high_stationary = [r for r in results if r['stationary_ratio'] > 0.1]
    if len(high_stationary) > 0:
        print(f"\n【高静止占比场景】静止比例>10%的场景:")
        print(f"  - 场景数: {len(high_stationary)}")
        print(f"  - 占比: {len(high_stationary)/len(results)*100:.2f}%")
        print(f"  - 示例场景ID:")
        for i, r in enumerate(high_stationary[:5]):
            print(f"    {i+1}. {r['seq_id']}: {r['stationary_ratio']*100:.1f}% 静止")
    
    print("\n" + "="*80)


def save_detailed_results(results: List[Dict], output_path: str):
    """保存详细结果到CSV"""
    records = []
    for r in results:
        records.append({
            'seq_id': r['seq_id'],
            'stationary_ratio': r['stationary_ratio'],
            'num_stationary_steps': r['num_stationary_steps'],
            'num_valid_steps': r['num_valid_steps'],
            't19_stationary': r['t19_stationary'],
            'all_stationary': r['all_stationary']
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\n详细结果已保存到: {output_path}")


def main():
    """主函数"""
    # 参数设置
    data_dir = "/root/vc/data/train/data"
    num_samples = 3000
    displacement_threshold = 0.1  # 0.1米
    output_csv = "stationary_av_analysis.csv"
    
    print("="*80)
    print("Argoverse AV静止状态分析工具")
    print("="*80)
    print(f"\n配置参数:")
    print(f"  - 数据目录: {data_dir}")
    print(f"  - 采样数量: {num_samples}")
    print(f"  - 静止阈值: {displacement_threshold} 米")
    print(f"  - 输出文件: {output_csv}")
    print()
    
    # 分析数据集
    results = analyze_dataset(
        data_dir=data_dir,
        num_samples=num_samples,
        displacement_threshold=displacement_threshold
    )
    
    if len(results) == 0:
        print("Error: No valid results obtained")
        return
    
    # 打印统计结果
    print_statistics(results)
    
    # 保存详细结果
    save_detailed_results(results, output_csv)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()