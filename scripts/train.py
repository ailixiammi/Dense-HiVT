"""
Dense-HiVT 极速训练引擎 (Train Loop)

核心特性:
1. 自动混合精度 (AMP) - 充分利用 RTX 4090 的 Tensor Core
2. AdamW 优化器 + CosineAnnealingLR 学习率调度
3. 梯度裁剪 (Clip Grad Norm = 5.0) - 防止 Laplace NLL 梯度爆炸
4. 最佳模型存档 (基于 Val minFDE)
5. TensorBoard 日志记录
6. 优雅的终端进度条和日志输出

运行方式:
    python scripts/train.py
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.dense_hivt import DenseHiVT
from src.model.loss import DenseHiVTLoss
from src.dataloader.dense_dataset import create_dataloaders
from src.trainer.metrics import compute_metrics


class TrainingEngine:
    """
    Dense-HiVT 训练引擎
    
    功能:
    - 管理训练和验证循环
    - 自动混合精度训练
    - 学习率调度
    - Checkpointing
    - TensorBoard 日志
    """
    
    def __init__(self, args):
        """
        初始化训练引擎
        
        Args:
            args: 命令行参数 (包含超参数配置)
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 初始化模型
        print()
        print("=" * 80)
        print("初始化 Dense-HiVT 模型".center(80))
        print("=" * 80)
        print()
        
        self.model = DenseHiVT(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            global_layers=args.num_global_interactor_layers,
            num_modes=args.num_modes,
            future_steps=args.future_steps,
            dropout=args.dropout
        ).to(self.device)
        
        print(f"模型已加载到设备: {self.device}")
        print(f"总参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print()
        
        # 初始化损失函数
        self.criterion = DenseHiVTLoss()
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min
        )
        
        # 初始化 GradScaler (AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_fde = float('inf')
        self.global_step = 0
        
        print("=" * 80)
        print("✓ 训练引擎初始化完成".center(80))
        print("=" * 80)
        print()
    
    def train_one_epoch(self, train_loader, epoch):
        """
        训练一个 Epoch
        
        Args:
            train_loader: 训练集 DataLoader
            epoch: 当前 Epoch 编号
        
        Returns:
            平均训练损失
        """
        self.model.train()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        
        # 创建进度条
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{self.args.epochs} [Train]",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch_idx, batch in enumerate(pbar):
            # ===================================================================
            # 步骤 1: 数据准备 - 推送到 GPU
            # ===================================================================
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # ===================================================================
            # 步骤 2: 前向传播 (使用 AMP)
            # ===================================================================
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                # 模型前向传播
                outputs = self.model(batch)
                
                # 计算损失
                loss_dict = self.criterion(
                    pi=outputs['pi'],
                    loc=outputs['loc'],
                    y=batch['agent_future_positions'],
                    agent_current_pos=batch['agent_history_positions'][:, :, -1, :],  # [B, N, 2]
                    agent_current_heading=batch['agent_heading'],  # [B, N]
                    reg_mask=batch['agent_future_positions_mask'],
                    valid_mask=batch['agent_is_target']
                )
                
                loss = loss_dict['total_loss']
            
            # ===================================================================
            # 步骤 3: 反向传播 + 梯度裁剪
            # ===================================================================
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.args.grad_clip_norm
            )
            
            # 优化器更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # ===================================================================
            # 步骤 4: 记录损失
            # ===================================================================
            total_loss += loss.item()
            total_reg_loss += loss_dict['reg_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reg': f"{loss_dict['reg_loss'].item():.4f}",
                'cls': f"{loss_dict['cls_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # TensorBoard 记录 (每 100 步)
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/RegLoss', loss_dict['reg_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/ClsLoss', loss_dict['cls_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'reg_loss': avg_reg_loss,
            'cls_loss': avg_cls_loss
        }
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """
        验证模型性能
        
        Args:
            val_loader: 验证集 DataLoader
            epoch: 当前 Epoch 编号
        
        Returns:
            验证指标字典 (minADE, minFDE, MR)
        """
        self.model.eval()
        
        # 累积评测指标
        total_ade = 0.0
        total_fde = 0.0
        total_mr = 0.0
        num_batches = 0
        
        # 创建进度条
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{self.args.epochs} [Val]  ",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch in pbar:
            # ===================================================================
            # 步骤 1: 数据准备
            # ===================================================================
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # ===================================================================
            # 步骤 2: 前向传播 (无梯度)
            # ===================================================================
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                outputs = self.model(batch)
            
            # ===================================================================
            # 步骤 3: 提取预测轨迹位置 (忽略尺度参数)
            # ===================================================================
            # outputs['loc']: [B, N, K, F, 4] (最后一维: [μ_x, μ_y, b_x, b_y])
            # 我们只需要位置预测: [B, N, K, F, 2]
            pred_trajs = outputs['loc'][..., :2]  # [B, N, K, F, 2]
            
            # ===================================================================
            # 步骤 4: 计算评测指标
            # ===================================================================
            metrics = compute_metrics(
                pred_trajs=pred_trajs,
                gt_trajs=batch['agent_future_positions'],
                gt_masks=batch['agent_future_positions_mask'],
                target_masks=batch['agent_is_target'],
                miss_threshold=2.0
            )
            
            # 累加指标 (忽略 NaN)
            if not torch.isnan(torch.tensor(metrics['minADE'])):
                total_ade += metrics['minADE']
                total_fde += metrics['minFDE']
                total_mr += metrics['MR']
                num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'ADE': f"{metrics['minADE']:.4f}",
                'FDE': f"{metrics['minFDE']:.4f}",
                'MR': f"{metrics['MR']:.2%}"
            })
        
        # ===================================================================
        # 步骤 5: 计算平均指标
        # ===================================================================
        avg_ade = total_ade / num_batches if num_batches > 0 else float('nan')
        avg_fde = total_fde / num_batches if num_batches > 0 else float('nan')
        avg_mr = total_mr / num_batches if num_batches > 0 else float('nan')
        
        return {
            'minADE': avg_ade,
            'minFDE': avg_fde,
            'MR': avg_mr
        }
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """
        保存模型 Checkpoint
        
        Args:
            epoch: 当前 Epoch
            val_metrics: 验证指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_fde': self.best_val_fde,
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        # 保存最新的 Checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.checkpoint_dir / "best_dense_hivt.pth"
            torch.save(checkpoint, best_path)
            print(f"\n✓ 最佳模型已保存: {best_path}")
            print(f"  - minFDE: {val_metrics['minFDE']:.4f} 米")
    
    def train(self, train_loader, val_loader):
        """
        主训练循环
        
        Args:
            train_loader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
        """
        print()
        print("=" * 80)
        print("开始训练".center(80))
        print("=" * 80)
        print()
        print(f"总 Epochs: {self.args.epochs}")
        print(f"训练集大小: {len(train_loader.dataset)} 样本")
        print(f"验证集大小: {len(val_loader.dataset)} 样本")
        print(f"Base LR: {self.args.lr}")
        print(f"Weight Decay: {self.args.weight_decay}")
        print(f"Gradient Clip Norm: {self.args.grad_clip_norm}")
        print(f"AMP 启用: {self.args.use_amp}")
        print()
        print("=" * 80)
        print()
        
        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch
            
            # ===================================================================
            # 训练一个 Epoch
            # ===================================================================
            train_metrics = self.train_one_epoch(train_loader, epoch)
            
            # ===================================================================
            # 验证模型
            # ===================================================================
            val_metrics = self.validate(val_loader, epoch)
            
            # ===================================================================
            # 学习率调度
            # ===================================================================
            self.scheduler.step()
            
            # ===================================================================
            # TensorBoard 记录 Epoch 级别指标
            # ===================================================================
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_minADE', val_metrics['minADE'], epoch)
            self.writer.add_scalar('Epoch/Val_minFDE', val_metrics['minFDE'], epoch)
            self.writer.add_scalar('Epoch/Val_MR', val_metrics['MR'], epoch)
            self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # ===================================================================
            # 打印 Epoch 总结
            # ===================================================================
            print()
            print("=" * 80)
            print(f"Epoch {epoch}/{self.args.epochs} 总结".center(80))
            print("=" * 80)
            print()
            print(f"[训练]")
            print(f"  - Total Loss: {train_metrics['loss']:.4f}")
            print(f"  - Reg Loss:   {train_metrics['reg_loss']:.4f}")
            print(f"  - Cls Loss:   {train_metrics['cls_loss']:.4f}")
            print()
            print(f"[验证]")
            print(f"  - minADE: {val_metrics['minADE']:.4f} 米")
            print(f"  - minFDE: {val_metrics['minFDE']:.4f} 米")
            print(f"  - MR:     {val_metrics['MR']:.2%}")
            print()
            print(f"[优化器]")
            print(f"  - Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print()
            
            # ===================================================================
            # 检查是否为最佳模型
            # ===================================================================
            is_best = False
            if val_metrics['minFDE'] < self.best_val_fde:
                self.best_val_fde = val_metrics['minFDE']
                is_best = True
                print(f"🎉 新的最佳 minFDE: {self.best_val_fde:.4f} 米")
                print()
            
            # ===================================================================
            # 保存 Checkpoint
            # ===================================================================
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            print("=" * 80)
            print()
        
        # ===================================================================
        # 训练结束
        # ===================================================================
        print()
        print("=" * 80)
        print("训练完成！".center(80))
        print("=" * 80)
        print()
        print(f"最佳验证 minFDE: {self.best_val_fde:.4f} 米")
        print(f"Checkpoints 保存位置: {self.checkpoint_dir}")
        print(f"TensorBoard 日志: {self.log_dir}")
        print()
        print("=" * 80)
        
        self.writer.close()


def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Dense-HiVT 训练脚本")
    
    # =========================================================================
    # 数据相关
    # =========================================================================
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/root/devdata/Dense-HiVT/data/processed/train',
        help='训练集目录路径'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='/root/devdata/Dense-HiVT/data/processed/val',
        help='验证集目录路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='输出目录（Checkpoints + Logs）'
    )
    
    # =========================================================================
    # 模型超参数
    # =========================================================================
    parser.add_argument('--embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--num_heads', type=int, default=8, help='Multi-Head Attention 头数')
    parser.add_argument('--num_local_encoder_layers', type=int, default=4, help='Local Encoder 层数')
    parser.add_argument('--num_global_interactor_layers', type=int, default=3, help='Global Interactor 层数')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Decoder 层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 概率')
    parser.add_argument('--num_modes', type=int, default=6, help='预测模态数')
    parser.add_argument('--future_steps', type=int, default=30, help='未来时间步数')
    
    # =========================================================================
    # 训练超参数
    # =========================================================================
    parser.add_argument('--epochs', type=int, default=64, help='总训练 Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='训练 Batch Size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base Learning Rate')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='最小学习率（CosineAnnealing）')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight Decay (AdamW)')
    parser.add_argument('--grad_clip_norm', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用 AMP（自动混合精度）')
    
    # =========================================================================
    # DataLoader 配置
    # =========================================================================
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader 进程数')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='使用 Pin Memory')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='预取批次数')
    
    return parser.parse_args()


def main():
    """
    主函数入口
    """
    # 解析命令行参数
    args = parse_args()
    
    # 打印启动信息
    print()
    print("=" * 80)
    print("Dense-HiVT 极速训练引擎".center(80))
    print("=" * 80)
    print()
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA 不可用，将使用 CPU 训练（性能会大幅下降）")
        print()
    else:
        print(f"✓ GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        print()
    
    # =========================================================================
    # 创建 DataLoader
    # =========================================================================
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor
    )
    
    # =========================================================================
    # 初始化训练引擎
    # =========================================================================
    engine = TrainingEngine(args)
    
    # =========================================================================
    # 开始训练
    # =========================================================================
    try:
        engine.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("训练被用户中断".center(80))
        print("=" * 80)
        print()
        engine.writer.close()
    except Exception as e:
        print()
        print("=" * 80)
        print("训练过程中发生错误".center(80))
        print("=" * 80)
        print()
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        engine.writer.close()


if __name__ == "__main__":
    main()