import os
import torch
import wandb
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from functools import partial
from typing import Dict, Any, Optional
from torch.nn.utils import clip_grad_norm_
# 导入DDP相关模块
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from matplotlib import pyplot as plt
from utils.loss import LossRecord
from utils.visualization import visualize_flow_field
import numpy as np

class BaseTrainer:
    """
    一个通用的、支持DDP（分布式数据并行）和断点重训的训练器基类。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        通过一个配置字典来初始化训练器。
        """
        self.config = config
        
        # --- 修正: 从 'train' 和 'log' 子字典中获取配置 ---
        self.train_config = config['train']
        self.log_config = config['log']
        
        # device_id 现在可以是 torch.device 对象或 rank 整数
        self.device_id = self.train_config['device'] 
        self.is_distributed = config['distributed']
        
        # saving_path 来自于 log 配置
        self.saving_path = Path(self.log_config['saving_path']) if self.log_config.get('saving_path') else None

        if self.is_main_process:
            self.logger = logging.info
        else:
            self.logger = lambda *args, **kwargs: None 
        
        self.logger(f"Trainer initialized. Distributed mode: {self.is_distributed}")
        self.logger(f"Device: {self.device_id}")
        self.logger(f"Full configuration loaded.")

    @property
    def is_main_process(self) -> bool:
        """判断当前是否为主进程。"""
        # DDP模式下，主进程的rank为0
        rank = self.config.get('rank', 0)
        return not self.is_distributed or rank == 0

    # --------------------------------------------------------------------------
    #  模块构建与加载
    # --------------------------------------------------------------------------
    
    def get_initializer(self, name: Optional[str]):
        """根据名称获取权重初始化器。"""
        if name is None:
            return None
        initializers = {
            'xavier_normal': partial(torch.nn.init.xavier_normal_),
            'kaiming_uniform': partial(torch.nn.init.kaiming_uniform_),
            'kaiming_normal': partial(torch.nn.init.kaiming_normal_),
        }
        if name in initializers:
            return initializers[name]
        else:
            self.logger(f"Warning: Initializer '{name}' not recognized. Returning None.")
            return None

    def build_optimizer(self, model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
        """根据优化器配置构建优化器。"""
        optimizer_name = optimizer_config.get('optimizer', 'Adam')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 0)
        
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = optimizer_config.get('momentum', 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")

    # --- 修正: 方法签名现在接收特定的配置子字典 ---
    def build_scheduler(self, optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """根据调度器配置构建学习率调度器。"""
        scheduler_name = scheduler_config.get('scheduler')
        if scheduler_name is None:
            return None
        
        # 从 optimize 配置中获取 lr 作为 OneCycleLR 的备用 max_lr
        lr = self.config['optimize'].get('lr', 1e-3)

        if scheduler_name == 'MultiStepLR':
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=scheduler_config['milestones'], gamma=scheduler_config['gamma'])
        elif scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=scheduler_config.get('max_lr', lr), 
                total_steps=scheduler_config.get('total_steps'), # OneCycleLR通常需要总步数
                pct_start=scheduler_config.get('pct_start', 0.3))
        elif scheduler_name == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_config['step_size'], gamma=scheduler_config['gamma'])
        else:
            raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")

    def build_model(self, **kwargs) -> torch.nn.Module:
        """
        构建模型的抽象方法。
        子类必须实现此方法以返回一个 torch.nn.Module 实例。
        """
        raise NotImplementedError("Subclasses must implement the `build_model` method.")
        
    def load_model_from_path(self, path: str) -> torch.nn.Module:
        """从指定路径加载模型配置和权重。"""
        # --- 修正: 假设config.yaml也具有层级结构 ---
        path = Path(path)
        config_path = path / "config.yaml"
        model_path = path / "best_model.pth"
        
        with open(config_path, 'r') as f:
            # 加载整个配置，然后提取模型部分
            full_config = yaml.safe_load(f)
            model_args = full_config['model']
            
        model = self.build_model(**model_args)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model

    # --------------------------------------------------------------------------
    #  核心训练与评估流程
    # --------------------------------------------------------------------------

    # _get_model_state_dict 和 _load_model_state_dict 无需修改

    def _get_model_state_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """智能获取模型状态字典，自动处理DDP包装。"""
        return model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

    def _load_model_state_dict(self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
        """智能加载模型状态字典，自动处理DDP包装。"""
        model_to_load = model.module if isinstance(model, DistributedDataParallel) else model
        model_to_load.load_state_dict(state_dict)

    # train_one_epoch 和 evaluate 无需修改其内部逻辑
    
    def train_one_epoch(self, model: torch.nn.Module, train_loader, optimizer, criterion, metrics, scheduler=None, **kwargs) -> LossRecord:
        loss_record = LossRecord(["train_loss"])
        model.to(self.device_id)
        model.train()
        for (x, y) in train_loader:
            x = x.to(self.device_id)
            y = y.to(self.device_id)
            
            # compute loss
            y_pred = model(x).reshape(y.shape)
            data_loss = criterion(y_pred, y)

            loss = data_loss
            
            # compute gradient 多卡训练时，计算每个GPU上模型副本的局部梯度，然后进行全局梯度求和
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.is_distributed:
                # 集合通信操作，会收集所有的GPU进程上的loss，然后平均，将结果存回每个进程的loss变量中
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            # loss_record.update({"train_loss": loss.sum().item()}, n=y_pred.shape[0])
            loss_record.update({"train_loss": loss.item()}, n=y_pred.shape[0])
        if scheduler is not None:
            scheduler.step()
        return loss_record

    @torch.no_grad()
    def evaluate(self, model, eval_loader, criterion, metrics, split="valid", epoch: int = 0, visualize: bool = False, visual_interval: int = 1, **kwargs):
        metric_names = ['mae','mse', 'rmse','relative_l2', 'psnr','ssim']
        tracked_metrics = [f"{split}_loss"] + [f"{split}_{name}" for name in metric_names]
        loss_record = LossRecord(tracked_metrics)
        
        # 只在主进程且满足可视化条件时创建可视化目录
        should_visualize = (
            self.is_main_process and 
            visualize and 
            split == "valid" and  # 只在验证集上可视化
            (epoch % visual_interval == 0)  # 满足可视化间隔
        )
        if should_visualize:
            vis_dir = self.saving_path / 'visualization' / f"epoch_{epoch:04d}"
            vis_dir.mkdir(exist_ok=True, parents=True)
            self.logger(f"Visualizing sample at epoch {epoch} to {vis_dir}")
        
        model.eval()
        visualized = False
        visualization_path = None
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(eval_loader):
                x = x.to(self.device_id)
                y = y.to(self.device_id)
                # compute loss
                y_pred = model(x).reshape(y.shape)
                
                y = eval_loader.dataset.y_normalizer.decode(y.view(y.shape[0], -1, y.shape[-1])).view(y_pred.shape)
                y_pred = eval_loader.dataset.y_normalizer.decode(y_pred.view(y_pred.shape[0], -1, y_pred.shape[-1])).view(y.shape)
                
                
                data_loss = criterion(y_pred, y)
                mae, mse, rmse, relative_l2, psnr, ssim = metrics(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
                loss = data_loss
                
                update_dict = {
                # f"{split}_loss": loss.sum().item(),
                f"{split}_loss": loss.item(),
                f"{split}_mae": float(mae),
                f"{split}_mse": float(mse),
                f"{split}_rmse": float(rmse),
                f"{split}_relative_l2": float(relative_l2),
                f"{split}_psnr": float(psnr),
                f"{split}_ssim": float(ssim)
            }
                loss_record.update(update_dict, n=y_pred.shape[0])
                
                # 可视化逻辑：每次评估只可视化第一个batch的第一个样本
                if should_visualize and not visualized and batch_idx == 0:
                    sample_factor = self.config['data'].get('sample_factor', [2, 2])[0]
                    
                    # 只取batch中的第一个样本
                    save_filename = vis_dir / f"sample_epoch_{epoch:04d}.png"
                    visualize_flow_field(
                        sr_output=y_pred[0:1], # shape: [1, H, W, C]
                        hr_gt=y[0:1],          # shape: [1, H, W, C]
                        save_path=save_filename,
                        title=f'Epoch {epoch} - Validation Sample',
                        sample_factor=sample_factor
                    )
                    visualized = True
                    visualization_path = save_filename
                    self.logger(f"Saved visualization to {save_filename}")
                    
        if visualization_path:
            loss_record.visualization_path = visualization_path
        return loss_record

    def process(self, model: torch.nn.Module, train_loader, valid_loader, test_loader, optimizer, 
                criterion, metrics, regularizer=None, scheduler=None, **kwargs):
        """
        完整的训练、验证和测试流程，支持断点重训。
        """
        # --- 1. 初始化和DDP封装 ---
        model.to(self.device_id)
        if self.is_distributed:
            model = DistributedDataParallel(model, device_ids=[self.device_id], find_unused_parameters=True)

        start_epoch = 0
        best_epoch = -1
        best_metrics = None
        early_stop_counter = 0

        # --- 2. 断点重训逻辑 ---
        # --- 从 'train' 子字典中获取 resume_path ---
        resume_path = self.train_config.get('resume_path')
        if resume_path and Path(resume_path).exists():
            self.logger(f"Attempting to resume training from checkpoint: {resume_path}")
            try:
                # checkpoint 加载逻辑保持不变
                checkpoint = torch.load(resume_path, map_location= self.device_id)
                self._load_model_state_dict(model, checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                self.logger(f"Successfully resumed from epoch {checkpoint['epoch']}. Starting next epoch at {start_epoch}.")
            except Exception as e:
                self.logger(f"Warning: Failed to load checkpoint. Starting from scratch. Error: {e}")
                start_epoch = 0
        
        self.logger("Starting training...")
        self.logger(f"Train dataset size: {len(train_loader.dataset)}")
        self.logger(f"Valid dataset size: {len(valid_loader.dataset)}")
        self.logger(f"Test dataset size: {len(test_loader.dataset)}")

        # --- 3. 训练主循环 ---
        # --- 修正: 从 'train' 子字典中获取 epochs ---
        epochs = self.train_config.get('epochs', 100)
        for epoch in range(start_epoch, epochs):
            if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_one_epoch(model, train_loader, optimizer, criterion, metrics, scheduler,  **kwargs)

            # --- 修正: 日志记录和评估的配置访问 ---
            if self.is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger(f"Epoch {epoch}/{epochs-1} | {train_loss} | lr: {current_lr:.6f}")
                if self.log_config.get('wandb', False):
                    train_metrics = train_loss.to_dict()
                    train_metrics['train_lr'] = current_lr
                    wandb.log(train_metrics, step=epoch)

            stop_signal = torch.tensor(0, device=self.device_id)

            # 从 'train' 子字典获取 eval_freq
            if (epoch + 1) % self.train_config.get('eval_freq', 1) == 0:
                if self.is_main_process:
                    visualize = self.train_config.get('visualize', False)
                    visual_interval = self.train_config.get('visual_interval', 1)
                    # 解决锁死最关键的步骤
                    model_to_eval = model.module if self.is_distributed else model
                    # valid_loss = self.evaluate(model, valid_loader, criterion, metrics, split="valid",visualize=visualize,num_visuals=num_visuals,epoch=epoch,**kwargs)
                    valid_loss = self.evaluate(model_to_eval, valid_loader, criterion, metrics, split="valid",visualize=visualize,visual_interval=visual_interval,epoch=epoch,**kwargs)
                
                if self.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    self.logger(f"Epoch {epoch}/{epochs-1} | {valid_loss}")
                    valid_metrics = valid_loss.to_dict()
                    valid_metrics['valid_lr'] = current_lr

                    if self.log_config.get('wandb', False):
                        if hasattr(valid_loss, 'visualization_path') and valid_loss.visualization_path:
                            valid_metrics['validation_visualization'] = wandb.Image(
                                str(valid_loss.visualization_path),
                                caption=f"Epoch {epoch} Validation Sample"
                        )
                        wandb.log(valid_metrics, step=epoch)

                    if not best_metrics or valid_metrics['valid_loss'] < best_metrics.get('valid_loss', float('inf')):
                        early_stop_counter = 0
                        best_epoch = epoch
                        best_metrics = valid_metrics
                        # 从 'train' 子字典获取 saving_best 开关
                        if self.train_config.get('saving_best', True) and self.saving_path:
                            best_model_path = self.saving_path /"checkpoint"/ "best_model.pth"
                            best_model_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(self._get_model_state_dict(model), best_model_path)
                            self.logger(f"New best model saved at epoch {epoch} to {best_model_path}")
                    else:
                        # 从 'train' 子字典获取 patience
                        patience = self.train_config.get('patience', -1)
                        if patience != -1:
                            early_stop_counter += 1
                            self.logger(f"Validation loss did not improve. Early stopping counter: {early_stop_counter}/{patience}")
                            if early_stop_counter >= patience:
                                self.logger(f"Early stopping triggered at epoch {epoch}.")
                                stop_signal.fill_(1)
            
            if self.is_distributed:
                dist.broadcast(stop_signal, src=0)#处理早停信号，防止不同进程状态不一致
            if stop_signal.item() == 1:
                break
            
            # --- 保存检查点的配置访问 ---
            if self.is_main_process and self.train_config.get('saving_checkpoint', True) and self.saving_path:
                checkpoint_path = self.saving_path / "checkpoint" / "latest_checkpoint.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                # 保存逻辑不变
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self._get_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_metrics': best_metrics,
                }, checkpoint_path)
                
                # 从 'train' 子字典获取 checkpoint_freq
                if (epoch + 1) % self.train_config.get('checkpoint_freq', 1) == 0:
                    hist_path = self.saving_path / "checkpoint" / f"checkpoint_epoch_{epoch}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict':self._get_model_state_dict(model),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, hist_path)
            if self.is_distributed:
                dist.barrier()      

        self.logger("Training finished. Evaluating on the test set with the best model.")
        
        # 获取底层模型
        model_to_eval = model.module if self.is_distributed else model
        best_model_path = self.saving_path / "checkpoint" / "best_model.pth"

        if best_model_path.exists():
            if self.is_main_process:
                self.logger(f"Loaded model from {best_model_path} (best at epoch {best_epoch}) for final testing.")
                model_to_eval.load_state_dict(torch.load(best_model_path, map_location=self.device_id))
        else:
            if self.is_main_process:
                self.logger("No best model was saved. Using the final model state for testing.")

        if self.is_main_process:
            test_loss = self.evaluate(model_to_eval, test_loader, criterion, metrics, split="test", **kwargs)
    
        # 只有主进程打印最终结果和处理 wandb
        if self.is_main_process:
            self.logger(f"Final Test Metrics: {test_loss}")
            if self.log_config.get('wandb', False):
                final_lr = optimizer.param_groups[0]['lr']
                test_metrics = test_loss.to_dict()
                test_metrics['test_lr'] = final_lr
                
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary["final_lr"] = final_lr
                wandb.run.summary.update(test_metrics)
                wandb.finish()
        
        # 在程序结束前同步所有进程
        if self.is_distributed:
            dist.barrier()
            
        return model