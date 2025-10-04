import argparse
import logging
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any
import os
import sys

from utils.helper import (
    set_seed,
    load_config,
    save_config,
    create_experiment_dir,
    setup_logging
)
# 假设训练流程在 'procedures' 模块中
from procedures import ns2d_procedure
from utils.dist_util import setup_ddp, cleanup_ddp


def main():
    """主训练脚本入口"""
    # 1. 初始化DDP环境 (这是第一步，因为它决定了后续操作的行为)
    ddp_info = setup_ddp()
    is_main_process = (not ddp_info["distributed"]) or (ddp_info["rank"] == 0)

    # 2. 解析命令行参数
    parser = argparse.ArgumentParser(description="Main training script.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    parser.add_argument('--gpu_id', type=int, default=None, help="GPU ID to use for single-card training.")
    args = parser.parse_args()

    # 3. 加载并整合配置
    # 从文件加载基础配置
    config = load_config(Path(args.config))
    if not config:
        if is_main_process:
            print("Error: Failed to load config file. Exiting.")
        return

    # 将DDP信息和命令行参数整合到config中
    # 注意：这里的更新是针对顶级键的，例如 'distributed', 'rank' 等
    config.update(ddp_info)
    if args.gpu_id is not None and not config['train']['distributed']:
        config['train']['device'] = torch.device(f"cuda:{args.gpu_id}")
    else:
        # 确保即使在DDP模式下，device信息也存在
        config['train']['device'] = ddp_info['device']

    
    # 4. (仅在主进程) 创建实验目录并设置日志
    exp_dir = None
    if is_main_process:
        try:
            # 修正：使用正确的层级访问配置
            exp_dir = create_experiment_dir(
                base_path=config['log'].get('log_dir', './experiments'), # 使用.get提供默认值
                model_name=config['train']['model_name'],
                dataset_name=config['data']['dataset']
            )
            setup_logging(exp_dir)
            # 将实验目录路径存入config，方便后续使用
            config['log']['saving_path'] = str(exp_dir) 
            save_config(config, exp_dir) # 保存最终的完整配置
        except KeyError as e:
            logging.error(f"Missing required key in config file: {e}")
            return
    
    # 所有进程同步实验目录路径
    # 确保了主进程也能知道模型保存在哪里
    if config['distributed']:
        # 将路径对象列表广播给所有进程
        path_list = [str(exp_dir)] if is_main_process else [None]
        dist.broadcast_object_list(path_list, src=0)
        if not is_main_process:
            config['log']['saving_path'] = path_list[0]

    set_seed(config['train'].get('random_seed', 42))
    
    logging.info(f"Starting experiment setup on rank {config['rank']}.")
    logging.info(f"Device: {config['train']['device']}")
    if is_main_process:
        logging.info(f"Experiment directory: {config['log'].get('saving_path', 'Not created')}")
        
    

        
    # 6. 执行具体的训练流程
    try:
        if config['data']['dataset'] == 'NavierStokes2D':
            ns2d_procedure(config)
        else:
            # 修正：使用正确的层级访问
            raise NotImplementedError(f"Dataset '{config['data']['dataset']}' is not implemented.")
            
    except Exception as e:
        logging.error(f"An error occurred during the training procedure on rank {config['rank']}: {e}", exc_info=True)
    finally:
        # 7. 清理DDP环境
        cleanup_ddp()
        logging.info(f"Process {config['rank']} finished.")


if __name__ == "__main__":
    main()