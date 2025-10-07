
import torch
import torch.distributed as dist
from typing import Dict, Any
import os



def setup_ddp() -> Dict[str, Any]:
    """
    检测并初始化DDP环境。

    Returns:
        Dict[str, Any]: 包含DDP状态的字典 
                         (is_distributed, rank, local_rank, world_size, device)。
    """
    # 当使用torchrun等工具启动时，会自动设置环境变量，环境变量中会包含RANK(进程的编号)和WORLD_SIZE(GPU数✖进程数)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        is_distributed = True
        
        # 告诉当前进程应该使用哪一块GPU，确保同一台机器上的每个进程都绑定到一块不同的GPU上
        torch.cuda.set_device(local_rank)
        # 初始化DDP进程组，确保不同进程之间可以互相通信
        dist.init_process_group(backend='nccl', init_method='env://')
        # 设置一个同步点，等待所有进程同步
        dist.barrier()
        
        device = torch.device(f"cuda:{local_rank}")
        
    else:
        # 单卡或CPU模式
        is_distributed = False
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "distributed": is_distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device
    }

def cleanup_ddp():
    """销毁DDP进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()