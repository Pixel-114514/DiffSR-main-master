
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
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        is_distributed = True
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
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