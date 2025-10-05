"""
多卡训练调试脚本
用于验证分布式训练的正确性
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np


class DummyDataset(Dataset):
    """简单的测试数据集"""
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], idx  # 返回索引用于调试


def test_data_sampling(distributed=True):
    """测试数据采样是否正确"""
    print("\n=== 测试数据采样 ===")
    
    dataset = DummyDataset(size=32)
    
    if distributed and dist.is_initialized():
        sampler = DistributedSampler(dataset)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        sampler = None
        rank = 0
        world_size = 1
    
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, shuffle=(sampler is None))
    
    # 收集所有索引
    all_indices = []
    for x, y, indices in loader:
        all_indices.extend(indices.tolist())
        print(f"Rank {rank}: batch indices = {indices.tolist()}")
    
    print(f"Rank {rank}: Total samples seen = {len(all_indices)}")
    print(f"Rank {rank}: Unique samples = {len(set(all_indices))}")
    
    # 验证是否有重复
    if len(all_indices) != len(set(all_indices)):
        print(f"⚠️  Rank {rank}: 发现重复样本!")
    else:
        print(f"✅ Rank {rank}: 无重复样本")
    
    # 在分布式环境中,收集所有rank的索引
    if distributed and dist.is_initialized():
        # 使用gather收集所有rank的索引
        gathered_indices = [None] * world_size
        dist.all_gather_object(gathered_indices, all_indices)
        
        if rank == 0:
            print("\n=== 全局采样检查 ===")
            all_global_indices = []
            for r, indices in enumerate(gathered_indices):
                print(f"Rank {r} processed {len(indices)} samples")
                all_global_indices.extend(indices)
            
            print(f"Total samples across all ranks: {len(all_global_indices)}")
            print(f"Unique samples: {len(set(all_global_indices))}")
            
            if len(set(all_global_indices)) != len(dataset):
                print(f"⚠️  警告: 某些样本被跳过或重复!")
            else:
                print(f"✅ 所有样本都被处理且无重复")


def test_loss_aggregation():
    """测试loss聚合逻辑"""
    print("\n=== 测试Loss聚合 ===")
    
    if not dist.is_initialized():
        print("❌ 分布式未初始化,跳过此测试")
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 模拟每个GPU的local loss
    local_losses = [0.5, 0.3]  # GPU0=0.5, GPU1=0.3
    local_loss = torch.tensor(local_losses[rank], device=f'cuda:{rank}')
    
    print(f"Rank {rank}: Local loss = {local_loss.item()}")
    
    # 方法1: 错误的做法 (原代码)
    loss_wrong = local_loss.clone()
    dist.all_reduce(loss_wrong, op=dist.ReduceOp.AVG)
    print(f"Rank {rank}: After all_reduce(AVG) = {loss_wrong.item()}")
    # 如果用 n=batch_size 更新,会导致double averaging
    
    # 方法2: 正确的做法 (修复后)
    loss_correct = local_loss.clone()
    dist.all_reduce(loss_correct, op=dist.ReduceOp.AVG)
    print(f"Rank {rank}: Correct approach - use n=1 with avg loss = {loss_correct.item()}")
    
    # 验证
    expected_avg = sum(local_losses) / len(local_losses)
    if abs(loss_correct.item() - expected_avg) < 1e-6:
        print(f"✅ Rank {rank}: Loss聚合正确! Expected={expected_avg}, Got={loss_correct.item()}")
    else:
        print(f"❌ Rank {rank}: Loss聚合错误! Expected={expected_avg}, Got={loss_correct.item()}")


def test_metric_aggregation():
    """测试评估指标聚合"""
    print("\n=== 测试指标聚合 ===")
    
    if not dist.is_initialized():
        print("❌ 分布式未初始化,跳过此测试")
        return
    
    rank = dist.get_rank()
    device = f'cuda:{rank}'
    
    # 模拟LossRecord的sum和count
    # GPU0处理8个样本,平均loss=0.5
    # GPU1处理8个样本,平均loss=0.3
    if rank == 0:
        local_sum = 0.5 * 8  # 4.0
        local_count = 8
    else:
        local_sum = 0.3 * 8  # 2.4
        local_count = 8
    
    print(f"Rank {rank}: local_sum={local_sum}, local_count={local_count}")
    
    # 聚合
    sum_tensor = torch.tensor(local_sum, device=device)
    count_tensor = torch.tensor(local_count, device=device)
    
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    
    global_avg = sum_tensor.item() / count_tensor.item()
    print(f"Rank {rank}: global_sum={sum_tensor.item()}, global_count={count_tensor.item()}, global_avg={global_avg}")
    
    # 验证
    expected_avg = (0.5 * 8 + 0.3 * 8) / 16
    if abs(global_avg - expected_avg) < 1e-6:
        print(f"✅ Rank {rank}: 指标聚合正确! Expected={expected_avg}, Got={global_avg}")
    else:
        print(f"❌ Rank {rank}: 指标聚合错误! Expected={expected_avg}, Got={global_avg}")


def main():
    """主测试函数"""
    # 检查是否在分布式环境中
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")
        distributed = True
    else:
        print("Running in non-distributed mode")
        distributed = False
    
    # 运行测试
    test_data_sampling(distributed=distributed)
    
    if distributed:
        test_loss_aggregation()
        test_metric_aggregation()
    
    # 清理
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    import os
    main()
