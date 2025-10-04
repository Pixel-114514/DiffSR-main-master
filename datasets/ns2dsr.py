import os.path as osp
import scipy.io as sio
import numpy as np

from h5py import File

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# 假设你的 normalizer 在 utils 文件夹下
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class NavierStokes2DDataset:
    def __init__(self, data_path, 
                 downsample_factor=4,
                 train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 normalize=True, normalizer_type='PGN', distributed=False,**kwargs):
        
        # 为了兼容性，我们将 downsample_factor 赋给内部变量 sample_factor
        self.sample_factor = downsample_factor 

        self.load_data(data_path=data_path, 
                       train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, 
                       sample_factor=self.sample_factor,
                       normalize=normalize, normalizer_type=normalizer_type)

        # --- DDP兼容性修改 ---
        # 根据distributed参数决定是否创建sampler
        # 单卡训练时, train_sampler为None
        # DDP训练时, 创建DistributedSampler实例
        train_sampler = DistributedSampler(self.train_dataset) if distributed else None
        valid_sampler = None
        test_sampler = None
        # 当sampler不为None时(DDP模式)，shuffle必须为False，因为sampler会处理打乱逻辑
        # 当sampler为None时(单卡模式), shuffle=(train_sampler is None) 的结果为True，保持原有行为
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=train_batchsize, 
                                       shuffle=(train_sampler is None), 
                                       sampler=train_sampler)
        self.valid_loader = DataLoader(self.valid_dataset, 
                                       batch_size=eval_batchsize, 
                                       shuffle=False, 
                                       sampler=valid_sampler)
        self.test_loader = DataLoader(self.test_dataset, 
                                      batch_size=eval_batchsize, 
                                      shuffle=False, 
                                      sampler=test_sampler)

    def load_data(self, data_path, sample_factor,
                  train_ratio, valid_ratio, test_ratio, 
                  normalize, normalizer_type):
        
        # 修改缓存文件名，以反映新的数据格式 (无坐标网格)
        process_path = data_path.split('.')[0] + f'_sr_upsampled_d{sample_factor}_noch.pt'

        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            # 注意: x_normalizer 已被移除，因为输入和输出共享 y_normalizer
            train_x, train_y, valid_x, valid_y, test_x, test_y, y_normalizer = torch.load(process_path,weights_only=False)
            if y_normalizer is not None:
                x_normalizer = y_normalizer
            else:
                x_normalizer = None
        else:
            print('Processing data for Super-Resolution (pure field, with upsampling)...')
            try:
                raw_data = sio.loadmat(data_path)
                data = torch.tensor(raw_data['u'], dtype=torch.float32)
            except:
                raw_data = File(data_path, 'r')
                data = torch.tensor(np.transpose(raw_data['u'], (3, 1, 2, 0)), dtype=torch.float32)
            
            data_size = data.shape[0]
            train_idx = int(data_size * train_ratio)
            valid_idx = int(data_size * (train_ratio + valid_ratio))
            test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
            
            train_x, train_y, x_normalizer, y_normalizer = self.pre_process(data[:train_idx], mode='train', normalize=normalize, normalizer_type=normalizer_type, downsample_factor=sample_factor)
            valid_x, valid_y = self.pre_process(data[train_idx:valid_idx], mode='valid', normalize=normalize, x_normalizer=x_normalizer, y_normalizer=y_normalizer, downsample_factor=sample_factor)
            test_x, test_y = self.pre_process(data[valid_idx:test_idx], mode='test', normalize=normalize, x_normalizer=x_normalizer, y_normalizer=y_normalizer, downsample_factor=sample_factor)
            
            print('Saving data...')
            # 只保存 y_normalizer，因为 x_normalizer 和它完全一样
            torch.save((train_x, train_y, valid_x, valid_y, test_x, test_y, y_normalizer), process_path)
            print('Data processed and saved to', process_path)

        self.train_dataset = NavierStokes2DBase(train_x, train_y, mode='train', y_normalizer=y_normalizer)
        self.valid_dataset = NavierStokes2DBase(valid_x, valid_y, mode='valid', y_normalizer=y_normalizer)
        self.test_dataset = NavierStokes2DBase(test_x, test_y, mode='test', y_normalizer=y_normalizer)

    def pre_process(self, data, mode='train', normalize=False, 
                    normalizer_type='PGN', x_normalizer=None, y_normalizer=None,
                    downsample_factor=4,
                    **kwargs):
        
        # 1. 将所有时间帧视为独立样本，作为高分辨率目标 (y)
        N, H_hr, W_hr, T = data.shape
        data_permuted = data.permute(0, 3, 1, 2)  # -> (N, T, H_hr, W_hr)
        # 直接得到 (B, C, H, W) 格式, C=1
        y = data_permuted.reshape(N * T, 1, H_hr, W_hr) 
        B, C, H, W = y.shape
        
        # 2. 对高分辨率目标数据 (y) 进行归一化
        if normalize:
            # Reshape for normalizer: (B, H*W, C)
            y_reshaped = y.permute(0, 2, 3, 1).reshape(B, -1, C) 
            if mode == 'train':
                if normalizer_type == 'PGN':
                    shared_normalizer = UnitGaussianNormalizer(y_reshaped)
                else:
                    shared_normalizer = GaussianNormalizer(y_reshaped)
                x_normalizer = shared_normalizer
                y_normalizer = shared_normalizer
            
            # Encode and reshape back to (B, C, H, W)
            y = y_normalizer.encode(y_reshaped).reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_normalizer = None
            y_normalizer = None

        # 3. 创建下采样->上采样的低分辨率输入 (x)
        x_lr_upsampled = F.interpolate(
            y, # 输入已经是 (B, C, H, W)
            scale_factor=1.0/downsample_factor, 
            mode='bicubic', 
            align_corners=False,
            recompute_scale_factor=True
        )
        x = F.interpolate(
            x_lr_upsampled, 
            size=(H, W), 
            mode='bicubic', 
            align_corners=False
        )
        
        print(f'Processed SR data for {mode}:')
        print('  Final input x shape:', x.shape)
        print('  Final target y shape:', y.shape)
        
        if mode == 'train':
            return x, y, x_normalizer, y_normalizer
        else:
            return x, y


class NavierStokes2DBase(Dataset):
    """此类现在接收并返回 (B, C, H, W) 格式的张量"""

    def __init__(self, x, y, mode='train', y_normalizer=None, **kwargs):
        self.mode = mode
        self.x = x
        self.y = y 
        self.y_normalizer = y_normalizer

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # 返回字典，其中的张量已经是 (C, H, W) 格式
        return {'HR': self.y[idx], 'SR': self.x[idx], 'Index': idx}