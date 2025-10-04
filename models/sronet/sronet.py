import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from .galerkin import simple_attn
from .edsr import EDSR
from  .utils import make_coord


class SRNO(nn.Module):
    """
    Super-Resolution Neural Operator for Navier-Stokes equations
    
    Args:
        encoder_config: EDSR 编码器的配置字典
        input_channels: 输入通道数 (不含坐标，默认1表示物理场)
        output_channels: 输出通道数 (默认1)
        use_coord_input: 是否使用输入中的坐标通道 (默认True)
        width: Galerkin Transformer 的宽度
        blocks: Galerkin Transformer 的块数
    """

    def __init__(self, encoder_config, 
                 input_channels=1, output_channels=1, 
                 use_coord_input=True,
                 width=256, blocks=16,upscale=2):
        super().__init__()
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_coord_input = use_coord_input
        self.upscale = upscale
        self.encoder = EDSR(**encoder_config)
        
        encoder_feat_dim = encoder_config.get('n_feats', 64)
        
        # 特征融合层，输入通道: (encoder_feat + 2相对坐标) * 4个邻域 + 2个cell尺寸
        fusion_input_channels = (encoder_feat_dim + 2) * 4 + 2
        self.conv00 = nn.Conv2d(fusion_input_channels, self.width, 1)

        # Galerkin Transformer
        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, self.output_channels, 1)
        
    def gen_feat(self, inp):
        """
        生成特征图
        Args:
            inp: [B, H, W, C] 其中 C = [x, y, a(x,y)] 或 [a(x,y)]
        Returns:
            feat: [B, encoder_feat_dim, H, W]
        """
        if inp.dim() == 4 and inp.shape[-1] <= 4:
            inp = inp.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        
        if self.use_coord_input and inp.shape[1] > self.input_channels:
            inp_physics = inp[:, 2:, :, :]  # 取 a(x,y) 部分
        else:
            inp_physics = inp
        
        self.inp = inp_physics
        self.feat = self.encoder(inp_physics)
        return self.feat
        
    def query_field(self, coord, cell):
        """
        查询任意坐标的物理场值
        Args:
            coord: [B, H', W', 2] 查询坐标 (归一化到[-1,1])
            cell: [B, 2] cell 尺寸
        Returns:
            ret: [B, output_channels, H', W']
        """
        feat = self.feat  # [B, feat_dim, H, W]
        
        # 生成低分辨率特征图的坐标网格
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # 计算邻域采样的偏移量
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        
        # 对4个邻域进行采样
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), 
                                     mode='nearest', align_corners=False)
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), 
                                         mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # 计算面积权重 (双线性插值的思想)
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
        
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
         
        # 拼接所有特征: [相对坐标×4, 加权特征×4, cell尺寸]
        grid = torch.cat([
            *rel_coords, 
            *feat_s,
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])
        ], dim=1)

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)
        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        
        # 残差连接
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                                  padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp):
        """
        前向传播
        Args:
            inp: [B, H, W, C] 输入数据 (包含坐标和物理场)
            coord: [B, H', W', 2] 查询坐标
            cell: [B, 2] cell 尺寸
        Returns:
            output: [B, output_channels, H', W'] 预测的物理场
        """
        # inp: [B,H,W,C]
        B, H, W, C = inp.shape
        H_out, W_out = H * self.upscale, W * self.upscale
        coord = make_coord((H_out, W_out)).to(inp.device)
        coord = coord.unsqueeze(0).expand(B, -1, -1)
        coord = coord.view(B, H_out, W_out, 2)
        cell = torch.ones(B, 2).to(inp.device)
        cell[:, 0] *= 2 / H_out
        cell[:, 1] *= 2 / W_out
        self.gen_feat(inp)
        out = self.query_field(coord, cell)
        return out