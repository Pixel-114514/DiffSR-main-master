import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers.Basic import MLP
from .layers.MWT_Layers import MWT_CZ2d

class MWT_SuperResolution(nn.Module):
    """
    Multi-Wavelet Transform for 2D Navier-Stokes Super-Resolution
    Input: [B, H, W, C]  # 输入已包含物理场和位置编码
    Output: [B, H_out, W_out, C_out]
    """
    def __init__(
        self, 
        in_channels=3,
        out_channels=1,
        input_size=32,
        output_size=64,
        hidden_dim=128,
        k=6,
        n_layers=6,
        c=1,
        alpha=2,
        L=0,
        base='legendre',
        act='gelu'
    ):
        super(MWT_SuperResolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.k = k
        self.n_layers = n_layers
        self.c = c
        self.WMT_dim = c * k ** 2
        target = 2 ** (math.ceil(np.log2(input_size)))
        self.padding = [target - input_size, target - input_size]
        self.augmented_resolution = [target, target]
        self.preprocess = MLP(
            in_channels, 
            hidden_dim * 2, 
            self.WMT_dim, 
            n_layers=0,
            res=False,
            act=act
        )
        self.spectral_layers = nn.ModuleList([
            MWT_CZ2d(k=k, alpha=alpha, L=L, c=c, base=base) 
            for _ in range(n_layers)
        ])
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.WMT_dim, self.WMT_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.WMT_dim, self.WMT_dim, kernel_size=3, padding=1)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.WMT_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x):
        """
        前向传播
        输入: x [B, H, W, C]
        输出: [B, H_out, W_out, C_out]
        """
        B, H, W, C = x.shape
        assert C == self.in_channels, f"输入通道数不匹配: {C} vs {self.in_channels}"
        assert H == W == self.input_size, f"输入尺寸不匹配: {H}×{W} vs {self.input_size}"

        x_flat = x.reshape(B, H*W, C)

        # 预处理映射到MWT空间
        x_mwt = self.preprocess(x_flat)  # [B, H*W, WMT_dim]

        # 重塑为空间结构 [B, WMT_dim, H, W]
        x_mwt = x_mwt.permute(0, 2, 1).reshape(B, self.WMT_dim, H, W)

        if not all(item == 0 for item in self.padding):
            x_mwt = F.pad(x_mwt, [0, self.padding[1], 0, self.padding[0]])

        # 重塑为MWT输入格式 [B, H, W, c, k^2]
        x_mwt = x_mwt.reshape(B, self.WMT_dim, -1).permute(0, 2, 1).contiguous()
        x_mwt = x_mwt.reshape(B, *self.augmented_resolution, self.c, self.k**2)

        # MWT谱层处理
        for i in range(self.n_layers):
            x_mwt = self.spectral_layers[i](x_mwt)
            if i < self.n_layers - 1:
                x_mwt = F.gelu(x_mwt)

        x_mwt = x_mwt.reshape(B, -1, self.WMT_dim).permute(0, 2, 1).contiguous()
        x_mwt = x_mwt.reshape(B, self.WMT_dim, *self.augmented_resolution)

        if not all(item == 0 for item in self.padding):
            x_mwt = x_mwt[..., :-self.padding[0], :-self.padding[1]]

        x_up = self.upsample(x_mwt)  # [B, WMT_dim, 64, 64]

        #  [B, H_out*W_out, WMT_dim]
        H_out = W_out = self.output_size
        x_up = x_up.permute(0, 2, 3, 1).reshape(B, H_out*W_out, self.WMT_dim)
        out = self.output_projection(x_up)  # [B, H_out*W_out, out_channels]
        out = out.reshape(B, H_out, W_out, self.out_channels)

        return out