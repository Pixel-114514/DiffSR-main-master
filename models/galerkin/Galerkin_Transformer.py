import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .layers.Basic import MLP, LinearAttention
from .layers.Embedding import unified_pos_embedding
from einops import rearrange


class Galerkin_Transformer_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_1a = nn.LayerNorm(hidden_dim)
        self.Attn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                    dropout=dropout, attn_type='galerkin')
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx), self.ln_1a(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Galerkin_Transformer(nn.Module):
    """
    Galerkin Transformer for 2D Navier-Stokes Super-Resolution
    Input:  [B, 32, 32, 1] - Low-resolution flow field
    Output: [B, 64, 64, 1] - High-resolution flow field
    """
    ## Factformer
    def __init__(
        self,
        fun_dim=1, 
        ref=8,            # 位置编码参考分辨率
        n_hidden=256,
        n_layers=6,
        n_heads=8,
        dropout=0.0,
        mlp_ratio=4,      # MLP 扩展倍数
        act='gelu',
        out_dim=1,
    ):
        super(Galerkin_Transformer, self).__init__()
        
        # 任务特定参数
        self.input_size = 32   # 输入分辨率
        self.output_size = 64  # 输出分辨率
        self.fun_dim = fun_dim
        self.ref = ref
        self.n_hidden = n_hidden
        
        # 位置编码：计算 64x64 网格到 8x8 参考网格的距离编码
        # 生成维度：[1, 4096, 64]（64 = 8x8）
        self.pos = unified_pos_embedding(
            shapelist=[self.output_size, self.output_size], 
            ref=ref,
            batchsize=1
        )
        
        # 输入维度：fun_dim(1) + ref^2(64) = 65
        self.preprocess = MLP(
            fun_dim + ref ** 2,  # 1 + 64 = 65
            n_hidden * 2,        # 512
            n_hidden,            # 256
            n_layers=0, 
            res=False, 
            act=act
        )

        self.blocks = nn.ModuleList([
            Galerkin_Transformer_block(
                num_heads=n_heads,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                last_layer=(_ == n_layers - 1)
            )
            for _ in range(n_layers)
        ])
        # 可学习偏置
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, 32, 32, 1] 低分辨率流场
            
        Returns:
            output: [B, 64, 64, 1] 高分辨率流场
        """
        B = x.shape[0]
        
        # 双线性插值上采样 [B,32,32,1] → [B,64,64,1]
        x = x.permute(0, 3, 1, 2)  # [B, 1, 32, 32]
        x = F.interpolate(
            x, 
            size=(self.output_size, self.output_size), 
            mode='bilinear', 
            align_corners=True
        )
        x = x.permute(0, 2, 3, 1)  # [B, 64, 64, 1]
        x = x[...,2:3]
        
        fx = rearrange(x, 'b h w c -> b (h w) c')
        
        # 拼接位置编码 [B, 4096, 1] + [1, 4096, 64] → [B, 4096, 65]
        pos_emb = self.pos.repeat(B, 1, 1)
        fx = torch.cat([fx, pos_emb], dim=-1)
        print('fx shape after pos emb:', fx.shape)
        
        fx = self.preprocess(fx)  # [B, 4096, 256]
        fx = fx + self.placeholder[None, None, :]
        
        for block in self.blocks:
            fx = block(fx)  # [B, 4096, 1]
        
        output = rearrange(fx, 'b (h w) c -> b h w c', h=self.output_size, w=self.output_size)
        
        return output
