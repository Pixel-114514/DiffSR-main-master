import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer='LayerNorm'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinSR(nn.Module):
    """
    Swin Transformer for 2D Navier-Stokes Super-Resolution
    Input: [B, H, W, C] where C=[x, y, a(x,y)]
    Output: [B, H_out, W_out, 1]
    """
    
    def __init__(
        self,
        in_channels=3,          # 输入通道数: [x, y, a(x,y)]
        out_channels=1,         # 输出通道数: a(x,y)
        input_size=32,          # 输入分辨率 32×32
        output_size=64,         # 输出分辨率 64×64
        n_hidden=96,            # 隐藏层维度
        window_size=4,          # 窗口大小
        n_layers=4,             # Swin层数
        depth=2,                # 每层的Block数量
        num_heads=3,            # 注意力头数
        mlp_ratio=4.,           # MLP隐藏层倍数
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer='LayerNorm',
        use_residual=True       # 是否使用残差连接
    ):
        super().__init__()
        self.__name__ = 'SwinSR'
        
        # 基础参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.window_size = window_size
        self.n_layers = n_layers
        self.use_residual = use_residual
        
        if isinstance(qk_scale, str) and qk_scale == 'None':
            qk_scale = None
        norm_layer = nn.LayerNorm if norm_layer == 'LayerNorm' else nn.BatchNorm2d
        
        # 参数验证
        assert input_size % window_size == 0, \
            f"input_size ({input_size}) must be divisible by window_size ({window_size})"
        
        # 1. 预处理层：剔除位置编码，只保留物理场 a(x,y)
        # 输入 [B, H, W, 3] → 取第3个通道 → MLP升维到 n_hidden
        self.preprocess = nn.Sequential(
            nn.Linear(1, n_hidden * 2),  # 单通道物理场升维
            nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden)
        )
        
        # 2. 可学习的位置嵌入（替代输入中的绝对位置编码）
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size * input_size, n_hidden))
        trunc_normal_(self.pos_embed, std=.02)
        
        # 3. Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 4. Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers * depth)]
        
        # 5. Swin Transformer层
        self.layers = nn.ModuleList()
        for i_layer in range(n_layers):
            layer = BasicLayer(
                dim=n_hidden,
                input_resolution=(input_size, input_size),
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i_layer * depth:(i_layer + 1) * depth],
                norm_layer=norm_layer
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(n_hidden)
        
        # 6. 上采样模块 (32×32 → 64×64)
        # 使用插值 + 卷积细化的方式
        self.upsample = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4),  # 特征扩展
            nn.GELU(),
            nn.Linear(n_hidden * 4, n_hidden)
        )
        
        # 卷积细化层（在空间维度上操作）
        self.conv_refine = nn.Sequential(
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden // 2, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # 7. 输出投影层
        self.output_proj = nn.Sequential(
            nn.Conv2d(n_hidden // 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # 8. 残差连接的上采样（用于skip connection）
        if use_residual:
            self.residual_upsample = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
            )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] 其中 C=[x, y, a(x,y)]
        
        Returns:
            out: [B, H_out, W_out, 1]
        """
        B, H, W, C = x.shape
        assert C == self.in_channels, f"输入通道数不匹配: {C} vs {self.in_channels}"
        assert H == W == self.input_size, f"输入尺寸不匹配: {H}×{W} vs {self.input_size}"
        
        # Step 1: 提取物理场 a(x,y)（第3个通道，索引为2）
        physical_field = x[..., 2:3]  # [B, H, W, 1]
        
        # 保存用于残差连接
        if self.use_residual:
            # 转换为 [B, 1, H, W] 格式
            residual_input = physical_field.permute(0, 3, 1, 2).contiguous()
        
        # Step 2: 展平空间维度 [B, H, W, 1] → [B, H*W, 1]
        x_flat = physical_field.reshape(B, H * W, 1)
        
        # Step 3: 预处理 - MLP升维
        x = self.preprocess(x_flat)  # [B, H*W, n_hidden]
        
        # Step 4: 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Step 5: Swin Transformer处理
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)  # [B, H*W, n_hidden]
        
        # Step 6: 上采样特征
        x = self.upsample(x)  # [B, H*W, n_hidden]
        
        # Step 7: 重塑为空间格式并进行双线性上采样
        x = x.reshape(B, H, W, self.n_hidden).permute(0, 3, 1, 2)  # [B, n_hidden, H, W]
        
        # 双线性插值上采样 32×32 → 64×64
        x = F.interpolate(x, size=(self.output_size, self.output_size), 
                         mode='bilinear', align_corners=True)  # [B, n_hidden, 64, 64]
        
        # Step 8: 卷积细化
        x = self.conv_refine(x)  # [B, n_hidden//2, 64, 64]
        
        # Step 9: 输出投影
        out = self.output_proj(x)  # [B, 1, 64, 64]
        
        # Step 10: 残差连接
        if self.use_residual:
            # 上采样残差输入
            residual = F.interpolate(residual_input, size=(self.output_size, self.output_size),
                                    mode='bilinear', align_corners=True)  # [B, 1, 64, 64]
            residual = self.residual_upsample(residual)  # [B, 1, 64, 64]
            out = out + residual
        
        # Step 11: 转换为 [B, H, W, C] 格式
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, 64, 64, 1]
        
        return out