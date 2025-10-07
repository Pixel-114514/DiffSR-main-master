import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_flow_field(sr_output, hr_gt, save_path, title='', sample_factor=2):
    """
    可视化流场结果
    
    Args:
        lr_input: 低分辨率输入 [1, H, W, C](可不传入)
        sr_output: 超分辨率输出 [1, H, W, C]  
        hr_gt: 高分辨率真值 [1, H, W, C]
        save_path: 保存路径
        title: 图像标题
        sample_factor: 下采样因子，默认为2
    """
    # 1. 将Tensor移动到CPU并转换为Numpy数组
    sr = sr_output.squeeze(0).squeeze(-1).cpu().numpy()
    hr = hr_gt.squeeze(0).squeeze(-1).cpu().numpy()

    lr = hr[::sample_factor, ::sample_factor]
    # 误差图
    error = np.abs(sr - hr)

    # 2. 准备绘图
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    cmap = 'viridis'

    # 3. 共用 HR 的范围
    vmin, vmax = hr.min(), hr.max()

    # 绘制低分辨率输入
    im_lr = axes[0].imshow(lr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Low-Res Input\n({lr.shape[0]}x{lr.shape[1]})')
    axes[0].axis('off')

    # 绘制模型超分输出
    im_sr = axes[1].imshow(sr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Super-Res Output\n({sr.shape[0]}x{sr.shape[1]})')
    axes[1].axis('off')

    # 绘制高分辨率真值
    im_hr = axes[2].imshow(hr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f'High-Res Ground Truth\n({hr.shape[0]}x{hr.shape[1]})')
    axes[2].axis('off')

    # 绘制误差
    im_err = axes[3].imshow(error, cmap='hot', vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    axes[3].set_title('Absolute Error (SR - HR)')
    axes[3].axis('off')

    # 4. 添加共用纵向 colorbar（放在右边）
    for ax, mappable in [(axes[0], im_lr), (axes[1], im_sr), (axes[2], im_hr)]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        fig.colorbar(mappable, cax=cax)

    # 为误差图单独加一个 colorbar
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="4%", pad=0.04)
    fig.colorbar(im_err, cax=cax, label="Error Value")

    # 5. 设置总标题
    fig.suptitle(title, fontsize=16)

    # 6. 保存图像
    fig.savefig(save_path, bbox_inches='tight' ,dpi=150)
    plt.close(fig)