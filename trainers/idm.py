from typing import Dict, Any
import torch

from models import FNO2d
from .base_diffusion import BaseTrainer 


class IDMTrainer(BaseTrainer):

    def __init__(self, config: Dict[str, Any]):


        super().__init__(config)
        
        self.logger("DDPMTrainer 已初始化。")


    def build_model(self, config: Any) -> torch.nn.Module:
        model_opt = config['model']
        from models.idm.model.sr3_modules import diffusion, unet, edsr, mlp
        model = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
        encoder = edsr.EDSR(n_resblocks=16, n_feats=64, res_scale=1,scale=8, no_upsampling=False, rgb_range=1)

        imnet = mlp.MLP(in_dim=64+2, out_dim=3, hidden_list=[256, 256, 256, 256])

        netG = diffusion.GaussianDiffusion(
        encoder,
        imnet,
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='lploss',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'])
        netG.set_new_noise_schedule(model_opt['beta_schedule']['train'],self.device_id)
        netG.set_loss(self.device_id)
        return netG