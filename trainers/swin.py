from typing import Dict, Any
import torch

from models import SwinSR
from .base import BaseTrainer


class SwinTransformerTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        self.logger("SwinTransformerTrainer 已初始化。")

    def build_model(self, model_config) -> torch.nn.Module:
        self.model_config = model_config['model']
        
        self.logger(f"正在使用以下参数构建 SwinSR 模型: {(self.model_config)}")
        model = SwinSR(**self.model_config)
        
        return model
