from typing import Dict, Any
import torch

from models import SRNO
from .base import BaseTrainer


class SRNOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        self.logger("SRNOTrainer 已初始化。")
        
    def build_model(self, model_config: Any) -> torch.nn.Module:
        self.model_config = model_config['model']
        
        self.logger(f"正在使用以下参数构建 SRNO 模型: {(self.model_config)}")
        model = SRNO(**self.model_config)
        
        return model