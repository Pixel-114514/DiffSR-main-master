from typing import Dict, Any
import torch

from models import MWT_SuperResolution
from .base import BaseTrainer


class MWTTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.logger("MWTTrainer 已初始化。")

    def build_model(self, model_config: Any) -> torch.nn.Module:
        self.model_config = model_config['model']
        
        self.logger(f"正在使用以下参数构建 MWT_SuperResolution 模型: {(self.model_config)}")
        model = MWT_SuperResolution(**self.model_config)
        return model
