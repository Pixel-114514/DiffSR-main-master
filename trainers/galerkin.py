from typing import Dict, Any
import torch

from models import Galerkin_Transformer
from .base import BaseTrainer


class GalerkinTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger("GalerkinTrainer 已初始化。")
    
    def build_model(self, model_config: Any) -> torch.nn.Module:      
        self.model_config = model_config['model']
        
        self.logger(f"正在使用以下参数构建 Galerkin_Transformer 模型: {(self.model_config)}")
        model = Galerkin_Transformer(**self.model_config)
        
        return model
