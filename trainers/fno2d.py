from typing import Dict, Any
import torch

from models import FNO2d
from .base import BaseTrainer 


class FNO2DTrainer(BaseTrainer):

    def __init__(self, config: Dict[str, Any]):


        super().__init__(config)
        
        self.logger("FNO2DTrainer 已初始化。")


    def build_model(self, model_config: Any) -> torch.nn.Module:
        
        self.model_config = model_config['model']


        self.logger(f"正在使用以下参数构建 FNO2d 模型: {(self.model_config)}")
        model = FNO2d(**self.model_config)
        
        return model