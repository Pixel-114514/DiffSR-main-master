from .base import BaseTrainer

from .fno2d import FNO2DTrainer
from .ddpm import DDPMTrainer
from .idm import IDMTrainer
from .MWT import MWTTrainer
from .swin import SwinTransformerTrainer
from .sronet import SRNOTrainer
from .galerkin import GalerkinTrainer

TRAINER_DICT = {
    'FNO2d': FNO2DTrainer,
    'DDPM': DDPMTrainer,
    'IDM': IDMTrainer,
    'MWT': MWTTrainer,
    'Swin_Transformer': SwinTransformerTrainer,
    'SRNO': SRNOTrainer,
    'Galerkin_Transformer': GalerkinTrainer,
}
