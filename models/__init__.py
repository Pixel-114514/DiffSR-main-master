from .fno import FNO2d
from .unet import UNet2d
from .MWT import MWT_SuperResolution
from .swin_Transformer import SwinSR
from .sronet import SRNO
from .galerkin import Galerkin_Transformer


ALL_MODELS = {
    "FNO2d": FNO2d,
    "UNet2d": UNet2d,
    "MWT": MWT_SuperResolution,
    "Swin_Transformer": SwinSR,
    "SRNO": SRNO,
    "Galerkin_Transformer": Galerkin_Transformer,
}
