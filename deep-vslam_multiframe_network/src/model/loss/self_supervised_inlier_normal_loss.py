from typing import Dict

import torch
import torch.nn as nn
from torch.jit import ScriptModule
from torch.nn.modules import loss

from ...parameters.parameter_config import configure

class SelfSupervisedInlierNormalLoss(ScriptModule):
    @configure
    def __init__(self):
        super().__init__()


    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        diff = torch.norm(x['weights_3d2d']-x['weights_3d2d_opencv'], dim=-1)
        # diff = self._loss(x['weights_3d2d'], x['weights_3d2d_opencv'])
        return diff.mean()