from typing import Dict

import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch.jit import ScriptModule
from torch.nn.modules import loss

from ...parameters.parameter_config import configure

class SelfSupervisedInlierLoss(ScriptModule):
    @configure
    def __init__(self):
        super().__init__()
        self._loss = BCELoss()
        # self._invs = InvSigmoid()

    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        # diff = torch.norm(x['weights_3d2d']-self._invs(x['weights_3d2d_opencv']), dim=-1)
        diff = self._loss(x['weights_3d2d'], x['weights_3d2d_opencv'])
        return diff

# class InvSigmoid(nn.Module):
#     def _init_(self):
#         super()._init_()
    
#     def forward(self, x):
#         assert torch.sum((x < 0)*(x > 1)) == 0
#         x = torch.where(x==0, x+1e-6, x)
#         x = torch.where(x==1, x-1e-6, x)
#         return torch.log(x/(1-x))
