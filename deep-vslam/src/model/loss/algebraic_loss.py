from typing import Dict

import torch
from torch.jit import ScriptModule

from ...parameters.parameter_config import configure

class PnPLastSVLoss(ScriptModule):
    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        return x['singular_values_vand'][:, -1].mean()


class PnPLastSVsRatioLoss(ScriptModule):
    __constants__ = ['_eps']

    @configure
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        last_sv = x['singular_values_vand'][:, -1]
        second_last_sv = x['singular_values_vand'][:, -2]
        ratio = last_sv / (second_last_sv + self._eps)
        return ratio.mean()