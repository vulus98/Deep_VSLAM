from typing import Dict

import torch
from torch.jit import ScriptModule

from ...parameters.parameter_config import configure

class SupervisedInlierLoss(ScriptModule):
    __constants__ = ['weight_key']

    @configure
    def __init__(self, mode: str = '3d2d', threshold: float = None):
        super().__init__()
        if mode in {'2d2d', '3d2d'}:
            self.weight_key = 'weights_' + mode
        else:
            raise Exception('unknown mode:' + str(mode))
        self._threshold = threshold

    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        diff = ~torch.eq(x[self.weight_key]>self._threshold, x['inliers'])
        return (diff.float().sum(dim=1)/x[self.weight_key].shape[-1]).mean()
