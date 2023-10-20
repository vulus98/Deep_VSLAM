from typing import Dict

import torch
from torch.jit import ScriptModule

from ...parameters.parameter_config import configure

class InlierLoss(ScriptModule):
    __constants__ = ['weight_key', '_eps']

    @configure
    def __init__(self, device, mode: str = '3d2d', eps: float = 1e-5):
        super().__init__()
        if mode in {'2d2d', '3d2d'}:
            self.weight_key = 'weights_' + mode
        else:
            raise Exception('unknown mode:' + str(mode))
        self._eps = eps
        self._zero = torch.tensor(0.0, device=device)

    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        num_points = x[self.weight_key].shape[-1]
        # weights = torch.sigmoid(10*(x[self.weight_key]-0.5)).sum(-1) + self._eps
        weights = x[self.weight_key].sum(-1) + self._eps
        # print(num_points * torch.div(1.0, weights))
        return num_points * torch.div(1.0, weights).mean()
        # weights = x[self.weight_key].sum(-1)
        # weights = torch.sigmoid(20*(x[self.weight_key]-0.5)).sum(-1)
        # return (torch.exp(torch.max(self._zero, 25.0-weights))-1.0).mean()
        # return 20.0-weights.mean()/num_points