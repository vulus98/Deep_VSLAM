from typing import Dict

import torch
from torch.nn import Module

from ...parameters.parameter_config import configure

def batch_invert_pose(pose: torch.Tensor):
    inv = torch.eye(4, dtype=pose.dtype, device=pose.device).repeat((pose.shape[0], 1, 1))
    inv[:, :3, :3] = pose[:, :3, :3].transpose(-1, -2)
    inv[:, :3, 3:4] = -inv[:, :3, :3] @ pose[:, :3, 3:4]
    return inv


def batch_projection_error(p3d2d: torch.Tensor, p) -> torch.Tensor:
    p3d_h = torch.nn.functional.pad(p3d2d[:, :, :3], [0, 1], "constant", 1.0)
    p2d_proj = (p @ p3d_h.transpose(-2, -1)).transpose(-2, -1)
    z = p2d_proj[:, :, 2:3]
    scale = torch.where(torch.abs(z) > 1e-4, z, 1e-4*torch.ones_like(z))
    p2d_proj = p2d_proj[:, :, :2] / scale
    p2d = p3d2d[:, :, 3:]
    return torch.norm(p2d_proj-p2d, dim=-1)


class ReprojectionLoss(Module):
    @configure
    def __init__(self):
        super().__init__()
        self._intrinsic_matrix = None

    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = self._intrinsic_matrix @ batch_invert_pose(x['pose_matrix_estimate'])
        # weighted_err = (batch_projection_error(x['p3d2d_keyframe'], p) * x['weights_3d2d']).mean(-1)/x['weights_3d2d'].sum(-1)
        weighted_err = batch_projection_error(x['p3d2d_keyframe'], p) * x['weights_3d2d']
        return weighted_err.mean()

    def set_intrinsic_matrix(self, intrinsic_matrix: torch.Tensor):
        del self._intrinsic_matrix
        self.register_buffer('_intrinsic_matrix', intrinsic_matrix)

