import torch 
from torch.nn import Module
from typing import Dict

from ..parameters.parameter_config import configure

class CorrespondenceNormalizer(Module):
    
    @configure
    def __init__(self):
        super().__init__()
        self._intrinsic_matrix = None

    def forward(self, batch: Dict[str, torch.Tensor]):
        p3d2d = batch['p3d2d_keyframe'].detach().clone()
        batch_size = p3d2d.shape[0]

        mean3d = p3d2d[:, :, :3].mean(dim=-2, keepdim=True)
        p3d2d[:, :, :3] = p3d2d[:, :, :3]-mean3d
        scale3d = torch.div(1.0, p3d2d[:, :, :3].std(dim=-2, keepdim=True))
        p3d2d[:, :, :3] *= scale3d
        norm_mat_3d = torch.eye(4, device=p3d2d.device).repeat(batch_size, 1, 1)
        norm_mat_3d[:, :3, :3] = torch.diag_embed(scale3d.squeeze(), dim1=-2, dim2=-1)
        norm_mat_3d[:, :3, 3] = (-mean3d*scale3d).squeeze()

        # mean2d = p3d2d[:, :, 3:].mean(dim=-2, keepdim=True)
        # p3d2d[:, :, 3:] = p3d2d[:, :, 3:]-mean2d
        # scale2d = torch.div(1.0, p3d2d[:, :, 3:].std(dim=-2, keepdim=True))
        # p3d2d[:, :, 3:] *= scale2d
        # norm_mat_2d = torch.eye(3, device=p3d2d.device).repeat(batch_size, 1, 1)
        # norm_mat_2d[:, :2, :2] = torch.diag_embed(scale2d.squeeze(), dim1=-2, dim2=-1).inverse()
        # norm_mat_2d[:, :2, 2] = mean2d.squeeze()
        p2d = torch.nn.functional.pad(p3d2d[:, :, 3:], (0, 1), "constant", 1)
        p3d2d[:, :, 3:] = (self._intrinsic_matrix.inverse() @ p2d.transpose(-1, -2)).transpose(-1, -2)[:, :, :2]

        return {'p3d2d_n': p3d2d, 
                'norm_mat_3d': norm_mat_3d,
                'denorm_mat_2d': self._intrinsic_matrix.inverse()} # @ norm_mat_2d

    def set_intrinsic_matrix(self, intrinsic_matrix: torch.Tensor):
        del self._intrinsic_matrix
        self.register_buffer('_intrinsic_matrix', intrinsic_matrix)
