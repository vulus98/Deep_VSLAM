from typing import Dict

import torch
from torch.jit import ScriptModule

from ...parameters.parameter_config import configure

class SupervisedPoseLoss(ScriptModule):
    @configure
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        rotation_composition = x['poses'][:, :3, :3].transpose(-1, -2) @ x['pose_matrix_estimate_wf'][:, :3, :3]
        rotation_arg = torch.clamp(0.5 * (torch.diagonal(rotation_composition, dim1=-2, dim2=-1).sum(-1) - 1.0), -1.0+1e-7, 1.0-1e-7)
        rotation_loss = torch.acos(rotation_arg)
        translation_arg = x['pose_matrix_estimate_wf'][:, :3, 3]-x['poses'][:, :3, 3]
        translation_loss = torch.norm(translation_arg, dim=-1)
        pose_loss = rotation_loss + translation_loss
        return pose_loss.mean()


class AuxiliarySupervisedPoseLoss(ScriptModule):
    @configure
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        sum_loss=torch.tensor(0,device=x['pose_matrix_estimate_wf'].device,dtype=torch.float)
        for i,gt_pose in enumerate(x['poses']):
            ground_truth=gt_pose.repeat(1,1,5)
            rotation_composition = ground_truth[:, :3, :3].transpose(-1, -2) @ x['pose_matrix_estimate_wf'][:, :3, :3]
            rotation_arg = torch.clamp(0.5 * (torch.diagonal(rotation_composition, dim1=-2, dim2=-1).sum(-1) - 1.0), -1.0+1e-7, 1.0-1e-7)
            rotation_loss = torch.acos(rotation_arg)
            translation_arg = x['pose_matrix_estimate_wf'][:, :3, 3]-ground_truth[:, :3, 3]
            translation_loss = torch.norm(translation_arg, dim=-1)
            pose_loss = rotation_loss + translation_loss
            sum_loss+=pose_loss.mean()
        return sum_loss