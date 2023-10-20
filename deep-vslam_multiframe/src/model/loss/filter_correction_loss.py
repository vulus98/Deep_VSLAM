from typing import Dict

from ...parameters.parameter_config import configure

import torch
from torch.jit import ScriptModule


class CorrectionL2loss(ScriptModule):
    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        correction = x['lstm_correction']
        # loss = torch.norm(correction)
        loss = torch.linalg.norm(correction, dim=-1).mean()
        return loss


class CorrectionMSEloss(ScriptModule):
    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        corr = x['corrected_kalman_prediction_lf'][2:,:]
        measurement = x['pose_vector_estimate_lf'][2:,:]
        loss = torch.linalg.norm(corr - measurement, dim=-1).mean()
        return loss

class SupervisedCorrectionMSEloss(ScriptModule):
    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        correction=x['corrected_pose_matrix_estimate_wf'][2:]
        reference=x['poses'][2:]
        loss = torch.linalg.matrix_norm(correction - reference).mean()
        return loss

class SupervisedCorrectionTrRotloss(ScriptModule):
    @torch.jit.script_method
    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
        correction=x['corrected_pose_matrix_estimate_wf'][2:]
        reference=x['poses'][2:]
        tr_loss = torch.linalg.norm(correction[:,:3,3] - reference[:,:3,3],dim=-1).mean()
        loss_calc_trace=torch.matmul(torch.transpose(reference[:,:3,:3],1,2),correction[:,:3,:3])
        loss_calc_trace=loss_calc_trace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        acos_input=torch.clip(0.5*(loss_calc_trace-1),-1+1e-7,1-1e-7)
        rot_loss=torch.arccos(acos_input).mean()
        return tr_loss+rot_loss

# class PoseConsistencyLoss(ScriptModule):
#     @torch.jit.script_method
#     def forward

# class SupervisedCorrectionTrRotloss(ScriptModule):
#     @configure
#     def __init__(self):
#         super().__init__()

#     @torch.jit.script_method
#     def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> torch.Tensor:
#         rotation_composition = x['poses'][:, :3, :3].transpose(-1, -2) @ x['corrected_pose_matrix_estimate_wf'][:, :3, :3]
#         rotation_arg = torch.clamp(0.5 * (torch.diagonal(rotation_composition, dim1=-2, dim2=-1).sum(-1) - 1.0), -1.0+1e-7, 1.0-1e-7)
#         rotation_loss = torch.acos(rotation_arg)
#         translation_arg = x['corrected_pose_matrix_estimate_wf'][:, :3, 3]-x['poses'][:, :3, 3]
#         translation_loss = torch.norm(translation_arg, dim=-1)
#         pose_loss = rotation_loss + translation_loss
#         return pose_loss.mean()
