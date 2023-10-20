from typing import Dict
import torch
from torch.nn import Module

from ..parameters.parameter_config import configure


class UpdatePointsToKeyFrame(Module):
    def __init__(self):
        super().__init__()
        self.keyframe_pose = None

    def forward(self, batch: Dict[str, torch.Tensor], keyframe_pose_mat: torch.Tensor) -> Dict[str, torch.Tensor]:
        p3d2d = batch['p3d2d'].detach().clone()
        self.keyframe_pose = keyframe_pose_mat.detach().clone()
        p3d_h = torch.nn.functional.pad(p3d2d[:, :, :3], [0, 1], "constant", 1.0)
        keyframe_pose_mat_inv = torch.eye(4, dtype=keyframe_pose_mat.dtype, device=keyframe_pose_mat.device)
        keyframe_pose_mat_inv[:3, :3] = self.keyframe_pose[:3, :3].transpose(-1, -2)
        keyframe_pose_mat_inv[:3, 3:4] = -keyframe_pose_mat_inv[:3, :3] @ self.keyframe_pose[:3, 3:4]
        p3d_keyframe = torch.matmul(keyframe_pose_mat_inv, p3d_h.transpose(-2, -1)).transpose(-2, -1)[:, :, :3]
        p3d2d[:, :, :3] = p3d_keyframe
        return {'p3d2d_keyframe': p3d2d}

class ReversePoseToWorldFrame(Module):
    @configure
    def __init__(self, keyframe_period: int, train_outlier_model: bool= True):
        super().__init__()
        self.keyframe_pose = None
        self.keyframe_period = keyframe_period
        self.train_outlier_model=train_outlier_model

    def forward(self, batch: Dict[str, torch.Tensor], keyframe_pose_mat: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.keyframe_pose = keyframe_pose_mat.detach().clone()
        if(self.train_outlier_model):
            estimated_poses=batch['pose_matrix_estimate']
        else:
            estimated_poses=batch['pose_matrix_estimate_opencv']
        batch_size = estimated_poses.shape[0]
        pose_matrix_estimate_wf = torch.zeros((1, 4, 4), device=keyframe_pose_mat.device)
        for k in range(batch_size):
            camera_pose_wf = torch.matmul(self.keyframe_pose, estimated_poses[k])
            pose_matrix_estimate_wf = torch.cat((pose_matrix_estimate_wf, torch.unsqueeze(camera_pose_wf, 0)), dim=0)
            if not (k+1)%self.keyframe_period:
                self.keyframe_pose = camera_pose_wf
        local_frame_pose = keyframe_pose_mat.detach().clone()
        local_frame_pose[:3, :3] = local_frame_pose[:3, :3].transpose(-1, -2)
        local_frame_pose[:3, 3:4] = -local_frame_pose[:3, :3] @ local_frame_pose[:3, 3:4]
        pose_matrix_estimate_lf = local_frame_pose @ pose_matrix_estimate_wf
        return {'pose_matrix_estimate_wf': pose_matrix_estimate_wf[1:], 'pose_matrix_estimate_lf': pose_matrix_estimate_lf[1:]}
