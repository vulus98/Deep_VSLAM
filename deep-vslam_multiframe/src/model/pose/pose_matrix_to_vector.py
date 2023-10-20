from typing import Dict
from numpy.core.numeric import zeros_like
import torch
from torch.nn import Module
import numpy as np

from ...parameters.parameter_config import configure


class PoseMatToVec(Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose_mat = batch['pose_matrix_estimate_lf'].detach().clone()
        batch_size = pose_mat.shape[0]

        pose_vec = torch.zeros((batch_size, 6), device=pose_mat.device)
        # pose_vec[:, 3:] = torch.from_numpy(R.from_matrix(pose_mat[:, :3, :3].detach().clone().cpu()).as_euler('xyz').astype('float32')).to(pose_mat.device)
        # pose_vec[:, :3] = pose_mat[:, :3, 3]

        R31 = pose_mat[:,2,0]
        R32 = pose_mat[:,2,1]
        R33 = pose_mat[:,2,2]
        R21 = pose_mat[:,1,0]
        R11 = pose_mat[:,0,0]
        R12 = pose_mat[:,0,1]
        R13 = pose_mat[:,0,2]

        pose_vec[:, :3] = pose_mat[:, :3, 3]

        # one = torch.tensor(1.0, device = pose_vec.device)
        # minus_one = torch.tensor(-1.0, device = pose_vec.device)

        mask = ~(R31 >= 1.0) & ~(R31 <= -1.0)
        mask1 = R31 <= -1.0
        mask2 = R31 >= 1.0
        # mask = ~torch.isclose(R31, one) & ~torch.isclose(R31, minus_one) & ~(R31 >= 1.0) & ~(R31 <= -1.0)
        # mask1 = torch.isclose(R31, minus_one) | (R31 <= -1.0)
        # mask2 = torch.isclose(R31, one) | (R31 >= 1.0)

        pose_vec[mask, 4] = - torch.asin(R31[mask])        
        pose_vec[mask, 3] = torch.atan2(R32[mask]/torch.cos(pose_vec[mask, 4]),R33[mask]/torch.cos(pose_vec[mask, 4]))
        pose_vec[mask, 5] = torch.atan2(R21[mask]/torch.cos(pose_vec[mask, 4]),R11[mask]/torch.cos(pose_vec[mask, 4]))
        # pose_vec[:, 4] = np.pi - pose_vec[:, 4]
        # psi2 = torch.atan2(R32/torch.cos(pose_vec[:, 4]2),R33/torch.cos(pose_vec[:, 4]2))
        # phi2 = torch.atan2(R21/torch.cos(pose_vec[:, 4]2),R11/torch.cos(pose_vec[:, 4]2))

        if mask.float().sum() < batch_size:
            print('****************GIMBAL LOCK********************')
            pose_vec[~mask, 5] = torch.zeros_like(pose_vec[~mask, 5])

            pose_vec[mask1, 4] = torch.tensor(np.pi/2, device = pose_vec.device)
            pose_vec[mask1, 3] = torch.atan2(R12[mask1], R13[mask1])

            pose_vec[mask2, 4] = torch.tensor(-np.pi/2, device = pose_vec.device)
            pose_vec[mask2, 3] = torch.atan2(-R12[mask2], -R13[mask2])

        return {'pose_vector_estimate_lf': pose_vec}


class PoseVecToMat(Module):
    @configure
    def __init__(self, device,train_correction_model: bool = False, supervised_training: bool = False):
        super().__init__()
        self._device = device
        self._pose_matrix_keyframe = None
        self._pose_matrix_keyframe_kalman_estimate = None
        self._pose_matrix_keyframe_kalman_prediction = None
        self.train_correction_model=train_correction_model
        self.supervised_training=supervised_training

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # pose_vec = batch['filtered_pose_vector_prediction_lf'].detach().clone()
        # batch_size = pose_vec.shape[0]
        
        # cos_alpha = torch.cos(pose_vec[:, 3])
        # sin_alpha = torch.sin(pose_vec[:, 3])
        # cos_beta = torch.cos(pose_vec[:, 4])
        # sin_beta = torch.sin(pose_vec[:, 4])
        # cos_gamma = torch.cos(pose_vec[:, 5])
        # sin_gamma = torch.sin(pose_vec[:, 5])
        
        # pose_mat = torch.eye(4, dtype = pose_vec.dtype, device = pose_vec.device).repeat(batch_size, 1, 1)
        # pose_mat[:, 0, 0] = cos_beta*cos_gamma
        # pose_mat[:, 1, 0] = cos_beta*sin_gamma
        # pose_mat[:, 2, 0] = -sin_beta
        # pose_mat[:, 0, 1] = sin_alpha*sin_beta*cos_gamma-cos_alpha*sin_gamma
        # pose_mat[:, 1, 1] = sin_alpha*sin_beta*sin_gamma+cos_alpha*cos_gamma
        # pose_mat[:, 2, 1] = sin_alpha*cos_beta
        # pose_mat[:, 0, 2] = cos_alpha*sin_beta*cos_gamma+sin_alpha*sin_gamma
        # pose_mat[:, 1, 2] = cos_alpha*sin_beta*sin_gamma-sin_alpha*cos_gamma
        # pose_mat[:, 2, 2] = cos_alpha*cos_beta

        # pose_mat[:, :3, 3] = pose_vec[:, :3]

        # pose_wf_kalman_pred = torch.matmul(self._pose_matrix_keyframe_kalman_prediction, pose_mat)

        # self._pose_matrix_keyframe_kalman_prediction = pose_wf_kalman_pred[-1]



        # pose_vec = batch['filtered_pose_vector_estimate_lf'].detach().clone()
        # batch_size = pose_vec.shape[0]
        
        # cos_alpha = torch.cos(pose_vec[:, 3])
        # sin_alpha = torch.sin(pose_vec[:, 3])
        # cos_beta = torch.cos(pose_vec[:, 4])
        # sin_beta = torch.sin(pose_vec[:, 4])
        # cos_gamma = torch.cos(pose_vec[:, 5])
        # sin_gamma = torch.sin(pose_vec[:, 5])
        
        # pose_mat = torch.eye(4, dtype = pose_vec.dtype, device = pose_vec.device).repeat(batch_size, 1, 1)
        # pose_mat[:, 0, 0] = cos_beta*cos_gamma
        # pose_mat[:, 1, 0] = cos_beta*sin_gamma
        # pose_mat[:, 2, 0] = -sin_beta
        # pose_mat[:, 0, 1] = sin_alpha*sin_beta*cos_gamma-cos_alpha*sin_gamma
        # pose_mat[:, 1, 1] = sin_alpha*sin_beta*sin_gamma+cos_alpha*cos_gamma
        # pose_mat[:, 2, 1] = sin_alpha*cos_beta
        # pose_mat[:, 0, 2] = cos_alpha*sin_beta*cos_gamma+sin_alpha*sin_gamma
        # pose_mat[:, 1, 2] = cos_alpha*sin_beta*sin_gamma-sin_alpha*cos_gamma
        # pose_mat[:, 2, 2] = cos_alpha*cos_beta

        # pose_mat[:, :3, 3] = pose_vec[:, :3]
        # # self._pose_matrix_keyframe = torch.eye(4, device=self._device)
        # # self._pose_matrix_keyframe_kalman_estimate = torch.eye(4, device=self._device)
        # # self._pose_matrix_keyframe_kalman_prediction = torch.eye(4, device=self._device)

        # #self._pose_matrix_keyframe=self._pose_matrix_keyframe.detach().clone()

        # pose_wf_kalman_est = torch.matmul(self._pose_matrix_keyframe_kalman_estimate, pose_mat)

        # self._pose_matrix_keyframe_kalman_estimate = pose_wf_kalman_est[-1]



        #pose_vec = batch['corrected_kalman_prediction_lf'].detach().clone()
        pose_vec = batch['corrected_kalman_prediction_lf']
        batch_size = pose_vec.shape[0]
        
        cos_alpha = torch.cos(pose_vec[:, 3])
        sin_alpha = torch.sin(pose_vec[:, 3])
        cos_beta = torch.cos(pose_vec[:, 4])
        sin_beta = torch.sin(pose_vec[:, 4])
        cos_gamma = torch.cos(pose_vec[:, 5])
        sin_gamma = torch.sin(pose_vec[:, 5])
        
        pose_mat = torch.eye(4, dtype = pose_vec.dtype, device = pose_vec.device).repeat(batch_size, 1, 1)
        pose_mat[:, 0, 0] = cos_beta*cos_gamma
        pose_mat[:, 1, 0] = cos_beta*sin_gamma
        pose_mat[:, 2, 0] = -sin_beta
        pose_mat[:, 0, 1] = sin_alpha*sin_beta*cos_gamma-cos_alpha*sin_gamma
        pose_mat[:, 1, 1] = sin_alpha*sin_beta*sin_gamma+cos_alpha*cos_gamma
        pose_mat[:, 2, 1] = sin_alpha*cos_beta
        pose_mat[:, 0, 2] = cos_alpha*sin_beta*cos_gamma+sin_alpha*sin_gamma
        pose_mat[:, 1, 2] = cos_alpha*sin_beta*sin_gamma-sin_alpha*cos_gamma
        pose_mat[:, 2, 2] = cos_alpha*cos_beta

        pose_mat[:, :3, 3] = pose_vec[:, :3]

        pose_wf_corr = torch.matmul(self._pose_matrix_keyframe, pose_mat)
        if (self.train_correction_model and self.supervised_training):
            self._pose_matrix_keyframe=batch['poses'][-1]
        else:
            self._pose_matrix_keyframe = pose_wf_corr[-1]
            
        return {'corrected_pose_matrix_estimate_wf': pose_wf_corr, 'corrected_pose_matrix_estimate_lf': pose_mat}
               # 'filtered_pose_matrix_estimate_wf': pose_wf_kalman_est, 'filtered_pose_matrix_prediction_wf': pose_wf_kalman_pred}

    def set_pose_matrix_keyframe(self, pose_matrix_keyframe: torch.Tensor):
            del self._pose_matrix_keyframe 
            self.register_buffer('_pose_matrix_keyframe', pose_matrix_keyframe)

    def set_pose_matrix_keyframe_kalman_estimate(self, pose_matrix_keyframe_kalman_estimate: torch.Tensor):
            del self._pose_matrix_keyframe_kalman_estimate 
            self.register_buffer('_pose_matrix_keyframe_kalman_estimate', pose_matrix_keyframe_kalman_estimate)
    
    def set_pose_matrix_keyframe_kalman_prediction(self, pose_matrix_keyframe_kalman_prediction: torch.Tensor):
            del self._pose_matrix_keyframe_kalman_prediction 
            self.register_buffer('_pose_matrix_keyframe_kalman_prediction', pose_matrix_keyframe_kalman_prediction)
