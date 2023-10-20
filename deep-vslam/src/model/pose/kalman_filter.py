from typing import Dict

import torch
from torch._C import device
from torch.nn import Module

from ...parameters.parameter_config import configure


class KalmanFilter(Module):
    __constants__ =['_measure_noise_cov','_process_noise_cov']
    
    @configure
    def __init__(self, process_noise_cov: torch.Tensor, measure_noise_cov: torch.Tensor):
        super().__init__()
        self._process_noise_cov = process_noise_cov
        self._measure_noise_cov  = measure_noise_cov
        self._covariance_matrix_pred = None
        self._model_matrix = torch.zeros((12, 12), device=measure_noise_cov.device)
        self._model_matrix[:6, :6] = 2*torch.eye(6, device=self._model_matrix.device)
        self._model_matrix[:6, 6:] = -torch.eye(6, device=self._model_matrix.device)
        self._model_matrix[6:, :6] = torch.eye(6, device=self._model_matrix.device)
        self._noise_model_matrix = torch.zeros((12, 6), device=measure_noise_cov.device)
        self._noise_model_matrix[:6, :6] = torch.eye(6, device=self._noise_model_matrix.device)
        

    def _predict(self, previous_state_est: torch.Tensor, velocity: torch.Tensor, pred_covariance_matrix: torch.Tensor):
        covariance_matrix = self._model_matrix @ pred_covariance_matrix @ self._model_matrix.T + self._noise_model_matrix @ self._process_noise_cov @ self._noise_model_matrix.T
        state_pred = torch.zeros_like(previous_state_est)
        state_pred[:6] = previous_state_est[:6] + velocity
        state_pred[6:] = previous_state_est[:6]
        return state_pred, covariance_matrix
    
    def _update(self, state_measure: torch.tensor, state_pred: torch.tensor, cov_matrix: torch.tensor):
        residual = state_measure.T - state_pred[:6].T
        S = cov_matrix[:6, :6] + self._measure_noise_cov
        K = cov_matrix[:, :6] @ torch.inverse(S)
        state_est = (state_pred.T + K @ residual).T
        cov_matrix_update = (torch.eye(12, dtype = torch.float, device = K.device)-torch.cat((K, torch.zeros((12, 6), device=K.device)), dim=-1)) @ cov_matrix
        return state_est, cov_matrix_update
        
    def forward(self, batch: Dict[str, torch.Tensor]):
        #velocities = batch['velocity']
        # estimated_velocities = batch['pose_vector_estimate'][1:, :] - batch['pose_vector_estimate'][:-1, :]
        # pose_estimate = torch.zeros_like(pose_measurements)
        # pose_prediction = torch.zeros_like(pose_measurements)
        # pose_estimate[0:2] = pose_measurements[0:2]
        # pose_prediction[0:2] = pose_measurements[0:2]
        # for i in range(1, pose_measurements.shape[0]-1):
        #     pose_prediction[i+1], pred_covariance = self._predict(pose_estimate[i], pose_estimate[i]-pose_estimate[i-1], self._covariance_matrix_pred)
        #     pose_estimate[i+1], self._covariance_matrix_pred = self._update(pose_measurements[i+1], pose_prediction[i+1], pred_covariance)
        # return {'filtered_pose_vector_estimate': pose_estimate, 'filtered_pose_vector_prediction': pose_prediction}
        pose_measurements = batch['pose_vector_estimate_lf'].detach().clone()
        pose_estimate = torch.zeros((pose_measurements.shape[0], 12), device=pose_measurements.device)
        pose_prediction = torch.zeros((pose_measurements.shape[0], 12), device=pose_measurements.device)
        # if not batch['first_frame'][0]:
        #     pose_prediction[0], pred_covariance = self._predict(self.last_estimate, self.last_estimate[:6]-self.last_estimate[6:], self._covariance_matrix_pred)
        #     pose_estimate[0], self._covariance_matrix_pred = self._update(pose_measurements[0], pose_prediction[0], pred_covariance)
        #     pose_prediction[1], pred_covariance = self._predict(pose_estimate[0], pose_estimate[0, :6] - pose_estimate[0, 6:], self._covariance_matrix_pred)
        #     pose_estimate[1], self._covariance_matrix_pred = self._update(pose_measurements[1], pose_prediction[1], pred_covariance)
        # else:
        pose_estimate[0:2, :6] = pose_measurements[0:2, :]
        pose_prediction[0:2, :6] = pose_measurements[0:2, :]
        pose_estimate[1, 6:] = pose_measurements[0, :]
        pose_prediction[1, 6:] = pose_measurements[0, :]
        for i in range(1, pose_measurements.shape[0]-1):
            pose_prediction[i+1], pred_covariance = self._predict(pose_estimate[i], pose_estimate[i, :6]-pose_estimate[i, 6:], self._covariance_matrix_pred)
            pose_estimate[i+1], self._covariance_matrix_pred = self._update(pose_measurements[i+1], pose_prediction[i+1], pred_covariance)
        # if batch['first_frame'][0]:
        velocity_estimate = pose_estimate[1:-1, :6]-pose_estimate[1:-1, 6:]
        # else:
        #     velocity_estimate = torch.cat(((self.last_estimate[:6]-self.last_estimate[6:]).unsqueeze(0), pose_estimate[:-1, :6]-pose_estimate[:-1, 6:]), dim=0)
        # self.last_estimate = pose_estimate[-1].detach().clone()
        return {'filtered_pose_vector_estimate_lf': pose_estimate[:, :6], 'filtered_pose_vector_prediction_lf': pose_prediction[:, :6], 
                'estimated_velocity': velocity_estimate}

    def set_covariance_matrix_pred(self, covariance_matrix_pred):
        del self._covariance_matrix_pred
        self.register_buffer('_covariance_matrix_pred', covariance_matrix_pred)
        

