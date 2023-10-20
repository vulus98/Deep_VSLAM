from typing import Dict
import torch
from torch.nn import Module


class TrajectoryNormalizer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, trajectory_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trajectory = trajectory_batch['pose_vector_estimate_lf'].detach().clone()
       # trajectory_prediction = trajectory_batch['filtered_pose_vector_prediction_lf'].detach().clone()
        #filtered_trajectory = trajectory_batch['filtered_pose_vector_estimate_lf'].detach().clone()
        global_features = trajectory_batch['global_features'].detach().clone() #TODO
        global_features_normalized = torch.div(global_features, torch.norm(global_features, dim=-2, keepdim=True))
        #filtered_trajectory_mean = filtered_trajectory.mean(dim=-2, keepdim=True)
        #filtered_trajectory_normalized = filtered_trajectory - filtered_trajectory_mean
        trajectory_mean = trajectory.mean(dim=-2, keepdim=True)
        trajectory_normalized = trajectory - trajectory_mean
      #  trajectory_prediction_mean = trajectory_prediction.mean(dim=-2, keepdim=True)
       # trajectory_prediction_normalized = trajectory_prediction - trajectory_prediction_mean

        return {#'normalized_filtered_pose_vector_estimate_wf': filtered_trajectory_normalized,
                'normalized_pose_vector_estimate_wf': trajectory_normalized,
                #'normalized_filtered_pose_vector_prediction_wf': trajectory_prediction_normalized, 
                'normalized_global_features': global_features_normalized}
                #'trajectory_normalizer': filtered_trajectory_mean}