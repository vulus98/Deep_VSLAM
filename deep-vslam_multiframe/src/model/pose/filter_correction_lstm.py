from typing import Dict, Tuple
import torch
from torch._C import device
import torch.nn as nn
from torch.nn import Module


# class LSTM(Module):
#     # input_size = 1036, 18
#     def __init__(self, input_size = 1036, output_size = 6, hidden_dim = 12 , n_layers = 2):
#         super(LSTM, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.lstm = nn.LSTM(input_size, hidden_dim, n_layers)   
#         self.fc = nn.Linear(hidden_dim, output_size)
    
#     def forward(self, trajectory_batch: Dict[str, torch.tensor]):
#         x = torch.cat((trajectory_batch['filtered_pose_vector_estimate_lf'][1:-1].unsqueeze(1), 
#                         trajectory_batch['estimated_velocity'].unsqueeze(1), 
#                         trajectory_batch['normalized_global_features'][2:].transpose(-2,-1)), dim = -1)
#         self.hidden_0 = (torch.zeros(2,1,12, device=x.device), torch.zeros(2,1,12, device=x.device))
#         out, hidden = self.lstm(x, self.hidden_0)
#         out = out.contiguous().view(-1, self.hidden_dim)
#         out = self.fc(out)
#         corrected_prediction = trajectory_batch['filtered_pose_vector_prediction_lf'].clone().detach()
#         corrected_prediction[2:, :] = corrected_prediction[2:, :] + out
#         #corrected_prediction = corrected_prediction + out 

#         return {'lstm_correction': out, 'corrected_kalman_prediction_lf': corrected_prediction, 'input': x}

class LSTM(Module):
    # input_size = 1036, 18
    def __init__(self, input_size = 1030, output_size = 6, hidden_dim = 12 , n_layers = 2):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    #    self.bn=nn.BatchNorm1d(1)
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers)   
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, trajectory_batch: Dict[str, torch.tensor]):
        #if not (self.training):
        x = torch.cat((trajectory_batch['pose_vector_estimate_lf'].unsqueeze(1),
                    # trajectory_batch['resnet_features'][2:].unsqueeze(1)), dim = -1)
                    trajectory_batch['normalized_global_features'].transpose(-2,-1)), dim = -1)
        corrected_prediction = trajectory_batch['pose_vector_estimate_lf'].clone().detach()
        # else:
        #     x = torch.cat((trajectory_batch['filtered_pose_vector_estimate_lf'][1:-1].unsqueeze(1),
        #                 # trajectory_batch['resnet_features'][2:].unsqueeze(1)), dim = -1)
        #                 trajectory_batch['normalized_global_features'][2:].transpose(-2,-1)), dim = -1)
        #     corrected_prediction = trajectory_batch['filtered_pose_vector_prediction_lf'].clone().detach()
        self.hidden_0 = (torch.zeros(2,1,12, device=x.device), torch.zeros(2,1,12, device=x.device))
        out, hidden = self.lstm(x, self.hidden_0)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        corrected_prediction[2:, :] = corrected_prediction[2:, :] + out 

        return {'lstm_correction': out, 'corrected_kalman_prediction_lf': corrected_prediction, 'input': x}
