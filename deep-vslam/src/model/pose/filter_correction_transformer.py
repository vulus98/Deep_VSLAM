from typing import Dict, Tuple
import math
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from ...parameters.parameter_config import configure


class TransformerCorrection(Module):
    # input_size = 1036, 18
    @configure
    def __init__(self, device, input_size = 1036, output_size=6, d_model=96, dim_feedforward=192, num_layers=4, n_heads=3, batch_size=10):
        super(TransformerCorrection, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_in = nn.Linear(input_size, d_model-12)
        self.fc_out = nn.Linear(d_model, output_size)
        self.mask = generate_square_subsequent_mask(batch_size-2).to(device)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc_in.bias.data.zero_()
        self.fc_in.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, trajectory_batch: Dict[str, torch.tensor]):
        x = trajectory_batch['normalized_global_features'][2:].transpose(-2,-1).detach().clone()
        inp = self.fc_in(x)
        # inp = torch.cat((trajectory_batch['filtered_pose_vector_estimate_lf'][1:-1].unsqueeze(1), 
        #                  trajectory_batch['estimated_velocity'].unsqueeze(1), 
        #                  inp), dim=-1)
        inp = torch.cat((trajectory_batch['pose_vector_estimate_lf'][1:-1].unsqueeze(1), 
                    trajectory_batch['estimated_velocity'].unsqueeze(1), 
                    inp), dim=-1)
        inp= self.pos_encoder(inp)
        out = self.transformer_encoder(inp, self.mask)
        out = self.fc_out(out)
        out = out.squeeze()
        #corrected_prediction = trajectory_batch['filtered_pose_vector_prediction_lf'].clone().detach()
        corrected_prediction = trajectory_batch['pose_vector_estimate_lf'].clone().detach()
        corrected_prediction[2:, :] = corrected_prediction[2:, :] + out 

        return {'lstm_correction': out, 'corrected_kalman_prediction_lf': corrected_prediction}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask