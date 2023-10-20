#!/usr/bin/env python3
import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml

from data.generate_synthetic_trajectory_data import SyntheticTrajectoryCorrespondences
from model.model_pipeline import ModelPipeline
from parameters.parameter_config import Config

def parse_args():
    parser = ArgumentParser(description='calculate process and measurement noise covariance matrices')
    parser.add_argument('config', help='YAML config file containing parameters')
    return parser.parse_args()


def prepare_config(arguments) -> dict:
    with open(arguments.config, 'r') as f:
        try:
            params = yaml.safe_load(f)

        except yaml.YAMLError as e:
            print(e)
            exit(1)

    if 'k_matrix' in params:
        params['k_matrix'] = torch.tensor(params['k_matrix'])
    
    params['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device used in the process is', params['device'])
    
    return params


if __name__ == '__main__':
    sys.path.append('..')
    args = parse_args()
    params = prepare_config(args)
    Config(config=params).make_global()

    model_pipeline = ModelPipeline()
    trajectory_batches = SyntheticTrajectoryCorrespondences()
    trajectory_batches_dl = DataLoader(trajectory_batches, batch_size=params['batch_size'], num_workers=0)

    model_pipeline.outlier_models[0].load_state_dict(torch.load('models/outlier_network_state_dict_best.pth', map_location=params['device']))

    num_batches = int(params['max_n_samples']/params['batch_size'])
    process_error = torch.zeros((num_batches*(params['batch_size']-2), 6))
    measure_error = torch.zeros((num_batches*params['batch_size'], 6))
    k=0
    l=0
    for index, trajectory_batch in enumerate(trajectory_batches_dl):
        trajectory_batch.update(model_pipeline(trajectory_batch))
        for i in range(params['batch_size']):
            if i>1:
                velocity = trajectory_batch['camera_pose_vec'][i-1, :] - trajectory_batch['camera_pose_vec'][i-2, :]
                pose_prediction_error = -trajectory_batch['camera_pose_vec'][i] + trajectory_batch['camera_pose_vec'][i-1] + velocity
                process_error[k, :] = pose_prediction_error
                k += 1            
            measure_error[l, :] = trajectory_batch['pose_vector_estimate'][i, :]-trajectory_batch['camera_pose_vec'][i, :]
            l += 1
    process_noise_cov = pd.DataFrame(np.cov(process_error, rowvar=False)).to_csv('data/processed/process_noise_covariance_matrix.csv', header=None, index=None)
    measure_noise_cov = pd.DataFrame(np.cov(measure_error, rowvar=False)).to_csv('data/processed/measurement_noise_covariance_matrix.csv', header=None, index=None)

