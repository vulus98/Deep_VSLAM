#!/usr/bin/env python3
from typing import List, Dict
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import yaml
import wandb
import pandas as pd
import os

#os.chdir('/home/vbozic/semester_project/deep-vslam')
from src.data.generate_synthetic_trajectory_data import SyntheticTrajectoryCorrespondences
from src.data.kitti_extracted_data import KITTIDataset
from src.model.model_pipeline import ModelPipeline
from src.model.loss.total_loss import TotalLoss
from src.parameters.parameter_config import Config, get_loggable_params
from src.training.training import TrainEngine


# Argument parser
def parse_args():
    parser = ArgumentParser(description='train pointnet for 3d2d correspondences outlier detection')
    parser.add_argument(
                        '--config', 
                        help='YAML config file containing parameters', 
                        default='/srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/deep-vslam/config/train_outlier_network_kitti.yml'
    )
    return parser.parse_args()

# Prepare config
def prepare_config(arguments) -> dict:
    with open(arguments.config, 'r') as f:
        try:
            params = yaml.safe_load(f)

        except yaml.YAMLError as e:
            print(e)
            exit(1)
    if 'k_matrix' in params:
        params['k_matrix'] = torch.tensor(params['k_matrix'])
    else:
        params['k_matrix'] = torch.eye(3)
    
    if not 'lr_decay' in params:
        params['lr_decay'] = None
    
    if not 'train_sequences' in params:
        params['train_sequences'] = []

    if not 'test_sequences' in params:
        params['test_sequences'] = []

    params['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device used in training process is', params['device'])

    if not 'process_noise_cov' in params:
        params['process_noise_cov'] = torch.tensor(pd.read_csv(params['kalman_parameters_path']+"process_cov.csv", header=None).values, dtype=torch.float, device = params['device'])
    else:
        params['process_noise_cov'] = torch.tensor(params['process_noise_cov'], device=params['device'])
        
    if not 'measure_noise_cov' in params:
        params['measure_noise_cov'] = torch.tensor(pd.read_csv(params['kalman_parameters_path']+"measurement_cov.csv", header=None).values, dtype=torch.float, device = params['device'])
    else:
        params['measure_noise_cov'] = torch.tensor(params['measure_noise_cov'], device=params['device'])
    
    return params


# Main
if __name__ == '__main__':
    args = parse_args()
    params = prepare_config(args)
    Config(config=params).make_global()

    with wandb.init(project=params['wandb_project_name'], config=get_loggable_params(), entity='vbozic'):
        if params['tag'] == 'kitti':
            train_sequences = KITTIDataset(
                dataset_path=params['extracted_data_path'],
                sequence_ids_to_load=params['train_sequences'],
            )

            test_sequences = KITTIDataset(
                dataset_path=params['extracted_data_path'],
                sequence_ids_to_load=params['test_sequences']
            )

            model_pipeline = ModelPipeline()
            
            if params['load_outlier_model']:
                model_pipeline.outlier_models[0].load_state_dict(torch.load(params['outlier_rejection_model_path'], params['device']))
            if params['load_correction_model']:
                model_pipeline.kalman_correction_models[0].load_state_dict(torch.load(params['kalman_correction_model_path'], params['device']))  

            loss = TotalLoss()

            train_engine = TrainEngine(
                model_pipeline=model_pipeline,
                loss=loss,
                train_sequences=train_sequences,
                test_sequences=test_sequences,
            )

        train_engine.train_model()
        train_engine.eval_model()