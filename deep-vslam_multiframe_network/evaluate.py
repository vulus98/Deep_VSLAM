from typing import List, Dict
import torch
import yaml
from src.data.kitti_extracted_data import KITTIDataset
from src.model.model_pipeline import ModelPipeline
from src.model.loss.total_loss import TotalLoss
from src.parameters.parameter_config import Config,configure
import numpy as np

import pandas as pd
from typing import Dict, List
import torch
from src.parameters.parameter_config import configure


def prepare_config(arguments) -> dict:
    with open(arguments, 'r') as f:
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


@configure
def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    if isinstance(batch, List):
        for i, tens in enumerate(batch):
            batch[i] = tens.to(device)
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    elif isinstance(batch, Dict):
        for key, item in batch.items():
            batch[key] = item.to(device)
    else:
        raise Exception('Batch data has to be torch.tensor or List[torch.tensor] or Dict[str, torch.tensor].')
    return batch

def main():

    params = prepare_config('/srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/deep-vslam_multiframe_network/config/test_opencv_kitti.yml')

    Config(config=params).make_global()

    test_sequences = KITTIDataset(
        dataset_path=params['extracted_data_path'],
        sequence_ids_to_load=params['test_sequences']
    )

    model_pipeline = ModelPipeline()

    loss = TotalLoss()
    model_pipeline.eval()

    # uncomment for debugging on local PC
 #   model_pipeline.outlier_models[0].load_state_dict(torch.load(params['outlier_rejection_model_path'],map_location=torch.device('cpu')))
#    model_pipeline.kalman_correction_models[0].load_state_dict(torch.load(params['kalman_correction_model_path'], map_location=torch.device('cpu')))
    model_pipeline.outlier_models[0].load_state_dict(torch.load(params['outlier_rejection_model_path'], params['device']))
    model_pipeline.pose_outlier_models[0].load_state_dict(torch.load(params['pose_outlier_model_path'], params['device']))
    #model_pipeline.kalman_correction_models[0].load_state_dict(torch.load(params['kalman_correction_model_path'], params['device']))  

    for seq_dataloader in test_sequences.sequences:
        seq_num = seq_dataloader.dataset.get_sequence_id()

        model_pipeline.reinitialize_for_new_sequence(
            intrinsic_matrix=to_device(seq_dataloader.dataset.intrinsic_matrix)
        )
        loss.reinitialize_for_new_sequence(
            intrinsic_matrix=to_device(seq_dataloader.dataset.intrinsic_matrix)
        )
        
        print('Sequence: ', seq_num)

        file = open(params['result_path'] + seq_num + '.txt', 'w+')
       
        for batch_i, batch in enumerate(seq_dataloader):
            batch = to_device(batch)
            output = model_pipeline(batch)
            for k in range(output['pose_matrix_estimate_wf'].shape[0]):
                if(params['test_with_correction']):
                    np.savetxt(file, np.reshape(output['corrected_pose_matrix_estimate_wf'][k, 0:3, :].cpu().numpy(), (-1, 1, 12)).squeeze(0))
                else:
                    np.savetxt(file, np.reshape(output['pose_matrix_estimate_wf'][k, 0:3, :].cpu().numpy(), (-1, 1, 12)).squeeze(0))
        file.close()
        print('finished evaluation')
        
if __name__ == "__main__":
    main()