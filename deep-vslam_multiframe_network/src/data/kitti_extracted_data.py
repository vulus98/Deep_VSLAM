import h5py
import torch
import numpy as np
import pandas as pd
#from torch._C import namedtuple_solution_cloned_coefficient
from torch.utils.data import IterableDataset, DataLoader
import random

from ..parameters.parameter_config import configure


class KITTIDataset:
    @configure
    def __init__(self, dataset_path: str, sequence_ids_to_load: list = [],  batch_size: int = 8, num_workers: int = 0):
        super(KITTIDataset).__init__()

        self.dataset_path = dataset_path
        self.sequence_ids_to_load = sequence_ids_to_load
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create a dictionary containing each sequence as an IterableDataset
        # (sequences contain corresponences, camera intrinsic matrices, sequence lenghts, ...)
        sequences = self._load_sequences(
            sequence_ids_to_load=sequence_ids_to_load,
        )

        self.sequences = self._wrap_sequences_in_data_loaders(
            sequences=sequences
        )

    def shuffle_sequence_list(self):
        random.shuffle(self.sequences)

    def get_sequence_list(self):
        return self.sequences

    def len(self):
        return len(self.sequences)

    def _load_sequences(self, sequence_ids_to_load: list = []) -> list:
        # Load prepared data for requested sequences.
        # Load from a HDF5 file.
        if not sequence_ids_to_load:
            for i in range(22):
                if i//10:
                    sequence_ids_to_load.append(str(i))
                else:
                    sequence_ids_to_load.append('0'+str(i))
        hf = h5py.File(self.dataset_path, 'r')
        hf_features=h5py.File('/srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/data/features/extracted_features_00.h5','r')
        # Create a list of sequences.
        # Every entry will be an interable dataset
        # corresponding to that sequence.
        sequences = []
        for key in hf.keys():
            if key in sequence_ids_to_load:
                # p3dBundleCloud = np.array(hf.get(key + '/p3dBundleCloud')).astype('float32')
                # p3dBundleCloud = np.concatenate(
                #     (
                #         p3dBundleCloud,
                #         np.ones((p3dBundleCloud.shape[0], p3dBundleCloud.shape[1], 1)).astype('float32')
                #     ),
                #     axis=2,
                # )
                # sequence_data = {
                #     'p3d2dMatchedForward': np.array(hf.get(key + '/p3d2dMatchedForward'))[1:].astype('float32'), # TODO <------------------ [1:]
                #     'p3d2dMatchedBackward': np.array(hf.get(key + '/p3d2dMatchedBackward'))[1:].astype('float32'),
                #     'p3d2dKeyFrames':  np.array(hf.get(key + '/p3d2dKeyFrames')).astype('float32'),
                #     'p3d2dKeyFramesMatchCount':  np.array(hf.get(key + '/p3d2dKeyFramesMatchCount')).astype('int32'),
                #     'p2dMatchedKeyFramesForward': np.array(hf.get(key + '/p2dMatchedKeyFramesForward')).astype('int32'),
                # }
                if(int(key)<11):
                    sequence_data = {
                        'p3d2dMatchedForward': np.array(hf.get(key + '/p3d2dMatchedForward'))[1:].astype('float32'), # TODO <------------------ [1:]
                       # 'resnet_features': np.array(hf_features.get(key + '/imageFeatures')).astype('float32'),
                        'poses': np.array(hf.get(key + '/poses')).astype('float32'),
                    }
                else:
                    sequence_data = {
                        'p3d2dMatchedForward': np.array(hf.get(key + '/p3d2dMatchedForward'))[1:].astype('float32'), # TODO <------------------ [1:]
                       # 'resnet_features': np.array(hf_features.get(key + '/imageFeatures')).astype('float32'),
                    }
                sequences.append(
                    KITTISequence(
                        sequence_data=sequence_data,
                        intrinsic_matrix=torch.tensor(np.array(hf.get(key + '/calibration_matrix')).astype('float32')),
                        sequence_id=key,
                    )
                )
        return sequences

    def _wrap_sequences_in_data_loaders(self, sequences: list) -> list:
        sequence_data_loaders = []
        for seq in sequences:
            sequence_data_loaders.append(
                DataLoader(
                    dataset=seq,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True
                )
            )
        return sequence_data_loaders


class KITTISequence(IterableDataset):
    @configure
    def __init__(self, sequence_data: dict, intrinsic_matrix: torch.Tensor, sequence_id: str):
        super(KITTISequence).__init__()

        self.sequence_data = sequence_data
        self.intrinsic_matrix = intrinsic_matrix
        self.sequence_id = sequence_id

    def __len__(self):
        return self.sequence_data['p3d2dMatchedForward'].shape[0]

    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration()
        p3d2d = self.sequence_data['p3d2dMatchedForward'][self.index]
        poses = self.sequence_data['poses'][self.index]
       # resnet_features = self.sequence_data['resnet_features'][self.index]
        self.index = self.index + 1
        return {
            'p3d2d_keyframe': p3d2d[:,:1024,:],     # TODO <------------------ Hardcoded 1024 for testing
            'poses': poses,
          #  'resnet_features': resnet_features
        }
    
    def fetch_BA_data(self):
        return {
            'p3d2dKeyFrames': self.sequence_data['p3d2dKeyFrames'],
            'p3d2dKeyFramesMatchCount': self.sequence_data['p3d2dKeyFramesMatchCount'],
            'p2dMatchedKeyFramesForward': self.sequence_data['p2dMatchedKeyFramesForward']
        }

    def get_intrinsic_matrix(self):
        return self.intrinsic_matrix

    def get_sequence_id(self):
        return self.sequence_id
