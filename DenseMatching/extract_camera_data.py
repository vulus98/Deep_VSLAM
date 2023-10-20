import pykitti
import numpy as np
import h5py

basedir = '../../data_sets/kitti/dataset'

sequence_number = []
for i in range(22):
    if i//10:
        sequence_number.append(str(i))
    else:
        sequence_number.append('0'+str(i))

hf = h5py.File('/srv/beegfs02/scratch/deep_slam/data/data_sets/kitti/extracted_data/PDC_correspondences_intrinsics_cam2/extracted_features_full_sequence_00_new_format_TMP.h5', 'a')

for sequence in sequence_number:
    dataset = pykitti.odometry(basedir, sequence, frames = range(0, 1))
    hf.create_dataset(sequence + '/calibration_matrix', data=dataset.calib.K_cam2)

hf.close()
    



