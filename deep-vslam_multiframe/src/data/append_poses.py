import h5py
import pykitti
basedir = '/srv/beegfs02/scratch/deep_slam/data/data_sets/kitti/dataset/'
hf = h5py.File('/srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/data/correspondences/extracted_data_5_frames_1024_features.h5', 'a')
sequence_number=[]
for i in range(11):
    if i//10:
        sequence_number.append(str(i))
    else:
        sequence_number.append('0'+str(i))
#for sequence in sequence_number:
  #  dataset = pykitti.odometry(basedir, sequence)
  #  hf.create_dataset(sequence + '/poses', data=dataset.poses)
print('finished writing poses matrix')
hf.close()