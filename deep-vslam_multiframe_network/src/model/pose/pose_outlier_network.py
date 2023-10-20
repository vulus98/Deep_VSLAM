from typing import Tuple, Dict
from time import time_ns
from numpy import squeeze
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import nn

class PoseEstimationOutlierFeatureTransform(Module):
    def __init__(self, k=64):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.k = k

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).expand(batch_size, -1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PoseEstimationOutlierEstimator(Module):
    def __init__(self,n_estimations=5):
        super().__init__()
        self.n_estimations=n_estimations
        #poses
        self.fc1 = nn.Linear(12, 32)
        self.bn1 = nn.BatchNorm1d(32)
        #features
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        #poses
        self.fc3 = nn.Linear(n_estimations*32, n_estimations*64)
        self.bn3 = nn.BatchNorm1d(n_estimations*64)
        self.fc4 = nn.Linear(n_estimations*64, n_estimations*128)
        self.bn4 = nn.BatchNorm1d(n_estimations*128)
        #features
        self.fc5 = nn.Linear(n_estimations*512, n_estimations*256)
        self.bn5 = nn.BatchNorm1d(n_estimations*256)
        self.fc6 = nn.Linear(n_estimations*256, n_estimations*128)
        self.bn6 = nn.BatchNorm1d(n_estimations*128)
        #aggregate
        self.fc7 = nn.Linear(n_estimations*256, n_estimations*64)
        self.bn7 = nn.BatchNorm1d(n_estimations*64)
        self.fc8 = nn.Linear(n_estimations*64, n_estimations*32)
        self.bn8 = nn.BatchNorm1d(n_estimations*32)
        self.fc9 = nn.Linear(n_estimations*32, n_estimations*16)
        self.bn9 = nn.BatchNorm1d(n_estimations*16)
        self.fc10 = nn.Linear(n_estimations*16, n_estimations*4)
        self.bn10 = nn.BatchNorm1d(n_estimations*4)
        self.fc11 = nn.Linear(n_estimations*4, n_estimations)
        self.bn11 = nn.BatchNorm1d(n_estimations)

        self.out=torch.nn.Softmax(dim=1)



    def forward(self,inputs) -> torch.Tensor:

        poses=inputs[:,:,:12]
        features=inputs[:,:,12:]

        x=torch.stack([F.relu(self.bn1(self.fc1(poses[:,i,:]))) for i in range(self.n_estimations)],dim=1)
        y=torch.stack([F.relu(self.bn2(self.fc2(features[:,i,:]))) for i in range(self.n_estimations)],dim=1)

        x=torch.flatten(x,start_dim=1)
        y=torch.flatten(y,start_dim=1)

        x=F.relu(self.bn3(self.fc3(x)))
        x=F.relu(self.bn4(self.fc4(x)))

        y=F.relu(self.bn5(self.fc5(y)))
        y=F.relu(self.bn6(self.fc6(y)))

        agg=torch.cat((x,y),dim=1)

        agg=F.relu(self.bn7(self.fc7(agg)))
        agg=F.relu(self.bn8(self.fc8(agg)))
        agg=F.relu(self.bn9(self.fc9(agg)))
        agg=F.relu(self.bn10(self.fc10(agg)))
        agg=F.relu(self.bn11(self.fc11(agg)))

        return self.out(agg)




# this network was introduced to try to alleviate for problems of averaging by doing weighted averaging. 
# Input of this network are n estimations for one pose and respective n feature vectors of their frames
# and output should be weights for these n estimations, so by weighted averaging we obtain final pose
# this didn't work until the finish of the project
class PoseEstimationOutlierNetwork(Module):
    def __init__(self, enable_timing: bool = False):
        super().__init__()
        self.add_module('point_net', PoseEstimationOutlierEstimator(n_estimations=5))
        self._enable_timing = enable_timing
        self.time_spent = 0

    def forward(self, x: Dict, keyframe_pose_mat: torch.Tensor) -> Dict:
        poses = x['pose_estimates_wf']
        n_estimations=5
        original_poses=torch.reshape(poses,(poses.shape[0]//n_estimations,n_estimations,4,4))
        poses=torch.reshape(original_poses,(original_poses.shape[0],n_estimations,16,1))
        poses=torch.squeeze(poses,dim=3)
        poses=poses[:,:,:12]
        features=x['global_features_estimations']
        features=torch.reshape(features,(features.shape[0]//5,5,1024,1))
        features=torch.squeeze(features,dim=3)
        inputs=torch.cat((poses,features),dim=2)
        if self._enable_timing and poses.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            inlier_weights = self.point_net(inputs)
            end.record()
            torch.cuda.synchronize()
            self.time_spent += start.elapsed_time(end)
        elif self._enable_timing:
            start = time_ns()
            inlier_weights = self.point_net(inputs)
            self.time_spent += (time_ns() - start) * 1e-6
        else:
            inlier_weights = self.point_net(inputs)
        max_indexes=torch.argmax(inlier_weights,dim=1)
        weighted_poses=original_poses.clone()
        for i in range(n_estimations):
            for j in range(original_poses.shape[0]):
                weighted_poses[j,i,:3,:4]=original_poses[j,i,:3,:4]*inlier_weights[j,i]
        global_features=torch.stack([features[j,index,:] for j,index in enumerate(max_indexes)])
        final_poses=[]
        for i in range(weighted_poses.shape[0]):
            final_poses.append(torch.sum(weighted_poses[i,:,:,:],dim=0))
        final_poses=torch.stack(final_poses)
        return {'pose_matrix_estimate_wf': final_poses, 'global_features': global_features}

    def local_pose(self,keyframe_pose_mat,pose_matrix_estimate_wf):
        local_frame_pose = keyframe_pose_mat.detach().clone()
        local_frame_pose[:3, :3] = local_frame_pose[:3, :3].transpose(-1, -2)
        local_frame_pose[:3, 3:4] = -local_frame_pose[:3, :3] @ local_frame_pose[:3, 3:4]
        return local_frame_pose @ pose_matrix_estimate_wf

    