from typing import Dict, List
import torch
from torch.nn import Module
import numpy as np
from ..parameters.parameter_config import configure
from scipy.spatial.transform import Rotation as R

class UpdatePointsToKeyFrame(Module):
    def __init__(self):
        super().__init__()
        self.keyframe_pose = None

    def forward(self, batch: Dict[str, torch.Tensor], keyframe_pose_mat: torch.Tensor) -> Dict[str, torch.Tensor]:
        p3d2d = batch['p3d2d'].detach().clone()
        self.keyframe_pose = keyframe_pose_mat.detach().clone()
        p3d_h = torch.nn.functional.pad(p3d2d[:, :, :3], [0, 1], "constant", 1.0)
        keyframe_pose_mat_inv = torch.eye(4, dtype=keyframe_pose_mat.dtype, device=keyframe_pose_mat.device)
        keyframe_pose_mat_inv[:3, :3] = self.keyframe_pose[:3, :3].transpose(-1, -2)
        keyframe_pose_mat_inv[:3, 3:4] = -keyframe_pose_mat_inv[:3, :3] @ self.keyframe_pose[:3, 3:4]
        p3d_keyframe = torch.matmul(keyframe_pose_mat_inv, p3d_h.transpose(-2, -1)).transpose(-2, -1)[:, :, :3]
        p3d2d[:, :, :3] = p3d_keyframe
        return {'p3d2d_keyframe': p3d2d}

# For estimation using future and not just previous frames uncomment comments in the following function
class ReversePoseToWorldFrame(Module):
    @configure
    def __init__(self, device,train_outlier_model: bool= True,multiframe_size:int= 5,similarity_threshold: int=0):
        super().__init__()
        self.keyframe_pose = None
        self.train_outlier_model=train_outlier_model
        self.multiframe_size=multiframe_size
        self.similarity_threshold=similarity_threshold

    def forward(self, batch: Dict[str, torch.Tensor], keyframe_pose_mat: torch.Tensor, pose_matrices_past: List,pose_matrices_full: List) -> Dict[str, torch.Tensor]:
        self.keyframe_pose = keyframe_pose_mat.detach().clone()
        if(self.train_outlier_model): # while training, use estimates from our PnP solver because it is differentiable and openCV one isn't
            estimated_poses=batch['pose_matrix_estimate']
        else:
            estimated_poses=batch['pose_matrix_estimate_opencv']
        batch_size = estimated_poses.shape[0]
        camera_pose_wf=[]
        pose_matrix_estimate_wf=torch.zeros_like(torch.unsqueeze(estimated_poses[0],0), device=keyframe_pose_mat.device)
        global_features=torch.zeros_like(torch.unsqueeze(batch['global_features'][0],0), device=keyframe_pose_mat.device)
        for k in range(batch_size):
            # obtain one estimation of current pose and store it
            camera_pose=torch.matmul(pose_matrices_past[k%(self.multiframe_size-1)], estimated_poses[k])
            camera_pose_wf.append(camera_pose)
            if not((k+1)%(self.multiframe_size-1)): # if we collect all available estimations in array camera_pose_wf
                if(camera_pose_wf[self.multiframe_size//2-1][3][3]<1e-5): # if this is first pose (start of estimations)
                    camera_pose_wf[self.multiframe_size//2-1]=torch.eye(4, device=keyframe_pose_mat.device)
                features=batch['global_features'][k-(self.multiframe_size-1)+1:k+1]
                # from n available estimations, determine final one and append array with final estimations 
                new_camera_pose,new_global_feature=self.estimate_pose(camera_pose_wf,features,pose_matrices_past[self.multiframe_size//2-1])
                pose_matrix_estimate_wf=torch.cat((pose_matrix_estimate_wf,torch.unsqueeze(new_camera_pose,0)),dim=0)
                global_features=torch.cat((global_features,torch.unsqueeze(new_global_feature,0)),dim=0)
                for i in range(1,self.multiframe_size//2): # here update matrices which store previous poses, which are used in estimation of global trajectory of next one
                    pose_matrices_past[i-1]=pose_matrices_past[i]
                # for i in range(1,self.multiframe_size):
                #     pose_matrices_full[i-1]=pose_matrices_full[i]
                pose_matrices_past[self.multiframe_size//2-1]=pose_matrix_estimate_wf[-1]
             #  pose_matrices_full[-1]=pose_matrix_estimate_wf[-1]
                for pose in pose_matrices_past:
                    pose=pose.detach().clone()
                # for pose in pose_matrices_full:
                #     pose=pose.detach().clone()
                camera_pose_wf=[]
                # if(k>(self.multiframe_size//2)*(self.multiframe_size-1)):
                #     temporal_pose_matrices=pose_matrices_full.copy()
                #     temporal_pose_matrices.pop(self.multiframe_size//2)
                #     for i in range(0,self.multiframe_size-1):
                #         camera_pose_wf.append(torch.matmul(temporal_pose_matrices[i], estimated_poses[k-(self.multiframe_size//2+1)*(self.multiframe_size-1)+i+1]))
                #     features=batch['global_features'][k-(self.multiframe_size//2+1)*(self.multiframe_size-1)+1:k-(self.multiframe_size//2)*(self.multiframe_size-1)+1]
                #     new_camera_pose,new_global_feature=self.estimate_pose(camera_pose_wf,features,pose_matrices_full[self.multiframe_size//2-1])
                #     pose_matrix_estimate_wf[-(self.multiframe_size//2+1)] = new_camera_pose
                #     global_features[-(self.multiframe_size//2+1)]=torch.unsqueeze(new_global_feature,0)
                #     camera_pose_wf=[]
        pose_matrix_estimate_lf=self.local_pose(keyframe_pose_mat,pose_matrix_estimate_wf) # determine estimates in local reference frame
        return {'pose_matrix_estimate_wf': pose_matrix_estimate_wf[1:], 'pose_matrix_estimate_lf': pose_matrix_estimate_lf[1:], 'global_features': global_features[1:]}
    
    # function which can be used to estimate final pose from n-estimations, first transforms poses in euler angles and then averages them
    def calculate_mean_euler(self,camera_poses,indexes,prob):
        mean_pose=torch.eye(4,device=camera_poses[0].device)
        mean_angle=np.zeros((1,3))
        mean_trans=torch.zeros((3,1),device=mean_pose.device)
        for pose in [camera_poses[index] for index in indexes]:
            mean_trans+=pose[:3,3:4]
            r = R.from_matrix(pose[:3,:3].cpu().detach().numpy())
            mean_angle+=r.as_euler('zyx', degrees=True)
        print(mean_angle/prob)
        mean_angle=R.from_euler('zyx',mean_angle/prob,degrees=True)
        mean_pose[:3,:3]=torch.from_numpy(mean_angle.as_matrix()).float().to(mean_pose.device)
        mean_pose[:3,3:4]=mean_trans/prob
        return mean_pose
    
    # similar to previous function, just uses different representation of pose matrices
    def calculate_mean_cayley(self,camera_poses,indexes,prob):
        mean_pose=torch.eye(4,device=camera_poses[0].device)
        mean_trans=torch.zeros((3,1),device=mean_pose.device)
        mean_rot=torch.zeros((3,1),device=mean_pose.device)
        rot_vect=torch.zeros((3,1),device=mean_pose.device)
        rot_mat=torch.zeros((3,3),device=mean_pose.device)
        for pose in [camera_poses[index] for index in indexes]:
            mean_trans+=pose[:3,3:4]
            rot_mat=torch.matmul(torch.inverse(pose[:3,:3]+torch.eye(3,device=mean_pose.device)),pose[:3,:3]-torch.eye(3,device=mean_pose.device))
            rot_vect[0]=rot_mat[2][1] #if rot_mat[2][1]>(-(np.pi-1e-2)) else np.pi 
            rot_vect[1]=rot_mat[0][2] #if rot_mat[0][2]>(-(np.pi-1e-2)) else np.pi 
            rot_vect[2]=rot_mat[1][0] #if rot_mat[1][0]>(-(np.pi-1e-2)) else np.pi 
            mean_rot+=rot_vect
        mean_rot=mean_rot/prob
        rot_mat=torch.tensor([[0,-rot_vect[2],rot_vect[1]],[rot_vect[2],0,-rot_vect[0]],[-rot_vect[1],rot_vect[0],0]],device=mean_pose.device)
        mean_pose[:3,:3]=torch.matmul(torch.inverse(torch.eye(3,device=mean_pose.device)-rot_mat),(torch.eye(3,device=mean_pose.device)+rot_mat))
        mean_pose[:3,3:4]=mean_trans/prob
        return mean_pose
    
    # final estimation of pose, does clustering and then averaging on cluster, averaging can be done with previous 2 functions or with jsut averaging of pose matrices
    def estimate_pose(self, camera_poses: torch.Tensor, global_features: torch.Tensor,previous_pose: torch.Tensor)-> torch.Tensor:
        poses=[]
        features=[]
        for i in range(1,self.multiframe_size-1):
            if(camera_poses[i][3][3]>1e-5):
                poses.append(camera_poses[i])
                features.append(global_features[i])
        means=[]
        indexes=[]
        prob=[]
        means.append(poses[0])
        indexes.append([])
        indexes[0].append(0)
        prob.append(1)
        for i in range(1,len(poses)):
            min_dist=1000
            for j in range(0,len(means)):
                distance= torch.linalg.matrix_norm(poses[i] - means[j]).mean()
                if(distance<min_dist):
                    min_dist=distance
                    min_ind=j
            if(min_dist<self.similarity_threshold):
                indexes[min_ind].append(i)
                prob[min_ind]+=1
                means[min_ind]=(means[min_ind]*(prob[min_ind]-1)+poses[i])/prob[min_ind]
                #means[min_ind]=self.calculate_mean_euler(poses,indexes[min_ind],prob[min_ind])
            else:
                means.append(poses[i])
                indexes.append([])
                indexes[-1].append(i)
                prob.append(1)
        distances=[torch.linalg.matrix_norm(mean-previous_pose).mean() for mean in means]
        score=np.divide(np.exp(prob),distances)
        index_max=np.argmax(score)
        return means[index_max],features[indexes[index_max][0]]

    def local_pose(self,keyframe_pose_mat,pose_matrix_estimate_wf):
        local_frame_pose = keyframe_pose_mat.detach().clone()
        local_frame_pose[:3, :3] = local_frame_pose[:3, :3].transpose(-1, -2)
        local_frame_pose[:3, 3:4] = -local_frame_pose[:3, :3] @ local_frame_pose[:3, 3:4]
        return local_frame_pose @ pose_matrix_estimate_wf
    

