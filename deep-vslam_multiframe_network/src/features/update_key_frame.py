from typing import Dict, List
import torch
from torch.nn import Module
from ..parameters.parameter_config import configure

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

class ReversePoseToWorldFrame(Module):
    @configure
    def __init__(self, device,train_outlier_model: bool= True,multiframe_size:int= 5,similarity_threshold: int=0):
        super().__init__()
        self.keyframe_pose = None
        self.train_outlier_model=train_outlier_model
        self.multiframe_size=multiframe_size
        self.similarity_threshold=similarity_threshold

    def forward(self, batch: Dict[str, torch.Tensor], keyframe_pose_mat: torch.Tensor, pose_matrices_past: List) -> Dict[str, torch.Tensor]:
        self.keyframe_pose = keyframe_pose_mat.detach().clone()
        for pose in pose_matrices_past:
            pose=pose.detach().clone()
        if(self.train_outlier_model):
            estimated_poses=batch['pose_matrix_estimate']
        else:
            estimated_poses=batch['pose_matrix_estimate_opencv']
        batch_size = estimated_poses.shape[0]

        camera_pose_wf=torch.zeros_like(torch.unsqueeze(estimated_poses[0],0), device=keyframe_pose_mat.device)
        pose_matrix_estimate_wf=torch.zeros_like(torch.unsqueeze(estimated_poses[0],0), device=keyframe_pose_mat.device)
        global_features=torch.zeros_like(torch.unsqueeze(batch['global_features_estimations'][0],0), device=keyframe_pose_mat.device)
        for k in range(batch_size):
            camera_pose=torch.matmul(pose_matrices_past[k%(self.multiframe_size-1)], estimated_poses[k])
            if (k%(self.multiframe_size-1))>0 and (k%(self.multiframe_size-1))<6:
                camera_pose_wf=torch.cat((camera_pose_wf,torch.unsqueeze(camera_pose,0)),dim=0)
            if not((k+1)%(self.multiframe_size-1)):
                if(camera_pose_wf[-1][3][3]<1e-5):
                    camera_pose_wf[-1]=torch.eye(4, device=keyframe_pose_mat.device)
                pose_matrix_estimate_wf=torch.cat((pose_matrix_estimate_wf,camera_pose_wf[1:]),dim=0)
                extracted_features=batch['global_features_estimations'][k-(self.multiframe_size-1)+2:k-(self.multiframe_size//2-1)]
                #new_camera_pose,new_global_feature=self.estimate_pose(pose_matrix_estimate_wf,features,pose_matrices_past[self.multiframe_size//2-1])
                

                #pose_matrix_estimate_wf.append(new_camera_pose)
                global_features=torch.cat((global_features,extracted_features),dim=0)
                for i in range(1,self.multiframe_size//2):
                    pose_matrices_past[i-1]=pose_matrices_past[i]
                pose_matrices_past[self.multiframe_size//2-1]=pose_matrix_estimate_wf[-1].detach().clone()
                camera_pose_wf=torch.zeros_like(torch.unsqueeze(estimated_poses[0],0), device=keyframe_pose_mat.device)
  
        #pose_matrix_estimate_wf=torch.stack(pose_matrix_estimate_wf)
        #pose_matrix_estimate_lf=self.local_pose(keyframe_pose_mat,pose_matrix_estimate_wf)
        return {'pose_estimates_wf': pose_matrix_estimate_wf[1:], 'global_features_estimations': global_features[1:]}


 

