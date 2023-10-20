from typing import Dict
import torch
from torch._C import device
import cv2 as cv2
import numpy as np

from ...parameters.parameter_config import configure

class PnPOpenCVSolver:
    @configure
    def __init__(self, device: torch.device, threshold: float = None):
        self._intrinsic_matrix = None
        self._device = device
        self._threshold = threshold

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._threshold:
            mask = batch['weights_3d2d']>self._threshold
        else:
            mask = batch['weights_3d2d']>0

        feasible = 0
        decrement = 1
        while not feasible:
            feasible = 1
            for i in range(mask.shape[0]):
                if not (mask[i].sum()>=8):
                    feasible = 0
                    mask[i] = batch['weights_3d2d'][i]>(self._threshold-decrement*0.05*self._threshold)
            decrement = decrement + 1

        ransac_weights = torch.zeros_like(batch['weights_3d2d'])

        pose_matrix_estimate = np.zeros((1, 4, 4))
        batch_size = batch['p3d2d_keyframe'].shape[0]
        for i in range(batch_size):
            repError=1
            # success=False
            # while not success:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(batch['p3d2d_keyframe'][i][mask[i], :3].cpu().numpy(), 
                                                        batch['p3d2d_keyframe'][i][mask[i], 3:].cpu().numpy(), 
                                                        self._intrinsic_matrix.cpu().numpy(), None, reprojectionError=repError, iterationsCount=100, confidence=0.99, 
                                                        flags=cv2.SOLVEPNP_EPNP)
                # if not (success and len(inliers)>=6):
                #     success=False
                #     repError+=0.1
            rmat, _ = cv2.Rodrigues(rvec)
            rmat = rmat.T
            tvec = -rmat @ tvec
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rmat
            pose_matrix[:3, 3:] = tvec
            pose_matrix_estimate = np.concatenate((pose_matrix_estimate, np.expand_dims(pose_matrix, axis=0)))
            idx = torch.where(mask[i])[0]
            if not inliers is None:
                ransac_weights[i, idx[inliers.squeeze()]] = 1.0
            # ransac_weights[i, inliers.squeeze()] = 1.0
        pose_matrix_estimate = torch.from_numpy(pose_matrix_estimate[1:].astype('float32')).to(self._device)

        return {'pose_matrix_estimate_opencv': pose_matrix_estimate, 'weights_3d2d_opencv': ransac_weights}

    def set_intrinsic_matrix(self, intrinsic_matrix: torch.Tensor):
        self._intrinsic_matrix = intrinsic_matrix
