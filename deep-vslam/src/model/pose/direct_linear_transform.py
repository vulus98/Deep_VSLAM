from typing import Dict

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from torch.nn import Module
import math

from ...parameters.parameter_config import configure


class DLTPoseSolver(Module):
    __constants__ = ['_threshold']

    @configure
    def __init__(self, threshold: float = None):
        super().__init__()
        self._threshold = threshold

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.training and self._threshold:
            weights = (batch['weights_3d2d']>self._threshold).float().unsqueeze(-1)
            feasible = 0
            decrement = 1
            while not feasible:
                feasible = 1
                for i in range(weights.shape[0]):
                    if not weights[i].sum()>8: ###### minimal number of points needed for DLT
                        feasible = 0
                        weights[i] = (batch['weights_3d2d'][i]>(self._threshold-decrement*0.05*self._threshold)).float().unsqueeze(-1)
                decrement = decrement + 1
        else:
            weights = batch['weights_3d2d'].unsqueeze(-1)
        
        p3d2d = batch['p3d2d_n']
        # p3d2d = batch['p3d2d_keyframe']

        x, y, z, u, v = p3d2d[..., 0], p3d2d[..., 1], p3d2d[..., 2], p3d2d[..., 3], p3d2d[..., 4]
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)

        raw0 = torch.stack((zero, zero, zero, zero, -x, -y, -z, -one, v * x, v * y, v * z, v), dim=-1)
        raw1 = torch.stack((x, y, z, one, zero, zero, zero, zero, -u * x, -u * y, -u * z, -u), dim=-1)
        raw2 = torch.stack((-v * x, -v * y, -v * z, -v, u * x, u * y, u * z, u, zero, zero, zero, zero), dim=-1)
        weighted_raws = raw0 * weights, raw1 * weights, raw2 * weights

        vandermonde = torch.cat(weighted_raws, dim=-2)
        projection_matrices = []
        singular_values = []

        for i in range(p3d2d.shape[0]):
            svd = vandermonde[i].svd()
            projection_matrices.append(svd.V[:, -1].view(3, 4))
            singular_values.append(svd.S)
        
        pose_raw = torch.stack(projection_matrices, dim=0)
        if 'norm_mat_3d' in batch and 'denorm_mat_2d' in batch:
            pose_norm = pose_raw @ batch['norm_mat_3d'] # batch['denorm_mat_2d'] @ 
        else:
            pose_norm = pose_raw
        # pose_norm = pose_raw

        return {
            'pose_matrix_raw': pose_raw,
            'pose_matrix_denormalized': pose_norm ,
            'singular_values_vand': torch.stack(singular_values, dim=0),
        }

class SO3Projector(Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose_mat = batch['pose_matrix_denormalized']
        batch_size = pose_mat.shape[0]
        u_list, v_list, s_list = [], [], []
        eye = torch.eye(3, device=pose_mat.device).expand(batch_size, -1, -1)
        for i in range(batch_size):
            u, s, v = pose_mat[i, :3, :3].svd()
            u_sign = u.det().sign()
            v_sign = v.det().sign()
            uv_sign = (u_sign*u @ (v_sign*v.T)).det().sign()
            eye[i, 2, 2] = uv_sign
            u_list.append(u*u_sign)
            v_list.append(v*v_sign)
            s_list.append(s*u_sign*v_sign)
        
        pose_u = torch.stack(u_list, dim=0)
        pose_v = torch.stack(v_list, dim=0)
        pose_s = torch.stack(s_list, dim=0)
        pose_proper = torch.eye(4, device=pose_mat.device).repeat(batch_size, 1, 1)

        pose_proper[:, :3, :3] = (pose_u @ eye @ pose_v.transpose(-1, -2)).transpose(-2, -1)
        # pose_proper[:, :3, 3:4] = -pose_proper[:, :3, :3] @ (pose_mat[:, :3, 3] / pose_s.mean(dim=-1, keepdim=True)).unsqueeze(-1)
        pose_proper[:, :3, 3:4] = -pose_proper[:, :3, :3] @ (math.sqrt(3) * pose_mat[:, :3, :3].det().sign().unsqueeze(-1)* pose_mat[:, :3, 3] / torch.norm(pose_mat[:, :3, :3], dim=(-2, -1)).unsqueeze(-1)).unsqueeze(-1)
        return {
            'pose_matrix_estimate': pose_proper
            # 'singular_values_pose': 1 / pose_s.mean(dim=-1, keepdim=True),
            # 'scale': math.sqrt(3) * pose_mat[:, :3, :3].det().sign().unsqueeze(-1) / torch.norm(pose_mat[:, :3, :3], dim=(-2, -1)).unsqueeze(-1)
        }


class PnPSolver(Module):
    def __init__(self):
        super().__init__()

    def forward(self, p3d2d: torch.Tensor, p3d2d_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, y, z, u, v = p3d2d[..., 0], p3d2d[..., 1], p3d2d[..., 2], p3d2d[..., 3], p3d2d[..., 4]
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)

        raw0 = torch.stack((zero, zero, zero, zero, -x, -y, -z, -one, v * x, v * y, v * z, v), dim=-1)
        raw1 = torch.stack((x, y, z, one, zero, zero, zero, zero, -u * x, -u * y, -u * z, -u), dim=-1)
        raw2 = torch.stack((-v * x, -v * y, -v * z, -v, u * x, u * y, u * z, u, zero, zero, zero, zero), dim=-1)
        raws = raw0, raw1, raw2
        
        vandermonde = torch.cat(raws, dim=-2)

        svd = vandermonde.svd()
        
        projection_matrix = svd.V[:, -1].view(3, 4)
        
        x, y, z, u, v = p3d2d_all[..., 0], p3d2d_all[..., 1], p3d2d_all[..., 2], p3d2d_all[..., 3], p3d2d_all[..., 4]
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)

        raw0 = torch.stack((zero, zero, zero, zero, -x, -y, -z, -one, v * x, v * y, v * z, v), dim=-1)
        raw1 = torch.stack((x, y, z, one, zero, zero, zero, zero, -u * x, -u * y, -u * z, -u), dim=-1)
        raw2 = torch.stack((-v * x, -v * y, -v * z, -v, u * x, u * y, u * z, u, zero, zero, zero, zero), dim=-1)
        raws = raw0, raw1, raw2
        
        vandermonde = torch.cat(raws, dim=-2)
        
        algebraic_vector = vandermonde @ svd.V[:, -1]

        # print('Algebraic vector: ', algebraic_vector)
        
        algebraic_values = torch.norm(algebraic_vector.reshape(3, -1), dim=0)

        return projection_matrix, algebraic_values, vandermonde

class REPnP(Module):
    @configure
    def __init__(self, delta_max: float = 0.017):
        super().__init__()
        self.add_module('pnp_solver', PnPSolver())
        self.delta_max = delta_max

    def forward(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch['p3d2d_keyframe'].shape[0]
        num_points = batch['p3d2d_keyframe'].shape[1]
        quartile_bound = num_points//10
        if 'weights_3d2d' in batch:
            weights = batch['weights_3d2d']>0.1
        else:
            weights = torch.ones((batch_size, num_points), dtype=batch['p3d2d_keyframe'].dtype, device=batch['p3d2d_keyframe'].device)>0.5
        weights_repnp = torch.zeros((1, num_points), device=batch['p3d2d_keyframe'].device)
        for i in range(batch_size):
            psi = 1e4
            p3d2d = batch['p3d2d_n'][i]
            weights_batch = weights[i]
            p3d2d_pnp = p3d2d[weights_batch].clone()
            p3d2d = p3d2d.clone()
            k=0
            while k<50:
                k+=1
                # print('Number of points: ', p3d2d_pnp.shape[0])
                # print('Iteration: ', k)
                projection_matrix, algebraic_values, vand = self.pnp_solver(p3d2d_pnp, p3d2d)
                eps_max = algebraic_values[torch.argsort(algebraic_values)[quartile_bound].item()]
                # print('Eps_max: ', eps_max)
                if eps_max >= psi:
                    if i==0:
                        projection_matrices = projection_matrix.unsqueeze(0)
                    else:
                        projection_matrices = torch.cat((projection_matrices, projection_matrix.unsqueeze(0)))
                    # print('***********CONVERGED AFTER ', k, ' ITERTATIONS**********')
                    weights_repnp = torch.cat((weights_repnp, weights_batch.float().unsqueeze(0)))
                    break
                else:
                    psi = eps_max
                # print('Psi: ', psi)
                if eps_max>self.delta_max:
                    max_value = eps_max
                else:
                    max_value = self.delta_max
                # print('Max value: ', max_value)
                weights_batch = algebraic_values<max_value
                p3d2d_pnp = p3d2d[weights_batch].clone()

                # print('Algebraic values: ', algebraic_values)
                # print('Weights batch: ', weights_batch)

        if 'norm_mat_3d' in batch and 'denorm_mat_2d' in batch:
            pose_norm = projection_matrices @ batch['norm_mat_3d'] # batch['denorm_mat_2d'] @ 
        else:
            pose_norm = projection_matrices   
        
        return {'pose_matrix_raw': projection_matrices,
                'pose_matrix_denormalized': pose_norm,
                'weights_3d2d': weights_repnp[1:]}





