from typing import Dict

import cv2
import numpy as np
import torch
from kornia import convert_points_to_homogeneous
from torch.nn import Module


class EssentialMatrixSolver(Module):
    def __init__(self, k_matrix):
        super().__init__()
        self.register_buffer('_k_matrix', k_matrix)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p2d2d_n = batch['p2d2d_n']
        weights = batch['weights_2d2d']
        batch_size = p2d2d_n.shape[0]
        n_points = p2d2d_n.shape[1]

        x, y, u, v = p2d2d_n[..., 0], p2d2d_n[..., 1], p2d2d_n[..., 2], p2d2d_n[..., 3]
        vandermonde = torch.stack((
                                    x * u, 
                                    x * v,  
                                    x,  
                                    y * u,  
                                    y * v,  
                                    y,
                                    u, 
                                    v, 
                                    torch.ones(batch_size, n_points, device=p2d2d_n.device)
                                   ), dim=-1)

        weighted_vandermonde = weights.unsqueeze(-1) * vandermonde

        fundamental_matrix_list = []
        singular_value_list = []
        for i in range(batch_size):
            svd = weighted_vandermonde[i].svd()
            f_matrix = svd.V[:, -1].view(3, 3)
            singular_value_list.append(svd.S)
            fundamental_matrix_list.append(f_matrix)

        singular_values_vand = torch.stack(singular_value_list, dim=0)
        fundamental_matrix_raw = torch.stack(fundamental_matrix_list, dim=0)
        if 'norm_mat_2d' in batch and 'denorm_mat_2d' in batch:
            fundamental_matrix_denorm = batch['denorm_mat_2d'].transpose(-2, -1) @ fundamental_matrix_raw @ batch['norm_mat_2d']
        else:
            fundamental_matrix_denorm = fundamental_matrix_raw
        essential_matrix_denorm = self._k_matrix.transpose(-2, -1) @ fundamental_matrix_denorm @ self._k_matrix

        return {
            'fundamental_matrix_raw': fundamental_matrix_raw,
            'essential_matrix_denorm': essential_matrix_denorm,
            'singular_values_vand': singular_values_vand,
        }


class EssentialToPose(Module):
    def __init__(self, k_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer('_k_matrix', k_matrix)
        self.register_buffer('_k_inv', k_matrix.inverse())

    def triangulate_points(self, r, t, p1, p2):
        d_1 = p1.transpose(2, 1)
        d_2 = r.bmm(p2.transpose(2, 1))
        n_1 = torch.cross(d_1, torch.cross(d_2, d_1, dim=1), dim=1)
        n_2 = torch.cross(d_2, torch.cross(d_1, d_2, dim=1), dim=1)

        d_1, d_2 = d_1.transpose(2, 1), d_2.transpose(2, 1)
        n_1, n_2 = n_1.transpose(2, 1), n_2.transpose(2, 1)
        t = t.unsqueeze(1)
        # t=Bx1x3, d_1=BxNx3, n_1=BxNx3
        a_1 = (t * n_2).sum(dim=-1, keepdim=True) / (d_1 * n_2).sum(dim=-1, keepdim=True) * d_1
        a_2 = t + (-t * n_1).sum(dim=-1, keepdim=True) / (d_2 * n_1).sum(dim=-1, keepdim=True) * d_2
        p_3d = (a_1 + a_2) / 2

        return p_3d

    def n_points_in_front(self, r, t, p1, p2):
        p_3d = self.triangulate_points(r, t, p1, p2)
        p_3d_homogeneous = torch.cat((p_3d, torch.ones(p1.shape[:2] + (1,), device=p1.device)), dim=-1)
        front_cam_1 = p_3d[:, :, 2] > 0
        tf = torch.eye(4, device=p1.device).repeat(p1.shape[0], 1, 1)
        tf[:, :3, :3] = r
        tf[:, :3, 3] = t
        p_3d_2 = tf.inverse().bmm(p_3d_homogeneous.transpose(2, 1)).transpose(2, 1)
        front_cam_2 = p_3d_2[:, :, 2] > 0

        return (front_cam_1 & front_cam_2).sum(dim=1, dtype=torch.int32)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_essentials = x['essential_raw']
        batch_size = raw_essentials.shape[0]

        last_singular_value_list = []
        u_list = []
        v_list = []
        for i in range(batch_size):
            svd = raw_essentials[i].svd()
            u_list.append(svd.U * svd.U.det().sign())
            v_list.append(svd.V * svd.V.det().sign())
            last_singular_value_list.append(svd.S[-1])

        last_svs = torch.stack(last_singular_value_list, dim=0)
        u = torch.stack(u_list, dim=0)
        v = torch.stack(v_list, dim=0)
        w = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], requires_grad=False, device=raw_essentials.device).expand(batch_size, -1, -1)

        c2d2d = x['c2d2d']
        p1 = (self._k_inv @ convert_points_to_homogeneous(c2d2d[:, :, :2]).transpose(-2, -1)).transpose(-2, -1)
        p2 = (self._k_inv @ convert_points_to_homogeneous(c2d2d[:, :, 2:]).transpose(-2, -1)).transpose(-2, -1)

        # possible transformations
        r_1 = v @ w @ u.transpose(2, 1)
        r_2 = v @ w.transpose(2, 1) @ u.transpose(2, 1)
        t_proposed = v[:, :, 2]

        score_r1_t = self.n_points_in_front(r_1, t_proposed, p1, p2)
        score_r2_t = self.n_points_in_front(r_2, t_proposed, p1, p2)
        score_r1_mt = self.n_points_in_front(r_1, -t_proposed, p1, p2)
        score_r2_mt = self.n_points_in_front(r_2, -t_proposed, p1, p2)

        r = torch.where((score_r1_t + score_r1_mt > score_r2_t + score_r2_mt).unsqueeze(1).unsqueeze(2), r_1, r_2)
        t = torch.where((score_r1_t + score_r2_t > score_r1_mt + score_r2_mt).unsqueeze(1), t_proposed, -t_proposed)

        # TODO create projection matrix from r, t

        tf_ab = torch.eye(4).repeat(batch_size, 1, 1)
        tf_ab[:, :3, :3] = r
        tf_ab[:, :3, 3] = t / torch.sqrt((t ** 2).sum(dim=-1, keepdim=True))

        return {
            'essential_tf_ab': tf_ab,
            'essential_decompose_last_sv': last_svs,
        }