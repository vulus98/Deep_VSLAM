from typing import List, Tuple, Dict

import torch
from kornia import convert_points_to_homogeneous, unproject_points
from scipy.spatial.transform import Rotation
from torch.distributions import Bernoulli, Uniform, Normal
import cv2 as cv

from .iterable_data_generator import IterableDataGenerator
from ..parameters.parameter_config import configure

class SyntheticImageCorrespondences(IterableDataGenerator): 
    @configure
    def __init__(self, k_matrix: torch.Tensor, image_dim: List[int], n_correspondences: int,
                 outlier_probability: float, min_noise_std: float, max_noise_std: float, 
                 max_n_samples: int, min_outlier_offset: float, max_depth: float, normalize: bool = False, visualize:bool = False):
        super().__init__(n_correspondences, image_dim, k_matrix, normalize, visualize)
        self._max_n_samples = max_n_samples
        self._min_outlier_offset = min_outlier_offset
        self._max_depth = max_depth
        self._outlier_distribution = Bernoulli(torch.tensor(outlier_probability))
        self._noise_std_distribution = Uniform(min_noise_std, max_noise_std)
    
    def _next_correspondences(self) -> Dict[str, torch.Tensor]:
        if self._index >= self._max_n_samples:
            raise StopIteration()
        self._index += 1

        p3d, p2d = self._generate_3d2d()
        p2d, inliers = self._add_outliers(p2d)
        p2d = self._add_noise(p2d)

        tr = self._add_rot_transl()
        p3d_h = convert_points_to_homogeneous(p3d)
        p3d_h = (tr @ p3d_h.transpose(-2, -1)).transpose(-1, -2)

        mask = (p2d[:, 0] >= 0) & (p2d[:, 0] <= self._image_dim[0] - 1) & \
               (p2d[:, 1] >= 0) & (p2d[:, 1] <= self._image_dim[1] - 1)

        p3d2d = torch.cat((p3d_h[mask, :3], p2d[mask]), dim=-1)

        inliers = inliers[mask]
        image = torch.ones((self._image_dim[1], self._image_dim[0]), 
                           dtype=torch.uint8) * 255
                           
        return {'p3d2d': p3d2d, 'image': image, 'inliers': inliers, 'camera_pose': tr}  

    def _fixed_size(self, points: Dict[str, torch.Tensor], n_correspondences: int) -> Dict[str, torch.Tensor]:
        key = 'p3d2d'
        n_valid_points = points[key].shape[0]
        if n_valid_points < n_correspondences:
            resampled_indices = torch.randint(0, n_valid_points, (n_correspondences - n_valid_points,))
            dim = points[key].shape[1]

            new_points = torch.zeros((n_correspondences, dim), device=points[key].device)
            new_points[:n_valid_points] = points[key][:n_valid_points]
            new_points[n_valid_points:] = points[key][resampled_indices]
            points[key] = new_points

            inliers = torch.zeros(n_correspondences, dtype=points['inliers'].dtype, device=points['inliers'].device)
            inliers[:n_valid_points] = points['inliers'][:n_valid_points]
            inliers[n_valid_points:] = points['inliers'][resampled_indices]
            points['inliers'] = inliers
        else:
            points[key] = points[key][:n_correspondences]
            points['inliers'] = points['inliers'][:n_correspondences]
        return points

    def _normalize_correspondences(self, points: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3d2d = points['p3d2d'].detach().clone()

        mean3d = p3d2d[:, :3].mean(dim=0)
        p3d2d[:, :3] = p3d2d[:,:3]-mean3d
        scale3d = torch.div(1.0, p3d2d[:, :3].std(dim=0))
        p3d2d[:, :3] *= scale3d
        norm_mat_3d = torch.eye(4)
        norm_mat_3d[:3, :3] = torch.diag(scale3d)
        norm_mat_3d[:3, 3] = -mean3d*scale3d

        mean2d = p3d2d[:, 3:].mean(dim=0)
        p3d2d[:, 3:] = p3d2d[:, 3:]-mean2d
        scale2d = torch.div(1.0, p3d2d[:, 3:].std(dim=0))
        p3d2d[:, 3:] *= scale2d
        norm_mat_2d = torch.eye(3)
        norm_mat_2d[:2, :2] = torch.diag(scale2d).inverse()
        norm_mat_2d[:2, 2] = mean2d

        points['p3d2d_n'] = p3d2d
        points['norm_mat_3d'] = norm_mat_3d
        points['denorm_mat_2d'] = self._k_matrix.inverse() @ norm_mat_2d

        return points 

    def _visualize_correspondences(self, points: Dict[str, torch.Tensor]):
        disp = cv.cvtColor(points['image'].numpy(), cv.COLOR_GRAY2RGB)
        p3d_h = convert_points_to_homogeneous(points['p3d2d'][:, :3]).transpose(-1, -2)
        tr = self._invert_rot_transl(points['camera_pose']) 
        proj = self._k_matrix @ tr[:3, :] @ p3d_h
        correspondences = points['p3d2d'] 
        inliers = points['inliers']
        for i in range(correspondences.shape[0]):
            disp = cv.drawMarker(disp, tuple(correspondences[i, 3:].numpy()), (0, 0, 255), cv.MARKER_DIAMOND, 20, 2, 8)
            disp = cv.drawMarker(disp, (int((proj[0, i]/proj[2, i]).item()), int((proj[1, i]/proj[2, i]).item())), 
                                (0, 255, 0) if inliers[i] else (255, 0, 0),
                                cv.MARKER_SQUARE, 20, 2, 8)
        cv.imshow('Sample Image with Correspondences', disp)
        cv.waitKey(5000)

    def _generate_3d2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p2d = torch.rand(self._n_correspondences, 2) * \
              torch.tensor(self._image_dim, dtype=torch.get_default_dtype())
        z = (torch.rand(self._n_correspondences, 1) + 1.0) * self._max_depth / 2.0
        p3d = unproject_points(p2d, z, self._k_matrix)
        return p3d, p2d     

    def _add_noise(self, p2d: torch.Tensor) -> torch.Tensor:
        noise_std = self._noise_std_distribution.sample((1,)).item()
        noise_distribution = Normal(loc=0, scale=noise_std)
        p2d += noise_distribution.sample((self._n_correspondences, 2))
        return p2d

    def _add_outliers(self, p2d: torch.Tensor) -> torch.Tensor:
        outlier_mask = self._outlier_distribution.sample((self._n_correspondences,)).bool()
        n_outliers = outlier_mask.sum()
        p2d[outlier_mask] = torch.rand(n_outliers, 2) * \
                            torch.tensor(self._image_dim, dtype=torch.get_default_dtype())
        return p2d, ~outlier_mask

    def _add_rot_transl(self) -> torch.Tensor:
        tr = torch.eye(4)
        tr[:3, :3] = torch.tensor(Rotation.random().as_matrix())
        tr[:3, 3] = torch.normal(0, 1, size=(3,))
        return tr

    def _invert_rot_transl(self, tr: torch.Tensor) -> torch.Tensor:
        tr_inv = torch.eye(4)
        tr_inv[:3, :3] = tr[:3, :3].transpose(-1, -2)
        tr_inv[:3, 3] = -tr[:3, :3].transpose(-1, -2) @ tr[:3, 3]
        return tr_inv 