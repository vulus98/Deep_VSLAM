from typing import Dict, List, Tuple
from torch.distributions import Bernoulli, Uniform, Normal
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
from scipy.spatial.transform import Rotation
from kornia import convert_points_to_homogeneous, convert_points_from_homogeneous
import cv2 as cv

from .iterable_data_generator import IterableDataGenerator
from ..parameters.parameter_config import configure

class SyntheticTrajectoryCorrespondences(IterableDataGenerator):
    @configure
    def __init__(self,  k_matrix: torch.Tensor, image_dim: List[int], n_correspondences: int, 
                 outlier_probability: float, min_noise_std: float, max_noise_std: float, 
                 max_n_samples: int, min_outlier_offset: float, x_range: float, y_range: float, 
                 z_range: float, n_map_features: int,  n_trajectory_points: int, n_frames: int, normalize: bool = False, visualize: bool = False):
        super().__init__(n_correspondences, image_dim, k_matrix, normalize, visualize)
        self._max_n_samples = max_n_samples
        self._min_outlier_offset = min_outlier_offset
        self._outlier_distribution = Bernoulli(torch.tensor(outlier_probability))
        self._noise_std_distribution = Uniform(min_noise_std, max_noise_std)
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range
        self._n_map_features = n_map_features
        self._n_trajectory_points = n_trajectory_points
        self._n_frames = n_frames
        self._trajectory_samples = torch.zeros((self._n_frames, 6))
        self.feature_map = torch.zeros((self._n_map_features, 3))

    def _sample_feature_map(self):
        self.feature_map[:, 0] = (2*torch.rand((self._n_map_features,))-1)*self._x_range
        self.feature_map[:, 1] = (2*torch.rand((self._n_map_features,))-1)*self._y_range
        self.feature_map[:, 2] = (2*torch.rand((self._n_map_features,))-1)*self._z_range

    def _generate_trajectory(self):
        ctrl_points = torch.zeros((self._n_trajectory_points, 6))
        ctrl_points[:, 0] = torch.sort(2*torch.rand(self._n_trajectory_points)-1)[0]*self._x_range/4.0
        ctrl_points[:, 1] = torch.sort(2*torch.rand(self._n_trajectory_points)-1)[0]*self._y_range/4.0
        ctrl_points[:, 2] = torch.sort((2*torch.rand(self._n_trajectory_points)-1)*0.05*self._z_range)[0]
        ctrl_points[:, 3] = (torch.rand(self._n_trajectory_points)*2-1)*np.pi*0.9
        ctrl_points[:, 4] = (torch.rand(self._n_trajectory_points)*2-1)*np.pi/2*0.9
        ctrl_points[:, 5] = (torch.rand(self._n_trajectory_points)*2-1)*np.pi*0.9

        ctrl_points_arr = torch.zeros_like(ctrl_points)
        ctrl_points_arr[:, :3] = ctrl_points[:, :3]
        # _, ind = NearestNeighbors(n_neighbors=ctrl_points.shape[0], p = 2).fit(ctrl_points[:,:2]).kneighbors(ctrl_points[:,:2])
        # ctrl_points_arr[:,:2] = ctrl_points[ind[0],:2]
        _, ind = NearestNeighbors(n_neighbors=ctrl_points.shape[0], p = 2).fit(ctrl_points[:,3:]).kneighbors(ctrl_points[:,3:])
        ctrl_points_arr[:,3:] = ctrl_points[ind[0],3:]

        tck_t, _ = interpolate.splprep(ctrl_points_arr[:, :3].transpose(-1, -2), s=500)
        tck_r, _ = interpolate.splprep(ctrl_points_arr[:, 3:].transpose(-1, -2), s=500)
        self.spline_t = torch.from_numpy(np.concatenate(tck_t[1]).astype('float32'))
        self.spline_r = torch.from_numpy(np.concatenate(tck_r[1]).astype('float32'))

        u_fine = torch.linspace(0, 1, self._n_frames)
        self._trajectory_samples[:,0], self._trajectory_samples[:,1], self._trajectory_samples[:,2] = [torch.from_numpy(item) for item in interpolate.splev(u_fine, tck_t)]
        self._trajectory_samples[:,3], self._trajectory_samples[:,4], self._trajectory_samples[:,5] = [torch.from_numpy(item) for item in interpolate.splev(u_fine, tck_r)]

    def _add_noise(self, p2d: torch.Tensor) -> torch.Tensor:
        noise_std = self._noise_std_distribution.sample((1,)).item()
        noise_distribution = Normal(loc=0, scale=noise_std)
        p2d += noise_distribution.sample((p2d.shape[0], 2))
        return p2d

    def _add_outliers(self, p2d: torch.Tensor) -> torch.Tensor:
        outlier_mask = self._outlier_distribution.sample((p2d.shape[0],)).bool()
        n_outliers = outlier_mask.sum()
        p2d[outlier_mask] = torch.rand(n_outliers, 2) * \
                            torch.tensor(self._image_dim, dtype=torch.get_default_dtype())
        return p2d, ~outlier_mask

    def _invert_rot_transl(self, tr: torch.Tensor) -> torch.Tensor:
        tr_inv = torch.eye(4)
        tr_inv[:3, :3] = tr[:3, :3].transpose(-1, -2)
        tr_inv[:3, 3] = -tr[:3, :3].transpose(-1, -2) @ tr[:3, 3]
        return tr_inv

    def _next_correspondences(self) -> Dict:
        if self._index >= self._max_n_samples:
            raise StopIteration()
        frame = self._index%self._n_frames
        if not frame:
            self._sample_feature_map()  
            self._generate_trajectory()
        current_frame = self._trajectory_samples[frame]
        pose_wc = torch.eye(4)
        pose_cw = torch.eye(4)
        pose_wc[:3, :3] = torch.from_numpy(Rotation.from_euler('xyz', current_frame[3:]).as_matrix())
        pose_wc[:3, 3] = current_frame[:3]
        pose_cw = self._invert_rot_transl(pose_wc)
        p3d_h = convert_points_to_homogeneous(self.feature_map)
        p2d_h = (self._k_matrix @ pose_cw[:3, :] @ p3d_h.T).T
        
        p2d = convert_points_from_homogeneous(p2d_h)
        mask =  (p2d_h[:, 2] > 0) & (p2d[:, 0] >= 0) & (p2d[:, 0] <= self._image_dim[0] - 1) & \
                (p2d[:, 1] >= 0) & (p2d[:, 1] <= self._image_dim[1] - 1)

        p2d, inliers = self._add_outliers(p2d[mask])
        p2d = self._add_noise(p2d)

        mask2 = (p2d[:, 0] >= 0) & (p2d[:, 0] <= self._image_dim[0] - 1) & \
                (p2d[:, 1] >= 0) & (p2d[:, 1] <= self._image_dim[1] - 1)

        inliers = inliers[mask2]
        p3d2d = torch.cat((p3d_h[mask][mask2, :3], p2d[mask2]), dim=-1)

        image = torch.ones((self._image_dim[1], self._image_dim[0]), 
                           dtype=torch.uint8) * 255

        if (frame > 1):
            current_velocity = self._trajectory_samples[frame-1] - self._trajectory_samples[frame-2]
        else:
            current_velocity = torch.zeros((6,))                       
        self._index += 1

        return {'p3d2d': p3d2d, 'image': image, 'inliers': inliers, 'camera_pose': pose_wc, 'camera_pose_vec': current_frame,
                'velocity': current_velocity, 'feature_mask': mask, 'first_frame': not frame, 'spline_t': self.spline_t, 'spline_r': self.spline_r} 
    
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

    def _normalize_correspondences(self, points: Dict) -> Dict:
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

    def _visualize_correspondences(self, points: Dict):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self._trajectory_samples.numpy()[:, 0], 
                self._trajectory_samples.numpy()[:, 1], 
                self._trajectory_samples.numpy()[:, 2], 
                label='trajectory', color='black')
        features = points['p3d2d'].numpy()[:, :3]
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c='green')
        features = self.feature_map.numpy()[~points['feature_mask']]
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c='red')
#        features = self.feature_map.numpy()[points['feature_mask'][0]][~points['feature_mask'][1]]
#        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c='red')
        pose = points['camera_pose'].numpy()
        ax.quiver(pose[0, 3], pose[1, 3], pose[2, 3], 10*pose[0, 2], 10*pose[1, 2], 10*pose[2, 2], color = 'blue')
        ax.quiver(pose[0, 3], pose[1, 3], pose[2, 3], 10*pose[0, 1], 10*pose[1, 1], 10*pose[2, 1], color = 'red')
        ax.quiver(pose[0, 3], pose[1, 3], pose[2, 3], 10*pose[0, 0], 10*pose[1, 0], 10*pose[2, 0], color = 'green')
        plt.show()
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
