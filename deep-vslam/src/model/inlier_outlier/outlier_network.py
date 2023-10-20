from typing import Tuple, Dict
from time import time_ns
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import nn

class PointNetFeatureTransform(Module):
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
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).expand(batch_size, -1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetInlierEstimator(Module):
    def __init__(self, point_dim=5):
        super().__init__()
        self.add_module("feature_transform1", PointNetFeatureTransform(point_dim))
        self.add_module("feature_transform2", PointNetFeatureTransform(64))
        self.conv1 = torch.nn.Conv1d(point_dim, 16, 1)
        self.conv2 = torch.nn.Conv1d(16, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)

        self.conv4 = torch.nn.Conv1d(64, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 128, 1)
        self.conv6 = torch.nn.Conv1d(128, 1024, 1)

        self.conv7 = torch.nn.Conv1d(1024 + 64, 512, 1)
        self.conv8 = torch.nn.Conv1d(512, 256, 1)
        self.conv9 = torch.nn.Conv1d(256, 128, 1)
        self.conv10 = torch.nn.Conv1d(128, 64, 1)
        self.conv11 = torch.nn.Conv1d(64, 1, 1)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(1024)

        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)
        self.bn10 = nn.BatchNorm1d(64)

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_points = points.shape[1]

        x = points.transpose(2, 1)

        transformation = self.feature_transform1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))

        transformation2 = self.feature_transform2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation2)
        x = x.transpose(2, 1)
        point_feat = x

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))

        x = torch.max(x, 2, keepdim=True)[0]
        global_features = x

        x = x.repeat_interleave(n_points, dim=2)
        combined_features = torch.cat((point_feat, x), dim=1)

        x = F.relu(self.bn7(self.conv7(combined_features)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        # x = F.relu(self.conv7(combined_features))
        # x = F.relu(self.conv8(x))
        # x = F.relu(self.conv9(x))
        # x = F.relu(self.conv10(x))

        x = torch.squeeze(self.conv11(x), dim=1)
        x = torch.sigmoid(x)
        return x, global_features


class PointNetInlierDetector3D2D(Module):
    def __init__(self, enable_timing: bool = False):
        super().__init__()
        self.add_module('point_net', PointNetInlierEstimator(point_dim=5))
        self._enable_timing = enable_timing
        self.time_spent = 0

    def forward(self, x: Dict) -> Dict:
        p3d2d_n = x['p3d2d_n']
        if self._enable_timing and p3d2d_n.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            inlier_weights, global_features = self.point_net(p3d2d_n)
            end.record()
            torch.cuda.synchronize()
            self.time_spent += start.elapsed_time(end)
        elif self._enable_timing:
            start = time_ns()
            inlier_weights, global_features = self.point_net(p3d2d_n)
            self.time_spent += (time_ns() - start) * 1e-6
        else:
            inlier_weights, global_features = self.point_net(p3d2d_n)

        return {'weights_3d2d': inlier_weights, 'global_features': global_features}