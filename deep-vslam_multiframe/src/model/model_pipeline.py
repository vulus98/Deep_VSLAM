from time import time_ns
from typing import List, Dict
import torch
from torch._C import device
from torch.cuda import Event
from torch.nn import Module
from torch.jit import ScriptModule

from .inlier_outlier.outlier_network import PointNetInlierDetector3D2D
from .pose.direct_linear_transform import DLTPoseSolver, SO3Projector, REPnP
from .pose.kalman_filter import KalmanFilter
from .pose.filter_correction_lstm import LSTM
from .pose.filter_correction_transformer import TransformerCorrection
from .pose.pose_matrix_to_vector import PoseMatToVec, PoseVecToMat
from .pose.opencv_pose import PnPOpenCVSolver
from ..features.normalize_3d2d_correspondences import CorrespondenceNormalizer
from ..features.update_key_frame import UpdatePointsToKeyFrame, ReversePoseToWorldFrame
from ..features.normalize_trajectory import TrajectoryNormalizer
from ..parameters.parameter_config import configure

class ModelPipeline:
    @configure
    def __init__(self, pipeline_steps: List[str], device: torch.device, tag: str, enable_timing: bool = False, training_mode: bool = False,
                 train_outlier_model: bool = False, train_correction_model: bool = False, multiframe_size: int = 5, 
                 batch_size: int=100, supervised_training : bool = True):
        available_steps = {
            'change-keyframe': lambda: UpdatePointsToKeyFrame().to(device),
            'normalize-correspondences': lambda: CorrespondenceNormalizer().to(device),
            'point-net-outlier-detector': lambda: PointNetInlierDetector3D2D(enable_timing).to(device),
            'dlt-pose-solver': lambda: DLTPoseSolver().to(device),
            'so3-matrix-projector': lambda: SO3Projector().to(device),
            'pose-matrix-to-vector': lambda: PoseMatToVec().to(device),
            'back-to-world-frame': lambda: ReversePoseToWorldFrame().to(device),
            'kalman-filter': lambda: KalmanFilter().to(device),
            'normalize-trajectory': lambda: TrajectoryNormalizer().to(device),
            'lstm-correction': lambda: LSTM().to(device),
            'transformer-correction': lambda: TransformerCorrection().to(device),
            'pose-vector-to-mat': lambda: PoseVecToMat().to(device),
            'opencv-pose-solver': lambda: PnPOpenCVSolver(),
            'repnp-pose-solver': lambda: REPnP().to(device)
        }
        #We need to figure out where to pass the intrinsic matrices of the current sequence. (in TrainEngine/ModelPipeline)
        self.intrinsic_matrix = None
        self.pose_matrix_keyframe = None 
        self._steps = []
        self._step_names = []
        self.outlier_models = []
        self.outlier_model_names = []
        self.kalman_correction_models = []
        self.kalman_correction_model_names = []
        self.training_mode = training_mode
        self._train_outlier_model = train_outlier_model
        self._train_correction_model = train_correction_model
        self.supervised_training=supervised_training
        self.enable_timing = enable_timing
        self._device = device
        self.batch_size=batch_size
        self.pose_matrix_keyframe = torch.eye(4, device=device)
        self.multiframe_size=multiframe_size
        self.pose_matrices_past=[]
        for i in range(0,self.multiframe_size-1):
            self.pose_matrices_past.append(torch.zeros((4,4),device=device))
        self.pose_matrices_full=[]
        for i in range(0,self.multiframe_size):
            self.pose_matrices_full.append(torch.zeros((4,4),device=device))
        self.tag = tag

        for step_name in pipeline_steps:
            if not step_name in available_steps: 
                raise Exception('unknown pipeline step {:s}, available steps are {:}'.format(step_name, list(available_steps.keys())))
            else:
                self._step_names.append(step_name)
                step = available_steps[step_name]()
                if step_name in {'point-net-outlier-detector'}:
                    self.outlier_models.append(step)
                    self.outlier_model_names.append(step_name)
                elif step_name in {'lstm-correction', 'transformer-correction'}:
                    self.kalman_correction_models.append(step)
                    self.kalman_correction_model_names.append(step_name)
                self._steps.append(step)

    def train(self):
        self.training_mode = True
        for step, step_name in zip(self._steps, self._step_names):
            if isinstance(step, Module) or isinstance(step, ScriptModule):
                if step_name in {'point-net-outlier-detector'}: #'dlt-pose-solver', 'so3-matrix-projector'
                    step.train(self._train_outlier_model)
                elif step_name in {'lstm-correction', 'transformer-correction'}:
                    step.train(self._train_correction_model)
                # else:
                #     step.eval()
        return self

    def eval(self):
        self.training_mode = False
        for step, step_name in zip(self._steps, self._step_names):
            if isinstance(step, Module) or isinstance(step, ScriptModule):
                if step_name in {'point-net-outlier-detector'}: #'dlt-pose-solver', 'so3-matrix-projector'
                    step.eval()
                elif step_name in {'lstm-correction', 'transformer-correction'}:
                    step.eval()
                else:
                    step.eval()
        return self

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = batch
        batch_size = next(tens.shape[0] for tens in batch.values() if torch.is_tensor(tens))

        if self.enable_timing:
            time_start = time_ns()
            self._exec(result, batch_size)
            result['exec_time'] = torch.tensor((time_ns() - time_start) * 1e-6 / batch_size)
        else:
            self._exec(result, batch_size)

        return result

    def _exec(self, result, batch_size):
        result['p3d2d_keyframe']=torch.flatten(result['p3d2d_keyframe'],0,1)
        for step, step_name in zip(self._steps, self._step_names):
            if self.enable_timing:
                start_time = time_ns()
            if step_name in {'point-net-outlier-detector', 'dlt-pose-solver', 'so3-matrix-projector'}:
                with torch.autograd.set_grad_enabled(self.training_mode and self._train_outlier_model):
                    result.update(step(result))
            elif step_name in {'lstm-correction', 'transformer-correction'}:
                with torch.autograd.set_grad_enabled(self.training_mode and self._train_correction_model):
                    result.update(step(result))
            else:
                with torch.autograd.set_grad_enabled(True):
                    if step_name in {'change-keyframe', 'back-to-world-frame'}:
                        result.update(step(result, self.pose_matrix_keyframe,self.pose_matrices_past,self.pose_matrices_full))
                    else:
                        result.update(step(result))
            if self.enable_timing:
                result['exec_time_' + step_name] = torch.tensor((time_ns() - start_time) * 1e-6 / batch_size)

        if(self._train_outlier_model and self.supervised_training):
            self.pose_matrix_keyframe = result['poses'][-1]
            if(len(result['poses'])==self.batch_size):
                for i in range(0,self.multiframe_size//2):
                    self.pose_matrices_past[i]=result['poses'][-(self.multiframe_size//2)+i]
                # for i in range(0,self.multiframe_size):
                #     self.pose_matrices_full[i]=result['poses'][-self.multiframe_size+i]
        else: 
            self.pose_matrix_keyframe = result['pose_matrix_estimate_wf'][-1]
            if(len(result['pose_matrix_estimate_wf'])==self.batch_size):
                for i in range(0,self.multiframe_size//2):
                    self.pose_matrices_past[i]=result['pose_matrix_estimate_wf'][-(self.multiframe_size//2)+i]
                # for i in range(0,self.multiframe_size):
                #     self.pose_matrices_full[i]=result['pose_matrix_estimate_wf'][-self.multiframe_size+i]



    def reinitialize_for_new_sequence(self, intrinsic_matrix):
        if self.tag == 'kitti':
            self.set_pose_matrix_keyframe(torch.eye(4, dtype=torch.float32, device=self._device))
            self.set_intrinsic_matrix(intrinsic_matrix)
            self.set_pose_matrices_past()
            self.set_pose_matrices_full()
            for step, step_name in zip(self._steps, self._step_names):
                if step_name in {'normalize-correspondences', 'opencv-pose-solver'}:
                    step.set_intrinsic_matrix(
                        intrinsic_matrix=intrinsic_matrix
                    )
                if step_name in {'kalman-filter'}:
                    step.set_covariance_matrix_pred(
                        covariance_matrix_pred=100*torch.eye(12, dtype = torch.float, device=self._device)
                    )
                if step_name in {'pose-vector-to-mat'}:
                    step.set_pose_matrix_keyframe(
                        pose_matrix_keyframe=torch.eye(4, device=self._device)
                    )
                    step.set_pose_matrix_keyframe_kalman_estimate(
                        pose_matrix_keyframe_kalman_estimate=torch.eye(4, device=self._device)
                    )
                    step.set_pose_matrix_keyframe_kalman_prediction(
                        pose_matrix_keyframe_kalman_prediction=torch.eye(4, device=self._device)
                    )
        else:
            raise NotImplementedError

    def set_intrinsic_matrix(self, intrinsic_matrix: torch.Tensor):
        self.intrinsic_matrix = intrinsic_matrix

    def set_pose_matrix_keyframe(self, pose_matrix_keyframe: torch.Tensor):
        self.pose_matrix_keyframe = pose_matrix_keyframe

    def set_pose_matrices_past(self):
        self.pose_matrices_past=[]
        for i in range(0,self.multiframe_size-1):
            self.pose_matrices_past.append(torch.zeros((4,4),device=self._device))
    

    def set_pose_matrices_full(self):
        self.pose_matrices_full=[]
        for i in range(0,self.multiframe_size):
            self.pose_matrices_full.append(torch.zeros((4,4),device=self._device))
