from typing import Dict, Tuple, List

import torch
from torch.nn import Module

from .inlier_loss import InlierLoss
from .algebraic_loss import PnPLastSVLoss, PnPLastSVsRatioLoss
from .reprojection_loss import ReprojectionLoss
from .supervised_inlier_loss import SupervisedInlierLoss
from .supervised_pose_loss import SupervisedPoseLoss, AuxiliarySupervisedPoseLoss
from .self_supervised_inlier_loss import SelfSupervisedInlierLoss
from .self_supervised_pose_loss import SelfSupervisedPoseLoss
from .self_supervised_inlier_normal_loss import SelfSupervisedInlierNormalLoss
from .filter_correction_loss import CorrectionL2loss, CorrectionMSEloss,SupervisedCorrectionMSEloss,SupervisedCorrectionTrRotloss
from ...parameters.parameter_config import configure

class TotalLoss(Module):
    __constants__ = ['weights']

    @configure
    def __init__(self, loss_weights: Dict[str, float], device: torch.device, tag: str):
        super().__init__()
        self.loss_names = []
        self.loss_functions = []
        self.contributes_to_final_loss = []
        
        self.tag = tag
        self.intrinsic_matrix = None
        self._device = device
        weights = []
        loss_criteria = {
            'last-sv-loss': lambda: PnPLastSVLoss().to(device),
            'last-svs-ratio-loss': lambda: PnPLastSVsRatioLoss().to(device),
            'reprojection-loss': lambda: ReprojectionLoss().to(device),
            'inlier-loss': lambda: InlierLoss().to(device), 
            'supervised-inlier-loss': lambda: SupervisedInlierLoss().to(device),
            'supervised-pose-loss': lambda: SupervisedPoseLoss().to(device),
            'auxiliary-supervised-pose-loss': lambda: AuxiliarySupervisedPoseLoss().to(device),
            'self-supervised-inlier-loss': lambda: SelfSupervisedInlierLoss().to(device),
            'self-supervised-pose-loss': lambda: SelfSupervisedPoseLoss().to(device),
            'self-supervised-inlier-normal-loss': lambda: SelfSupervisedInlierNormalLoss().to(device), 
            'kalman-correction-l2-loss': lambda: CorrectionL2loss().to(device),
            'kalman-correction-mse-loss': lambda: CorrectionMSEloss().to(device),
            'supervised-kalman-correction-mse-loss': lambda: SupervisedCorrectionMSEloss().to(device),
            'supervised-kalman-correction-tr-rot-loss': lambda: SupervisedCorrectionTrRotloss().to(device)
        }

        for loss_name, weight in loss_weights.items():
            if loss_name in loss_criteria:
                self.loss_names.append(loss_name)
                self.loss_functions.append(loss_criteria[loss_name]())
                self.contributes_to_final_loss.append(weight > 0.0)
                if weight > 0.0:
                    weights.append(weight)
            else:
                raise Exception('Unknown loss criterion "{:s}" supported loss criteria are {:}'
                                .format(loss_name, list(loss_criteria.keys())))
        
        if len(weights) == 0:
            raise Exception('At least one loss criterion in loss_weights needs to have non zero weight.')

        self.weights = torch.tensor(weights, dtype=torch.get_default_dtype(), device=device)

    def forward(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        final_losses = []
        for func, contributes, name in zip(self.loss_functions, self.contributes_to_final_loss, self.loss_names):
            loss = func(x, y)
            losses[name] = loss.detach()
            if contributes:
                final_losses.append(loss)

        final_loss = (torch.stack(final_losses) * self.weights).sum()
        losses['final_loss'] = final_loss.detach()
        return final_loss, losses

    def reinitialize_for_new_sequence(self, intrinsic_matrix):
        del self.intrinsic_matrix
        self.register_buffer('intrinsic_matrix', intrinsic_matrix)
        
        if self.tag == 'kitti':
            for loss_func, loss_name in zip(self.loss_functions, self.loss_names):
                if loss_name in {'reprojection-loss'}:
                    loss_func.set_intrinsic_matrix(
                        intrinsic_matrix=torch.nn.functional.pad(intrinsic_matrix, [0, 1]).unsqueeze(0)
                    )