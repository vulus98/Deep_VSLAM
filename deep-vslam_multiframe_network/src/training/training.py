from typing import Dict, List, Union
import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler, MultiplicativeLR, MultiStepLR, StepLR, ExponentialLR
from torch.utils.data.dataloader import DataLoader
# from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm, trange
import wandb
import math
import numpy as np

from ..model.model_pipeline import ModelPipeline
from ..model.loss.total_loss import TotalLoss
from ..parameters.parameter_config import configure


class TrainEngine:
    @configure
    def __init__(self, model_pipeline: ModelPipeline, loss: TotalLoss, train_sequences: DataLoader, 
                 test_sequences: DataLoader, num_epochs: int, batch_size: int, lr_start: float, 
                 train_outlier_model: bool, train_correction_model: bool, train_pose_outlier_model: bool,tag: str, 
                 log_batch_period: int = 32, save_model_epoch_period: int = 16, write_path: str = './'):
        self.model_pipeline = model_pipeline.train()
        self.loss = loss
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.optimizers = []
        self.lr_schedulers = []
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self.lr_start = lr_start
        self._train_outlier_model = train_outlier_model
        self._train_correction_model = train_correction_model
        self._train_pose_outlier_model = train_pose_outlier_model
        self.tag = tag
        self.log_batch_period = log_batch_period
        self.save_model_period = save_model_epoch_period
        self.write_path = write_path

    def train_model(self):
        if self._train_outlier_model:
            for model in self.model_pipeline.outlier_models:
                self.optimizers.append(get_configured_optimizer(model))
                self.lr_schedulers.append(get_configured_lr_scheduler(self.optimizers[-1]))
                wandb.watch(model, log='all', log_freq=10)
        elif self._train_correction_model:
            for model in self.model_pipeline.kalman_correction_models:
                self.optimizers.append(get_configured_optimizer(model))
                self.lr_schedulers.append(get_configured_lr_scheduler(self.optimizers[-1]))
                wandb.watch(model, log='all', log_freq=10)
        elif self._train_pose_outlier_model:
            for model in self.model_pipeline.pose_outlier_models:
                self.optimizers.append(get_configured_optimizer(model))
                self.lr_schedulers.append(get_configured_lr_scheduler(self.optimizers[-1]))
                wandb.watch(model, log='all', log_freq=10)
        self.loss.train()
        example_num = 0
        min_loss = 1e5
        for epoch in trange(self._num_epochs):
            for seq_dataloader in self.train_sequences.sequences:
                # TODO Set devices in a better way
                self.model_pipeline.reinitialize_for_new_sequence(
                    intrinsic_matrix=to_device(seq_dataloader.dataset.intrinsic_matrix)
                )
                self.loss.reinitialize_for_new_sequence(
                    intrinsic_matrix=to_device(seq_dataloader.dataset.intrinsic_matrix)
                )
                
                for batch_i, batch in enumerate(seq_dataloader):
                    example_num += batch['p3d2d_keyframe'].shape[0]
                    batch = to_device(batch)
                    # try: 
                    output = self.model_pipeline(batch)
                    with torch.autograd.set_grad_enabled(True):
                        total_loss, losses = self.loss(output, output)
                    self.optimizers[0].zero_grad()
                    total_loss.backward()
                    if(self._train_correction_model):
                        torch.nn.utils.clip_grad_norm_(self.model_pipeline.kalman_correction_models[0].parameters(), 0.5)
                    self.optimizers[0].step()
                    if not batch_i % self.log_batch_period:
                        if self.lr_schedulers[0]:
                            lr = self.lr_schedulers[0].get_last_lr()[0]
                        else:
                            lr = self.lr_start
                        self._log_train_data(losses, example_num, epoch+1, lr)
                    # except:
                        # pass
                if self.lr_schedulers[0]:
                    self.lr_schedulers[0].step()
                if not (epoch+1) % self.save_model_period:
                    if self._train_outlier_model:
                        torch.save(self.model_pipeline.outlier_models[0].state_dict(), self.write_path+'outlier_network_state_dict_checkpoint.pth')
                        wandb.save(self.write_path+'outlier_network_state_dict_checkpoint.pth')
                        if losses['final_loss'] < min_loss:
                            torch.save(self.model_pipeline.outlier_models[0].state_dict(), self.write_path+'outlier_network_state_dict_best.pth')
                            wandb.save(self.write_path+'outlier_network_state_dict_best.pth')
                            min_loss = losses['final_loss']
                    elif self._train_correction_model:
                        torch.save(self.model_pipeline.kalman_correction_models[0].state_dict(), self.write_path+'kalman_correction_state_dict_checkpoint.pth')
                        wandb.save(self.write_path+'kalman_correction_state_dict_checkpoint.pth')
                        if losses['final_loss'] < min_loss:
                            torch.save(self.model_pipeline.kalman_correction_models[0].state_dict(), self.write_path+'kalman_correction_state_dict_best.pth')
                            wandb.save(self.write_path+'kalman_correction_state_dict_best.pth')
                            min_loss = losses['final_loss']
                    elif self._train_pose_outlier_model:
                        torch.save(self.model_pipeline.pose_outlier_models[0].state_dict(), self.write_path+'pose_outlier_network_state_dict_checkpoint.pth')
                        wandb.save(self.write_path+'outlier_network_state_dict_checkpoint.pth')
                        if losses['final_loss'] < min_loss:
                            torch.save(self.model_pipeline.pose_outlier_models[0].state_dict(), self.write_path+'pose_outlier_network_state_dict_best.pth')
                            wandb.save(self.write_path+'pose_outlier_network_state_dict_best.pth')
                            min_loss = losses['final_loss']

    def eval_model(self):
        self.model_pipeline.eval()
        self.loss.eval()
        for seq_dataloader in self.test_sequences.sequences:
            # TODO Set devices in a better way
            self.model_pipeline.reinitialize_for_new_sequence(
                intrinsic_matrix=to_device(seq_dataloader.dataset.intrinsic_matrix)
            )
            self.loss.reinitialize_for_new_sequence(
                intrinsic_matrix=to_device(seq_dataloader.dataset.intrinsic_matrix)
            )
            preds = {}
            for batch_i, batch in enumerate(seq_dataloader):
                # seq_num = 0
                # if self.tag == 'kitti':
                #     if batch['first_frame'][0]:
                #         if batch_num > 0:
                #             file.close()
                #         sequence = self.test_sequence_numbers[seq_num]
                #         seq_num = seq_num + 1
                #         file = open(self.write_path + sequence + '.txt', 'w')
                #         initial_position = np.reshape(np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1), (1, 12))
                #         np.savetxt(file, initial_position)
                batch = to_device(batch)
                output = self.model_pipeline(batch)
                # if self.tag == 'kitti':
                #     np.savetxt(file, np.reshape(output['pose_matrix_estimate_wf'][:, 0:3, :].cpu().numpy(), (-1, 1, 12)).squeeze(0))
                with torch.autograd.set_grad_enabled(False):
                    _, losses = self.loss(output, output)
                # if not batch_num % self.log_batch_period:
                self._log_test_data(losses, batch_i)

        # if self.tag == 'kitti':
        #     file.close()
        if self._train_outlier_model:
            torch.save(self.model_pipeline.outlier_models[0].state_dict(), self.write_path+'outlier_network_state_dict.pth')
            wandb.save(self.write_path+'outlier_network_state_dict.pth')
        elif self._train_correction_model:
            torch.save(self.model_pipeline.kalman_correction_models[0].state_dict(), self.write_path+'kalman_correction_state_dict.pth')
            wandb.save(self.write_path+'kalman_correction_state_dict.pth')
        elif self._train_pose_outlier_model:
            torch.save(self.model_pipeline.pose_outlier_models[0].state_dict(), self.write_path+'pose_outlier_network_state_dict.pth')
            wandb.save(self.write_path+'pose_outlier_network_state_dict.pth')
        # if self.tag == 'kitti':
        #     for sequence in self.test_sequence_numbers:
        #         wandb.save(self.write_path + sequence + '.txt')

    def _log_train_data(self, loss: Dict[str, torch.tensor], example_num: int, epoch_num: int, lr_value: float):
        for key, value in loss.items():
            loss[key] = float(value)
        loss['epoch'] = epoch_num
        loss['lr_value'] = lr_value
        wandb.log(loss, step=example_num)
        print('At epoch {:}, after {:} training examples in total, the loss is {:.8}'.format(epoch_num, example_num, loss['final_loss']))

    def _log_test_data(self, loss: Dict[str, torch.tensor], batch_num: int):
        loss_test = {}
        for key, value in loss.items():
            loss_test[key + '_test'] = float(value)
        wandb.log(loss_test)
        print('At batch {:}, averaged on {:} test examples, the loss is {:.8}'.format(batch_num, self._batch_size, loss['final_loss']))


@configure
def get_configured_lr_scheduler(optimizer: Optimizer, num_epochs: int, lr_start: float, lr_decay: float=None) -> Union[_LRScheduler, None]:
    if lr_decay is None:
        return None

    scheduler_steplr =  CosineAnnealingLR(
        optimizer,
        eta_min=lr_start * lr_decay,
        T_max=num_epochs-5,
    )
    # scheduler_steplr = StepLR(optimizer, step_size=2, gamma=0.1)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
    # return scheduler_warmup
    return scheduler_steplr

    #lmbda = lambda epoch: lr_decay if (epoch<100) else 1.0
    #return MultiplicativeLR(optimizer, lr_lambda=lmbda)

    # return MultiStepLR(optimizer, milestones=[100], gamma=lr_decay)

@configure
def get_configured_optimizer(model: Module, lr_start: float) -> Optimizer:
    # return Adam(model.parameters(), lr=lr_start)
    #AdamW weight decay
    return AdamW(model.parameters(), lr=lr_start, weight_decay=0.05)
    # return SGD(model.parameters(), lr_start)

@configure
def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    if isinstance(batch, List):
        for i, tens in enumerate(batch):
            batch[i] = tens.to(device)
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    elif isinstance(batch, Dict):
        for key, item in batch.items():
            batch[key] = item.to(device)
    else:
        raise Exception('Batch data has to be torch.tensor or List[torch.tensor] or Dict[str, torch.tensor].')
    return batch
    