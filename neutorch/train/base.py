import os
import random
from abc import ABC, abstractproperty
from functools import cached_property
from time import time
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np
from yacs.config import CfgNode
from chunkflow.lib.cartesian_coordinate import Cartesian

import lightning.pytorch as pl

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from neutorch.data.dataset import worker_init_fn
from neutorch.data.patch import collate_batch
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.model.io import load_chkpt, log_tensor, save_chkpt
from neutorch.model.IsoRSUNet import Model
from neutorch.utils.log_utils import get_logger

logger = get_logger()


class TrainerBase(pl.Trainer):
    def __init__(self, cfg: CfgNode, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(cfg, str) and os.path.exists(cfg):
            with open(cfg) as file:
                cfg = CfgNode.load_cfg(file)
        cfg.freeze()
               
        self.cfg = cfg
        self.patch_size = Cartesian.from_collection(cfg.train.patch_size)

    @cached_property
    def batch_size(self):
        # return self.num_gpus * self.cfg.train.batch_size
        # this batch size is for a single GPU rather than the total number!
        return self.cfg.train.batch_size

    # @cached_property
    # def path_list(self):
    #     glob_path = os.path.expanduser(self.cfg.dataset.glob_path)
    #     path_list = glob(glob_path, recursive=True)
    #     path_list = sorted(path_list)
    #     logger.info(f'path_list \n: {path_list}')
    #     assert len(path_list) > 1
        
    #     # sometimes, the path list only contains the label without corresponding image!
    #     # assert len(path_list) % 2 == 0, \
    #         # "the image and synapses should be paired."
    #     return path_list

    # def _split_path_list(self):
    #     training_path_list = []
    #     validation_path_list = []
    #     for path in self.path_list:
    #         assignment_flag = False
    #         for validation_name in self.cfg.dataset.validation_names:
    #             if validation_name in path:
    #                 validation_path_list.append(path)
    #                 assignment_flag = True
            
    #         for test_name in self.cfg.dataset.test_names:
    #             if test_name in path:
    #                 assignment_flag = True

    #         if not assignment_flag:
    #             training_path_list.append(path)

    #     logger.info(f'split {len(self.path_list)} ground truth samples to {len(training_path_list)} training samples, {len(validation_path_list)} validation samples, and {len(self.path_list)-len(training_path_list)-len(validation_path_list)} test samples.')
    #     self.training_path_list = training_path_list
    #     self.validation_path_list = validation_path_list

    @cached_property
    def model(self):
        model = Model(self.cfg.model.in_channels, self.cfg.model.out_channels)
        if 'preload' in self.cfg.train:
            fname = self.cfg.train.preload
        else:
            fname = os.path.join(self.cfg.train.output_dir, 
                f'model_{self.cfg.train.iter_start}.chkpt')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if os.path.exists(fname) and self.local_rank==0:
            model = load_chkpt(model, fname)
        return model

    @cached_property
    @abstractproperty
    def training_dataset(self):
        pass

    @cached_property
    def training_data_loader(self):
        dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    
    @cached_property
    def validation_data_loader(self):
        dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    @cached_property
    def validation_data_iter(self):
        validation_data_iter = iter(self.validation_data_loader)
        return validation_data_iter

    @cached_property
    def voxel_num(self):
        return np.product(self.patch_size) * self.batch_size

    def label_to_target(self, label: torch.Tensor):
        return label.cuda()

    def post_processing(self, prediction: torch.Tensor):
        if isinstance(self.loss_module, BinomialCrossEntropyWithLogits):
            return torch.sigmoid(prediction)
        else:
            return prediction

    # def __call__(self) -> None:
    #     writer = SummaryWriter(log_dir=self.cfg.train.output_dir)
    #     accumulated_loss = 0.
    #     iter_idx = self.cfg.train.iter_start
    #     for image, label in self.training_data_loader:
    #         target = self.label_to_target(label)
    #
    #         iter_idx += 1
    #         if iter_idx > self.cfg.train.iter_stop:
    #             logger.info(f'exceeds the maximum iteration: {self.cfg.train.iter_stop}')
    #             return
    #
    #         ping = time()
    #         logger.debug(f'preparing patch takes {round(time()-ping, 3)} seconds')
    #         predict = self.model(image)
    #         # predict = self.post_processing(predict)
    #         loss = self.loss_module(predict, target)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         accumulated_loss += loss.tolist()
    #         logger.debug(f'iteration {iter_idx} takes {round(time()-ping, 3)} seconds.')
    #
    #         if iter_idx % self.cfg.train.training_interval == 0 and iter_idx > 0:
    #             per_voxel_loss = accumulated_loss / \
    #                 self.cfg.train.training_interval / \
    #                 self.voxel_num
    #             per_iter_seconds = round((time() - ping) / self.cfg.train.training_interval, 3)
    #
    #             logger.info(f'Iteration {iter_idx}: training loss {round(per_voxel_loss, 3)}, {per_iter_seconds} sec/iter')
    #             accumulated_loss = 0.
    #             predict = self.post_processing(predict)
    #             writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
    #             log_tensor(writer, 'train/image', image, 'image', iter_idx)
    #             log_tensor(writer, 'train/prediction', predict.detach(), 'image', iter_idx)
    #             log_tensor(writer, 'train/target', target, 'image', iter_idx)
    #
    #         if iter_idx % self.cfg.train.validation_interval == 0 and iter_idx > 0:
    #             if iter_idx >= self.cfg.train.start_saving:
    #                 fname = os.path.join(self.cfg.train.output_dir, f'model_{iter_idx}.chkpt')
    #                 logger.info(f'save model to {fname}')
    #                 save_chkpt(self.model, self.cfg.train.output_dir,
    #                     iter_idx, self.optimizer
    #                 )
    #
    #             logger.info('evaluate prediction: ')
    #             validation_image, validation_label = next(self.validation_data_iter)
    #             validation_target = self.label_to_target(validation_label)
    #
    #             with torch.no_grad():
    #                 validation_predict = self.model(validation_image)
    #                 validation_loss = self.loss_module(validation_predict, validation_target)
    #                 validation_predict = self.post_processing(validation_predict)
    #                 per_voxel_loss = validation_loss.tolist() / self.voxel_num
    #                 logger.info(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
    #                 writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
    #                 log_tensor(writer, 'evaluate/image', validation_image, 'image', iter_idx)
    #                 log_tensor(writer, 'evaluate/prediction', validation_predict, 'image', iter_idx)
    #                 log_tensor(writer, 'evaluate/target', validation_target, 'image', iter_idx)
    #
    #     writer.close()
