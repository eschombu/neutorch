#/bin/env python

import os
import pdb
import sys
import traceback
from functools import cached_property

import click
from yacs.config import CfgNode

from neutorch.data.dataset import AffinityMapVolumeWithMask, load_cfg
from neutorch.train.base import TrainerBase, setup, cleanup

import torch
import torch.distributed as dist
# torch.multiprocessing.set_start_method('spawn')


class WholeBrainAffinityMapTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode, 
            device: torch.DeviceObjType=None,
            local_rank: int = int(os.getenv('LOCAL_RANK', -1))
        ) -> None:
        super().__init__(cfg, device=device, local_rank=local_rank)
        assert isinstance(cfg, CfgNode)

    @cached_property
    def training_dataset(self):
        return AffinityMapVolumeWithMask.from_config(self.cfg, mode='training')
       
    @cached_property
    def validation_dataset(self):
        return AffinityMapVolumeWithMask.from_config(self.cfg, mode='validation')



@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help='configuration file containing all the parameters.'
)
@click.option('--local-rank', '-r',
    type=click.INT, default=int(os.getenv('LOCAL_RANK', -1)),
    help='rank of local process. It is used to assign batches and GPU devices.'
)
@click.option('--pdb/--no-pdb', 'pdb_debug', default=False, help='Enable pdb upon exception.')
def main(config_file: str, local_rank: int, pdb_debug: bool):
    if local_rank != -1:
        dist.init_process_group(backend="nccl", init_method='env://')
        print(f'local rank of processes: {local_rank}')
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
    # else:
    #     setup()
    try:
        cfg = load_cfg(config_file)
        if pdb_debug:
            cfg.defrost()
            cfg.system.cpus = 0
            # cfg.system.gpus = 0
            cfg.freeze()
        trainer = WholeBrainAffinityMapTrainer(cfg)
        trainer()
    except (KeyboardInterrupt, pdb.bdb.BdbQuit):
        sys.exit(1)
    except Exception as e:
        if pdb_debug:
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise e
    finally:
        # cleanup()
        pass
