import logging
import pdb
import sys
import traceback
from functools import cached_property

import click
import torch.multiprocessing
from yacs.config import CfgNode

from neutorch.data.dataset import AffinityMapVolumeWithMask, load_cfg
from neutorch.train.base import TrainerBase
from neutorch.utils import log_utils


class WholeBrainAffinityMapTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode) -> None:
        assert isinstance(cfg, CfgNode)
        super().__init__(cfg)
        self.cfg = cfg

    @cached_property
    def training_dataset(self):
        return AffinityMapVolumeWithMask.from_config(self.cfg, is_train=True)
       
    @cached_property
    def validation_dataset(self):
        return AffinityMapVolumeWithMask.from_config(self.cfg, is_train=False)


@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help='configuration file containing all the parameters.'
)
@click.option('--pdb/--no-pdb', 'pdb_debug', default=False, help='Enable pdb upon exception.')
@click.option('--debug/--no-debug', default=False, help='Set log level to DEBUG upon exception.')
def main(config_file: str, pdb_debug: bool, debug: bool):
    try:
        cfg = load_cfg(config_file)
        if pdb_debug:
            cfg.defrost()
            cfg.system.cpus = 0
            # cfg.system.gpus = 0
            cfg.freeze()
            log_utils.set_level(logging.DEBUG)
        else:
            if debug:
                log_utils.set_level(logging.DEBUG)
            else:
                log_utils.set_level(logging.INFO)
            torch.multiprocessing.set_start_method('spawn')
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
