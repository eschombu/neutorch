import logging
import pdb
import sys
import traceback
from functools import cached_property

import click
from yacs.config import CfgNode

from neutorch.data.dataset import AffinitiesLSDsDataset, LSDsDataset, load_cfg
from neutorch.train.base import TrainerBase
from neutorch.utils import log_utils


def _modify_config(cfg, key, val):
    if '.' in key:
        key, subkey = key.split('.', 1)
        _modify_config(cfg[key], subkey, val)
    else:
        cfg[key] = val


class AffinitiesLSDsTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode) -> None:
        assert isinstance(cfg, CfgNode)
        task = cfg.train.task.lower()
        if task == 'lsd':
            cfg.model.out_channels = cfg.train.num_lsds
        elif task == 'aff+lsd':
            cfg.model.out_channels = cfg.train.num_affinities + cfg.train.num_lsds
        else:
            raise ValueError(f"Invalid task string in config: '{cfg.train.task}'")
        assert cfg.model.out_channels is not None
        cfg.freeze()
        super().__init__(cfg)

    @cached_property
    def training_dataset(self):
        if self.cfg.train.task.lower() == 'lsd':
            return LSDsDataset.from_config(self.cfg, is_train=True)
        else:
            return AffinitiesLSDsDataset.from_config(self.cfg, is_train=True)
       
    @cached_property
    def validation_dataset(self):
        if self.cfg.train.task.lower() == 'lsd':
            return LSDsDataset.from_config(self.cfg, is_train=False)
        else:
            return AffinitiesLSDsDataset.from_config(self.cfg, is_train=False)


@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.')
@click.option('--test-mode/--train-mode', default=False,
    help='Evaluate model on test dataset.')
@click.option('--config-params', type=str, default=None,
    help='Comma separated list of key=value pairs to override config parameters.')
@click.option('--pdb/--no-pdb', 'pdb_debug', default=False,
    help='Enable pdb upon exception.')
@click.option('--pdb-step/--no-pdb-step', default=False,
    help='Use pdb to step through program.')
@click.option('--debug/--no-debug', default=False,
    help='Set log level to DEBUG.')
def main(config_file: str, test_mode: bool, config_params: str, pdb_debug: bool, pdb_step: bool, debug: bool):
    if pdb_step:
        pdb_debug = True
        debug = True
        pdb.set_trace()

    try:
        cfg = load_cfg(config_file, freeze=False)
        if config_params:
            cfg_raw = cfg.clone()
            for param in config_params.split(','):
                key, val = param.split('=')
                _modify_config(cfg, key, val)

        if pdb_debug:
            cfg.system.cpus = 0
            # cfg.system.gpus = 0
            log_utils.set_level(logging.DEBUG)
        else:
            if debug:
                log_utils.set_level(logging.DEBUG)
            else:
                log_utils.set_level(logging.INFO)
            torch.multiprocessing.set_start_method('spawn')
        trainer = AffinitiesLSDsTrainer(cfg)
        if test_mode:
            trainer.test()
        else:
            trainer()
    except (KeyboardInterrupt, pdb.bdb.BdbQuit):
        sys.exit(1)
    except Exception as e:
        if pdb_debug:
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise e


if __name__ == '__main__':
    main()
