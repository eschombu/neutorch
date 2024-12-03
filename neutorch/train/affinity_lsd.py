from functools import cached_property

import click
from yacs.config import CfgNode

from neutorch.data.dataset import AffinitiesLSDsDataset, LSDsDataset

from .base import TrainerBase


class AffinitiesLSDsTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode, include_affinities=False) -> None:
        assert isinstance(cfg, CfgNode)
        super().__init__(cfg)
        self.cfg = cfg

    @cached_property
    def training_dataset(self):
        if include_affinities:
            return AffinitiesLSDsDataset.from_config(self.cfg, is_train=True)
        else:
            return LSDsDataset.from_config(self.cfg, is_train=True)
       
    @cached_property
    def validation_dataset(self):
        if include_affinities:
            return AffinitiesLSDsDataset.from_config(self.cfg, is_train=False)
        else:
            return LSDsDataset.from_config(self.cfg, is_train=False)


@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
def main(config_file: str):
    from neutorch.data.dataset import load_cfg
    cfg = load_cfg(config_file)
    trainer = AffinityMapTrainer(cfg)
    trainer()