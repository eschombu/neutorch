import math
import random
from functools import cached_property

import numpy as np
import torch
from chunkflow.lib.cartesian_coordinate import Cartesian
from yacs.config import CfgNode

from neutorch.data.sample import *
from neutorch.data.transform import *

DEFAULT_PATCH_SIZE = Cartesian(128, 128, 128)
sample_classes = {name: c for name, c in locals().items() if isinstance(c, type) and issubclass(c, AbstractSample)}


def load_cfg(cfg_file: str, freeze: bool = True):
    with open(cfg_file) as file:
        cfg = CfgNode.load_cfg(file)
    if freeze:
        cfg.freeze()
    return cfg


def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    
    # the dataset copy in this worker process
    dataset = worker_info.dataset
    overall_start = 0
    overall_end = dataset.sample_num

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil(
        (overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def path_to_dataset_name(path: str, dataset_names: list):
    for dataset_name in dataset_names:
        if dataset_name in path:
            return dataset_name


def to_tensor(arr, cuda=False):
    if isinstance(arr, np.ndarray):
        # Pytorch only supports types: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        if np.issubdtype(arr.dtype, np.uint16):
            arr = arr.astype(np.int32)
        elif np.issubdtype(arr.dtype, np.uint64):
            arr = arr.astype(np.int64)
        arr = torch.tensor(arr)
    if cuda and torch.cuda.is_available():
        arr = arr.cuda()
    return arr


class DatasetBase(torch.utils.data.IterableDataset):
    default_sample_class = None

    def __init__(self, samples: list, cuda=True):
        super().__init__()
        print(f'got {len(samples)} samples.')
        assert len(samples) > 0
        self.samples = samples
        self.cuda = cuda

    @classmethod
    def from_config_v5(cls, 
            sample_config_files: List[str], 
            mode: str = 'training',
            inputs = ['image'],
            labels = ['label'],
            output_patch_size: Cartesian = Cartesian(128, 128, 128),
            ):

        samples = []

        for sample_config_file in sample_config_files:
            sample_dir = os.path.dirname(sample_config_file)
            sample_cfg = load_cfg(sample_config_file)
            for sample_name, properties in sample_cfg.items():
                if properties.mode != mode:
                    print(f'skip {sample_name} with mode of {properties.mode} since current mode is {mode}')
                    continue
                sample = Sample.from_config_v5(
                    properties,
                    sample_dir,
                    output_patch_size=output_patch_size,
                    inputs = inputs,
                    labels = labels,
                )
                if sample is not None:
                    samples.append(sample)

        return cls(samples)

    def label_to_target(self, label: Chunk) -> Chunk:
        return label

    @cached_property
    def sample_num(self):
        return len(self.samples)
    
    @cached_property
    def sample_weights(self):
        """use the number of candidate patches as volume sampling weight
        if there is None, replace it with average value
        """
        sample_weights = []
        for sample in self.samples:
            sample_weights.append(sample.sampling_weight)

        # replace None weight with average weight 
        ws = []
        for x in sample_weights:
            if x is not None:
                ws.append(x)
        average_weight = np.mean(ws)
        for idx, w in enumerate(sample_weights):
            if w is None:
                sample_weights[idx] = average_weight 
        return sample_weights

    @property
    def random_patch(self):
        # only sample one subject, so replacement option could be ignored
        sample_index = random.choices(
            range(0, self.sample_num),
            weights=self.sample_weights,
            k=1,
        )[0]
        sample = self.samples[sample_index]
        patch = sample.random_patch
        # patch.to_tensor()
        return patch.image, patch.label
    
    def __len__(self):
        patch_num = 0
        for sample in self.samples:
            assert sample is not None
            patch_num += len(sample)
        assert patch_num > 0
        return patch_num

    def __next__(self):
        image_chunk, label_chunk = self.random_patch
        image = to_tensor(image_chunk.array, self.cuda)
        label = to_tensor(label_chunk.array, self.cuda)
        assert image.ndim == 4
        return image, label

    def __getitem__(self, index: int):
        """return a random patch from a random sample, the index does not matter!"""
        return next(self)

    @classmethod
    def _sample_from_config(cls, sample_name: str, sample_cfg: CfgNode, patch_size: CfgNode, **kwargs):
        if 'type' in sample_cfg:
            if sample_cfg.type in sample_classes:
                sample_class = sample_classes[sample_cfg.type]
            else:
                raise ValueError(f"Unrecognized Sample class: '{sample_cfg.type}'")
        elif cls.default_sample_class is not None:
            sample_class = cls.default_sample_class
        else:
            raise ValueError('Sample class is not specified')

        if 'nonzero_bounding_boxes_path' in sample_cfg:
            sample_nz_bbox_path = sample_cfg.nonzero_bounding_boxes_path
        elif 'mask' in sample_cfg:
            mask_filename = os.path.splitext(os.path.split(sample_cfg.mask.split('#', 1)[0])[1])[0]
            sample_nz_bbox_path = f'{sample_name}_{mask_filename}_nonzero_bboxes.npy'
        else:
            sample_nz_bbox_path = None

        return sample_class.from_config(
            sample_cfg,
            output_patch_size=patch_size,
            nonzero_bounding_boxes_path=sample_nz_bbox_path,
            **kwargs)


    @classmethod
    def from_config(cls, cfg: CfgNode, is_train: bool = True, **kwargs):
        if is_train:
            sample_names = cfg.dataset.training
        else:
            sample_names = cfg.dataset.validation
        patch_size = Cartesian.from_collection(cfg.train.patch_size)

        samples = []
        for sample_name in sample_names:
            # Get Sample class from config, or use default_sample_class
            sample_cfg = cfg.samples[sample_name]
            sample = cls._sample_from_config(sample_name, sample_cfg, patch_size, **kwargs)
            samples.append(sample)

        return cls(samples, cuda=(cfg.system.gpus > 0))


class SemanticDataset(DatasetBase):
    default_sample_class = SemanticSample

    def label_to_target(self, label_chunk: Chunk) -> Chunk:
        target_chunk = (label_chunk>0).astype(np.float32)
        assert isinstance(target_chunk, Chunk)
        # label = (label > 0).to(torch.float32)
        return target_chunk

    @classmethod
    def _sample_from_config(cls, sample_name: str, sample_cfg: CfgNode, **kwargs):
        semantic_kwargs = dict(num_classes=cfg.model.out_channels)
        semantic_kwargs.update(kwargs)
        return super(cls)._sample_from_config(sample_name, sample_cfg, **semantic_kwargs)


class AffinityMapDataset(DatasetBase):
    default_sample_class = AffinityMapSample


class LSDsDataset(DatasetBase):
    default_sample_class = LSDsSample


class AffinitiesLSDsDataset(DatasetBase):
    default_sample_class = AffinitiesLSDsSample


class OrganelleDataset(SemanticDataset):
    default_sample_class = OrganelleSample

    def __init__(self, samples: list,
            num_classes: int = 1,
            skip_classes: list = None,
            selected_classes: list = None):
        """Dataset for organelle semantic segmentation

        Args:
            paths (list): list of samples
            sample_name_to_image_versions (dict, optional): map sample name to image volumes. Defaults to None.
            patch_size (Cartesian, optional): size of a output patch. Defaults to Cartesian(128, 128, 128).
            num_classes (int, optional): number of semantic classes to be classified. Defaults to 1.
            skip_classes (list, optional): skip some classes in the label. Defaults to None.
        """
        super().__init__(samples)

        self.samples = samples
        self.num_classes = num_classes
        
        if skip_classes is not None:
            skip_classes = sorted(skip_classes, reverse=True)
        self.skip_classes = skip_classes

        self.selected_classes = selected_classes

    @classmethod
    def from_path_list(cls, path_list: list,
            patch_size: Cartesian = Cartesian(128, 128, 128),
            num_classes: int = 1,
            skip_classes: list = None,
            selected_classes: list = None):
        path_list = sorted(path_list)
        samples = []
        # for img_path, sem_path in zip(self.path_list[0::2], self.path_list[1::2]):
        for label_path in path_list:
            sample = OrganelleSample.from_label_path(
                label_path, 
                num_classes, 
                patch_size=self.patch_size_before_transform,
                skip_classes=skip_classes,
                selected_classes = selected_classes,
            ) 
            
            samples.append(sample)
        
        return cls(samples, patch_size, num_classes, skip_classes, selected_classes)

    @cached_property
    def voxel_num(self):
        voxel_nums = [sample.voxel_num for sample in self.samples]
        return sum(voxel_nums)

    @cached_property
    def class_counts(self):
        counts = np.zeros((self.num_classes,), dtype=np.int)
        for sample in self.samples:
            counts += sample.class_counts
        return counts


class BoundaryAugmentationDataset(DatasetBase): 
    def __initi__(self, samples: list):
        super.__init__(samples)
    
    @classmethod
    def from_config(cls, cfg: CfgNode, is_train: bool, **kwargs):
        """Construct a semantic dataset with chunk or volume."""
        if is_train:
            name2chunks = cfg.dataset.training
        else:
            name2chunks = cfg.dataset.validation

        samples = []
        for type_name2paths in name2chunks.values():
            paths = [x for x in type_name2paths.values()][0]
            sample = SelfSupervisedSample.from_explicit_paths(
                    paths,
                    output_patch_size=cfg.train.patch_size,
                    num_classes=cfg.model.out_channels,
                    **kwargs)
            samples.append(sample)

        return cls(samples)

    
if __name__ == '__main__':

    from yacs.config import CfgNode
    from neutorch.data.patch import collate_batch
    batch_size = 1
    patch_num = 0

    # cfg_file = '/mnt/home/jwu/wasp/jwu/15_rna_granule_net/11/config.yaml'
    cfg_file = '/mnt/ceph/users/jwu/31_organelle/09_susumu_image/whole_brain_image.yaml'
    with open(cfg_file) as file:
        cfg = CfgNode.load_cfg(file)
    cfg.freeze()

    from neutorch.train.base import setup
    setup()

    training_dataset = AffinityMapDataset.from_config(cfg, mode='training')


    sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset,
        shuffle=False,
    )
    if cfg.system.cpus > 0:
        #prefetch_factor = cfg.system.cpus
        prefetch_factor = None
        multiprocessing_context='spawn'
    else:
        prefetch_factor = None
        multiprocessing_context=None


    dataloader = torch.utils.data.DataLoader(
        training_dataset,
        shuffle=False,
        num_workers = cfg.system.cpus,
        prefetch_factor = prefetch_factor,
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn,
        batch_size=batch_size,
        multiprocessing_context=multiprocessing_context,
        # pin_memory = True, # only dense tensor can be pinned. To-Do: enable it.
        # sampler=sampler
    )

    # from tqdm import tqdm
    # for idx in tqdm(range(1000)):
    #     image, label = training_dataset.random_patch

    patch_idx = 0
    while True:
        image, label = next(iter(dataloader))
    # for image, label in dataloader:
        patch_idx += 1
        print(f'patch {patch_idx} with size {image.shape} and {label.shape}')
        # breakpoint()