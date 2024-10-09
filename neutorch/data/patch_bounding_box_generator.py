from random import randrange, choice
from abc import ABC, abstractproperty
from functools import cached_property
from typing import Iterator, List, Union

import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import Cartesian, BoundingBox
from chunkflow.volume import AbstractVolume


def generate_patch_bounding_boxes(
        volume_bbox: BoundingBox,
        patch_size: Cartesian,
        overlap: tuple = (0, 0, 0),
        forbidden_distance_to_boundary: Union[int, tuple] = (0, 0, 0),
) -> Iterator[BoundingBox]:

    if not isinstance(patch_size, Cartesian):
        patch_size = Cartesian.from_collection(patch_size)
    if not isinstance(overlap, Cartesian):
        overlap = Cartesian.from_collection(overlap)
    if isinstance(forbidden_distance_to_boundary, int):
        forbidden_distance_to_boundary = (forbidden_distance_to_boundary,) * 3
    assert len(forbidden_distance_to_boundary) == 3 or len(forbidden_distance_to_boundary) == 6

    start = volume_bbox.start + forbidden_distance_to_boundary[:3]
    stop = volume_bbox.stop - forbidden_distance_to_boundary[-3:]
    stride = patch_size - overlap

    for z in range(start.z, stop.z - stride.z + 1, stride.z):
        for y in range(start.y, stop.y - stride.y + 1, stride.y):
            for x in range(start.x, stop.x - stride.x + 1, stride.x):
                yield BoundingBox.from_delta(Cartesian(z, y, x), stride)


class AbstractPatchBoundingBoxGenerator(ABC):
    def __init__(self, 
            patch_size: Cartesian, 
            volume_bbox: BoundingBox,
            forbidden_distance_to_boundary: Cartesian = None
        ) -> None:
        super().__init__()
        assert isinstance(patch_size, Cartesian)
        assert isinstance(volume_bbox, BoundingBox)
        self.patch_size = patch_size
        self.volume_bbox = volume_bbox

        if forbidden_distance_to_boundary is None:
            forbidden_distance_to_boundary = self.patch_size_before_transform // 2
        assert len(forbidden_distance_to_boundary) == 3 or len(forbidden_distance_to_boundary)==6
        self.forbidden_distance_to_boundary = forbidden_distance_to_boundary

    @abstractproperty
    def patch_bbox(self):
        pass


class PatchBoundingBoxGeneratorInChunk(AbstractPatchBoundingBoxGenerator):
    def __init__(self, 
            patch_size: Cartesian, 
            volume_bbox: BoundingBox,
            forbidden_distance_to_boundary: tuple = None
        ) -> None:
        super().__init__(
            patch_size, volume_bbox, 
            forbidden_distance_to_boundary=forbidden_distance_to_boundary
        )

        if forbidden_distance_to_boundary is None:
            left = patch_size // 2
            # ceiling division
            right = -(-patch_size // 2)
            forbidden_distance_to_boundary = left.tuple + right.tuple

        for idx in range(3):
            # the center of random patch should not be too close to boundary
            # otherwise, the patch will go outside of the volume
            assert forbidden_distance_to_boundary[idx] >= self.patch_size_before_transform[idx] // 2
            assert forbidden_distance_to_boundary[-idx] >= self.patch_size_before_transform[-idx] // 2
        
        self.center_start = Cartesian.from_collection(
            forbidden_distance_to_boundary[:3])
        self.center_stop = Cartesian.from_collection(
            tuple(s - d for s, d in zip(
            volume_bbox.shape[-3:], forbidden_distance_to_boundary[-3:]
        )))
        assert self.center_stop > self.center_start, \
            f'center start: {self.center_start}, center stop: {self.center_stop}'

    @property
    def random_patch_center(self):
        center_start = self.center_start
        center_stop = self.center_stop
        cz = randrange(center_start[0], center_stop[0])
        cy = randrange(center_start[1], center_stop[1])
        cx = randrange(center_start[2], center_stop[2])
        center = Cartesian(cz, cy, cx)
        return center

    @property
    def random_patch_bbox(self):
        center  = self.random_patch_center 
        start = center - self.patch_size // 2
        bbox = BoundingBox.from_delta(start, self.patch_size)
        return bbox


class PatchBoundingBoxGeneratorInsideMask(PatchBoundingBoxGeneratorInChunk):
    def __init__(self, 
            patch_size: Cartesian, image_volume: AbstractVolume, 
            mask_volume: AbstractVolume,
            forbidden_distance_to_boundary: tuple = None,
            ) -> None:
        """Generate patch location that is inside a mask.
        The mask is normally a downsampled volume that could be loaded in RAM.

        Args:
            patch_size (Cartesian): patch size before transform
            image_volume (AbstractVolume): volume of image
            mask_volume (AbstractVolume): a binary volume that the ROI is 1 and the background is 0.
            patch_voxel_size (Cartesian): the voxel size of the image or patch for training.
            forbidden_distance_to_boundary (tuple, optional): 
                distance from the boundary of patch to boundary of volume or chunk. 
                Defaults to None.
        """
        super().__init__(
            patch_size, image_volume.bounding_box, forbidden_distance_to_boundary)
        self.image_volume = image_volume
        self.mask_volume = mask_volume
        
    @cached_property
    def block_size(self):
        """block/chunk size of the image and label volume. The sampled patch should be completely inside the blocks to reduce the computational cost of reading and decompression."""
        bs = self.image_volume.block_size
        assert bs >= self.patch_size, 'if this is not satisfied, we can double/triple the axis of block size to make sure that the patch can fit in a block.'
        return bs

    @cached_property
    def mask_factor(self):
        assert self.mask_volume.voxel_size % self.image_volume.voxel_size == Cartesian(0,0,0), 'mask volume voxel size should be dividable by image voxel size'
        return self.mask.voxel_size // self.patch_voxel_size

    @cached_property
    def nonzero_block_bounding_boxes(self) -> List[BoundingBox]:
        """find the image bounding boxes that the corresponding mask chunk is all positive.
        Note that the mask volume voxel size might not be the same with the image volume.
        It is normally downsampled recursivly by 2x2 or 2x2x2.

        Returns:
            List[BoundingBox]: the image bounding boxes that is inside the mask blocks.
        """
        nnz_block_bboxes = []
        image_bbox = self.image_volume.bounding_box

        for z in range(image_bbox.start.z, image_bbox.stop.z, self.block_size.z ):
            for y in range(image_bbox.start.y, image_bbox.stop.y, self.block_size.y):
                for x in range(image_bbox.start.x, image_bbox.stop.x, self.block_size.x):
                    start = Cartesian(z, y, x)
                    image_bbox = BoundingBox.from_delta(start, self.block_size)
                    # map to the mask mip level 
                    mask_bbox = image_bbox // self.mask_factor
                    mask_chunk = self.mask_volume.cutout(mask_bbox)
                    if np.all(mask_chunk>0):
                        nnz_block_bboxes.append(image_bbox)
        return nnz_block_bboxes
    
    @property 
    def random_patch_bbox(self):
        # select a random block
        bbox = choice(self.nonzero_block_bounding_boxes)
        range_start = bbox.start + Cartesian.from_collection(
            self.forbidden_distance_to_boundary[:3])
        range_stop = bbox.stop - self.patch_size - \
            Cartesian.from_collection(
                self.forbidden_distance_to_boundary[-3:]
            )
        range_bbox = BoundingBox(range_start, range_stop)
        patch_start = range_bbox.random_coordinate
        patch_bbox = BoundingBox.from_delta(patch_start, self.patch_size)
        return patch_bbox

        
