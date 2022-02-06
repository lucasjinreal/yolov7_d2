from fvcore.transforms.transform import NoOpTransform
import numpy as np
import sys
from typing import Tuple
from PIL import Image

from detectron2.data.transforms.augmentation import AugInput, Augmentation, _transform_to_aug
from .transform import (
    GridMaskTransform, Grid,
    YOLOFJitterCropTransform,
    YOLOFDistortTransform,
    HFlipTransform,
    VFlipTransform,
    ResizeTransform,
    YOLOFShiftTransform
)


__all__ = [
    "YOLOFJitterCrop",
    "YOLOFResize",
    "YOLOFRandomDistortion",
    "RandomFlip",
    "YOLOFRandomShift",
    "RandomGridMask"
]
 

class RandomGridMask(Augmentation):

    def __init__(self, prob=0.5, *, use_h=True, use_w=True, mode=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        self.use_h = use_h
        self.use_w = use_w
        self.prob = prob
        self.mode = mode
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return GridMaskTransform(op=Grid(use_w=self.use_w, use_h=self.use_h, mode=self.mode))
        else:
            return NoOpTransform()


class YOLOFJitterCrop(Augmentation):
    """Jitter and crop the image and box."""

    def __init__(self, jitter_ratio):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        oh, ow = image.shape[:2]
        dw = int(ow * self.jitter_ratio)
        dh = int(oh * self.jitter_ratio)
        pleft = np.random.randint(-dw, dw)
        pright = np.random.randint(-dw, dw)
        ptop = np.random.randint(-dh, dh)
        pbot = np.random.randint(-dh, dh)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot
        return YOLOFJitterCropTransform(
            pleft=pleft, pright=pright, ptop=ptop, pbot=pbot,
            output_size=(swidth, sheight))


class YOLOFResize(Augmentation):
    """
        Resize image to a target size
        """

    def __init__(self, shape, interp=Image.BILINEAR, scale_jitter=None):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
            scale_jitter (optional, tuple[float, float]): None or (0.8, 1.2)
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        assert (scale_jitter is None or isinstance(scale_jitter, tuple))
        self._init(locals())

    def get_transform(self, image):
        if self.scale_jitter is not None:
            if len(self.scale_jitter) > 2:
                assert isinstance(self.scale_jitter[0], tuple)
                idx = np.random.choice(range(len(self.scale_jitter)))
                shape = self.scale_jitter[idx]
            else:
                jitter = np.random.uniform(self.scale_jitter[0],
                                           self.scale_jitter[1])
                shape = (
                    int(self.shape[0] * jitter), int(self.shape[1] * jitter)
                )
        else:
            shape = self.shape
        return ResizeTransform(
            image.shape[0], image.shape[1], shape[0], shape[1], self.interp
        )


class YOLOFRandomDistortion(Augmentation):
    """
    Random distort image's hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        """
        RandomDistortion Initialization.
        Args:
            hue (float): value of hue
            saturation (float): value of saturation
            exposure (float): value of exposure
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        return YOLOFDistortTransform(self.hue, self.saturation, self.exposure)


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class YOLOFRandomShift(Augmentation):
    """
    Shift the image and box given shift pixels and probability.
    """

    def __init__(self, prob=0.5, max_shifts=32):
        """
        Args:
            prob (float): probability of shifts.
            max_shifts (int): the max pixels for shifting.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            shift_x = np.random.randint(low=-self.max_shifts,
                                        high=self.max_shifts)
            shift_y = np.random.randint(low=-self.max_shifts,
                                        high=self.max_shifts)
            return YOLOFShiftTransform(shift_x, shift_y)
        else:
            return NoOpTransform()
