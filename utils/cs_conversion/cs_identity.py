"""This module provides a no-op implementation that is used when no color
space conversion is requested"""
from typing import Tuple

import numpy as np

from core.constants import DIM_3

from . import ChannelRange, ColorSpaceConverter

RGB_MIN = 0.0
RGB_MAX = 1.0


class IdentityConverter(ColorSpaceConverter):
    """ IdentityConverter just gives back the original image color space """

    def __init__(self, dim: int = DIM_3):
        self._dim = int(dim)

    def convert(self, image: np.ndarray) -> np.ndarray:
        return image

    def convert_back(self, image: np.ndarray) -> np.ndarray:
        return image

    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        return tuple([ChannelRange(RGB_MIN, RGB_MAX)] * self._dim)
