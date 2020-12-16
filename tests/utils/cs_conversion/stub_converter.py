# pylint: disable=missing-module-docstring (C0114)
from typing import Tuple

import numpy as np

from utils.cs_conversion import ChannelRange, ColorSpaceConverter

RANGE_MIN = 0.0
RANGE_MAX = 1.0


class StubConverter(ColorSpaceConverter):
    """
    StubConverter is used for stubbing tests
    """

    def convert(self, image: np.ndarray) -> np.ndarray:
        return image * 0.21

    def convert_back(self, image: np.ndarray) -> np.ndarray:
        return image * 0.22

    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        return tuple([ChannelRange(RANGE_MIN, RANGE_MAX)] * 3)
