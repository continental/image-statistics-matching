"""This module provides a color space converter from rgb to lab"""
from typing import Tuple

import cv2
import numpy as np

from . import ChannelRange, ColorSpaceConverter

L_MIN = 0.0
L_MAX = 100.0
AB_MIN = -127.0
AB_MAX = 127.0


class RgbToLabConverter(ColorSpaceConverter):
    """ this ColorSpaceConverter converts images from rgb to lab
    and vice versa """

    def convert(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    def convert_back(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        return (ChannelRange(L_MIN, L_MAX),
                ChannelRange(AB_MIN, AB_MAX),
                ChannelRange(AB_MIN, AB_MAX))
