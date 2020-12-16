"""This module provides a color space converter from rgb to hsv"""
from typing import Tuple

import cv2
import numpy as np

from . import ChannelRange, ColorSpaceConverter

H_MIN = 0.0
H_MAX = 360.0
SV_MIN = 0.0
SV_MAX = 1.0


class RgbToHsvConverter(ColorSpaceConverter):
    """ this ColorSpaceConverter converts images from rgb to hsv
    and vice versa """

    def convert(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def convert_back(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        return (ChannelRange(H_MIN, H_MAX),
                ChannelRange(SV_MIN, SV_MAX),
                ChannelRange(SV_MIN, SV_MAX))
