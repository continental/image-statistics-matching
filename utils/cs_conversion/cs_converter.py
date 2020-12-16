"""This module defines the ColorSpaceConverter interface"""
import abc
from typing import NamedTuple, Tuple

import numpy as np

ChannelRange = NamedTuple('ChannelRange', [('min', float), ('max', float)])


class ColorSpaceConverter(abc.ABC):
    """ the ColorSpaceConverter interface declares operations common to all
    color space conversion algorithms """

    @abc.abstractmethod
    def convert(self, image: np.ndarray) -> np.ndarray:
        """ converts image from source to target color space """

    @abc.abstractmethod
    def convert_back(self, image: np.ndarray) -> np.ndarray:
        """ converts image from target color space back
        to source color space """

    @abc.abstractmethod
    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        """ returns the ranges of the color space """
