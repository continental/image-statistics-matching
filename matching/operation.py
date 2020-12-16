"""This module defines Operation interface"""
import abc
from typing import Tuple

import numpy as np

from core import DIM_3

ChannelsType = Tuple[int, ...]

_FLOAT_TYPES = (np.float32, np.float64)


class Operation(abc.ABC):
    """ The interface declares operations common to all operations """

    # pylint: disable=too-few-public-methods (R0903)

    def __init__(self, channels: ChannelsType, check_input: bool = False):
        self.channels = channels
        self.check_input = check_input

    @property
    def channels(self) -> ChannelsType:
        """ Returns the channels of the color space """
        return self._channels

    @channels.setter
    def channels(self, channels: ChannelsType) -> None:
        if not isinstance(channels, tuple):
            raise TypeError(f'channels has to be of type {repr(ChannelsType)}')

        for channel in channels:
            if not isinstance(channel, int):
                raise TypeError(
                    f'Each element of channels has to be of type {repr(int)}')
        self._channels = channels

    @property
    def check_input(self) -> bool:
        """ Returns the input check flag """
        return self._check_input

    @check_input.setter
    def check_input(self, check_input: bool) -> None:
        self._check_input = bool(check_input)

    def _verify_input(self, source: np.ndarray,
                      reference: np.ndarray) -> None:
        if source.ndim != reference.ndim:
            raise ValueError(
                f'Source and reference have to be of the same dimension, '
                f'but source has {source.ndim} and reference has '
                f'{reference.ndim}')
        if source.ndim != DIM_3:
            raise ValueError(
                f'Input images have to be 3 dimensional, but '
                f'they are {source.ndim} dimensional')
        if source.shape[-1] != reference.shape[-1]:
            raise ValueError(
                f'The number of channels in source and reference '
                f'have to be the same, but source has '
                f'{source.shape[-1]} and reference has '
                f'{reference.shape[-1]}')
        if source.dtype not in _FLOAT_TYPES:
            raise TypeError(f'Source has to be of float type,'
                            f'but it is {source.dtype}')
        if reference.dtype not in _FLOAT_TYPES:
            raise TypeError(f'Reference has to be of float type, '
                            f'but it is {reference.dtype}')

        for idx, channel in enumerate(self._channels):
            if abs(channel) >= source.shape[-1]:
                raise IndexError(
                    f'{idx} channel is out of range')

    def __call__(self, source: np.ndarray,
                 reference: np.ndarray) -> np.ndarray:
        """ Calls operation implementation """
        if self.check_input:
            self._verify_input(source, reference)
        return self._apply(source, reference)

    @abc.abstractmethod
    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        """ Operation implementation """
