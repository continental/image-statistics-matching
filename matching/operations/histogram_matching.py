"""This module implements Histogram Matching operation"""
import sys

import numpy as np

from core import MATCH_FULL, MATCH_ZERO
from matching import ChannelsType, Operation


class HistogramMatching(Operation):
    """Histogram Matching operation class"""

    def __init__(self, channels: ChannelsType, check_input: bool = False,
                 match_prop: float = MATCH_FULL):
        super().__init__(channels, check_input)
        self.match_prop = float(match_prop)

    @property
    def match_prop(self) -> float:
        """ Returns the matching proportion value """
        return self._match_prop

    @match_prop.setter
    def match_prop(self, matching_proportion: float) -> None:
        if MATCH_ZERO <= matching_proportion <= MATCH_FULL:
            self._match_prop = matching_proportion
        else:
            raise ValueError(f'matching proportion has to be '
                             f'in range [{MATCH_ZERO}, {MATCH_FULL}], '
                             f'the given value is {matching_proportion}')

    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        result = np.copy(source)
        for channel in self.channels:
            result[:, :, channel] = \
                self._match_channel(source[:, :, channel],
                                    reference[:, :, channel])
        return result.astype(np.float32)

    def _match_channel(self, source: np.ndarray,
                       reference: np.ndarray) -> np.ndarray:
        if self.match_prop == MATCH_ZERO:
            return source

        source_shape = source.shape
        source = source.ravel()
        reference = reference.ravel()

        # get unique pixel values (sorted),
        # indices of the unique array and counts
        _, s_indices, s_counts = np.unique(source,
                                           return_counts=True,
                                           return_inverse=True)
        r_values, r_counts = np.unique(reference, return_counts=True)

        # compute the cumulative sum of the counts
        s_quantiles = np.cumsum(s_counts).astype(float) / (
            source.size + sys.float_info.epsilon)
        r_quantiles = np.cumsum(r_counts).astype(float) / (
            reference.size + sys.float_info.epsilon)

        # interpolate linearly to find the pixel values in the reference
        # that correspond most closely to the quantiles in the source image
        interp_values = np.interp(s_quantiles, r_quantiles, r_values)

        # pick the interpolated pixel values using the inverted source indices
        result = interp_values[s_indices]

        # apply matching proportion
        if self.match_prop < MATCH_FULL:
            diff = source.astype(float) - result
            result = source.astype(float) - (diff * self.match_prop)

        return result.reshape(source_shape)
