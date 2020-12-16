"""This module implements visualization for Histogram Matching operation"""
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pylab import rcParams

from core import DIM_1, IMAGE_CHANNELS
from utils.cs_conversion import ColorSpaceConverter

_PLOT_WIDTH = 16
_PLOT_HEIGHT = 12
_PLOT_SIZE = (_PLOT_WIDTH, _PLOT_HEIGHT)
_GRAY_AXIS = 1

_BINS = 50
_HIST_RANGE = (0, 1)
_GRAY = 'gray'
_HIST_RGB_COLORS = ('red', 'green', 'blue')
_IMAGE_TITLES = ('Source', 'Reference', 'Result')

Histogram = NamedTuple('Histogram', [('name', str), ('hist', np.ndarray)])

Images = NamedTuple('Images', [('source', np.ndarray),
                               ('reference', np.ndarray),
                               ('result', np.ndarray)])


def make_plot(file_name: str, images: Images, converter: ColorSpaceConverter,
              color_space: str, channels: str) -> None:
    """ Plot image histograms and cumulative distribution functions """

    sns.set()

    rcParams['figure.figsize'] = _PLOT_SIZE
    _, axes = plt.subplots(3, 3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # plot images
    for i, axis in enumerate(axes[0]):
        _plot_image(axis, images[i], _IMAGE_TITLES[i])

    # plot image histograms
    for i, axis in enumerate(axes[1]):
        if images[i].shape[-1] == DIM_1:
            _plot_histogram_gray(axis, images[i], _IMAGE_TITLES[i])
        else:
            _plot_histogram_rgb(axis, images[i], _IMAGE_TITLES[i])

    # plot cumulative distributions
    if images[0].shape[-1] == DIM_1:
        _plot_cum_dist_gray(axes[2], images, converter, color_space, channels)
    else:
        _plot_cum_dist_rgb(axes[2], images, converter, color_space, channels)

    plt.savefig(file_name, bbox_inches='tight')


def _plot_image(axis: plt.axes, image: np.ndarray, title: str) -> None:
    axis.imshow(image, cmap=_GRAY)
    axis.set_title(title)
    axis.set_xticklabels(list())
    axis.set_yticklabels(list())
    axis.grid(False)


def _plot_histogram_gray(axis: plt.axes, image: np.ndarray,
                         title: str) -> None:
    image_hist = _image_histogram_gray(image)

    norm = image_hist.hist / image.size
    axis.fill_between([float(b) / _BINS for b in range(_BINS)], norm,
                      facecolor='tab:blue', alpha=0.7)
    axis.set_title(f'{title} Grayscale histogram')
    axis.grid(True)


def _plot_histogram_rgb(axis: plt.axes, image: np.ndarray, title: str) -> None:
    for name, color in _image_histogram_rgb(image):
        norm = color / image.size
        axis.fill_between([float(b) / _BINS for b in range(_BINS)], norm,
                          facecolor=name, alpha=0.5)
    axis.set_title(f'{title} RGB histogram')
    axis.grid(True)


def _plot_cum_dist_channel(axes: plt.axes, images: Images,
                           converter: ColorSpaceConverter,
                           axis_index: int, channel: int) -> None:
    source = converter.convert(images.source)
    reference = converter.convert(images.reference)
    result = converter.convert(images.result)

    src_cdf = _cdf(source[:, :, channel])
    ref_cdf = _cdf(reference[:, :, channel])
    res_cdf = _cdf(result[:, :, channel])

    axes[axis_index].plot(src_cdf[0], src_cdf[1], label='Source')
    axes[axis_index].plot(ref_cdf[0], ref_cdf[1], label='Reference')
    axes[axis_index].plot(res_cdf[0], res_cdf[1], '--r', lw=2, label='Result')
    axes[axis_index].legend(loc=10, bbox_to_anchor=(0.8, 0.15), frameon=False)
    axes[axis_index].grid(True)


def _plot_cum_dist_gray(axes: plt.axes, images: Images,
                        converter: ColorSpaceConverter, color_space: str,
                        channels: str) -> None:
    axes[_GRAY_AXIS].set_title(
        'cumulative distribution ' + color_space.upper())
    _plot_cum_dist_channel(axes, images, converter, _GRAY_AXIS, int(channels))


def _plot_cum_dist_rgb(axes: plt.axes, images: Images,
                       converter: ColorSpaceConverter, color_space: str,
                       channels: str) -> None:
    all_channels = tuple(int(c) for c in IMAGE_CHANNELS.split(','))
    for channel in all_channels:
        axis_index = channel
        axes[axis_index].set_title(
            'cumulative distribution ' + color_space[channel].upper())

    used_channels = tuple(int(c) for c in channels.split(','))
    for channel in used_channels:
        axis_index = channel
        _plot_cum_dist_channel(axes, images, converter, axis_index, channel)


def _cdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(arr, return_counts=True)
    cdf = np.cumsum(counts).astype(float) / arr.size
    return values, cdf


def _hist(arr: np.ndarray) -> np.ndarray:
    return np.histogram(arr, bins=_BINS, range=_HIST_RANGE)[0]


def _image_histogram_gray(image: np.ndarray) -> Histogram:
    return Histogram(_GRAY, _hist(image[:, :, 0]))


def _image_histogram_rgb(image: np.ndarray) -> Tuple[Histogram, ...]:
    return tuple([Histogram(_HIST_RGB_COLORS[i], _hist(image[:, :, i]))
                  for i in range(image.shape[-1])])
