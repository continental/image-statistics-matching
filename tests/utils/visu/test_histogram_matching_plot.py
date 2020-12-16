# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import os

import numpy as np
import pytest

from core import GRAY, IMAGE_CHANNELS, RGB, Params
from matching.operations import HistogramMatching
from tests import TEST_DIR
from tests.constants import (MUNICH_1_GRAY_PATH, MUNICH_1_PATH,
                             MUNICH_2_GRAY_PATH, MUNICH_2_PATH)
from utils.cs_conversion.cs_converter_builder import build_cs_converter
from utils.image_io import read_image
from utils.visu import histogram_matching_plot as hm_plot

TEST_IMAGE = np.array([[10, 21, 15, 15, 15],
                       [5, 10, 165, 21, 15],
                       [5, 5, 10, 7, 7],
                       [5, 9, 8, 7, 10]], dtype=float) / 255.


def test_cdf() -> None:
    # pylint: disable=protected-access (W0212)
    img = read_image(MUNICH_1_PATH)
    expected_values, counts = np.unique(img, return_counts=True)
    expected_cdf = np.cumsum(counts).astype(float) / img.size

    values, cdf = hm_plot._cdf(img)

    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(cdf, expected_cdf)


def test_hist() -> None:
    # pylint: disable=protected-access (W0212)
    hist = hm_plot._hist(TEST_IMAGE)
    expected_hist = np.histogram(TEST_IMAGE, bins=hm_plot._BINS,
                                 range=hm_plot._HIST_RANGE)[0]
    np.testing.assert_array_equal(hist, expected_hist)


def test_image_histogram_gray() -> None:
    # pylint: disable=protected-access (W0212)
    img = read_image(MUNICH_1_GRAY_PATH)
    gray_hist = hm_plot._image_histogram_gray(img)

    expected_hist = np.histogram(img[:, :, 0], bins=hm_plot._BINS,
                                 range=hm_plot._HIST_RANGE)[0]
    assert gray_hist.name == hm_plot._GRAY
    np.testing.assert_array_equal(gray_hist.hist, expected_hist)


def test_image_histogram_rgb() -> None:
    # pylint: disable=protected-access (W0212)
    img = read_image(MUNICH_1_PATH)
    hist = hm_plot._image_histogram_rgb(img)

    assert len(hist) == img.shape[-1]

    for channel in range(img.shape[-1]):
        expected_hist = np.histogram(img[:, :, channel], bins=hm_plot._BINS,
                                     range=hm_plot._HIST_RANGE)[0]

        assert hist[channel].name == hm_plot._HIST_RGB_COLORS[channel]
        np.testing.assert_array_equal(hist[channel].hist, expected_hist)


@pytest.mark.parametrize(
    'color_space, image_channels, source_path, reference_path',
    [(RGB, IMAGE_CHANNELS, MUNICH_2_PATH, MUNICH_1_PATH),
     (GRAY, '0', MUNICH_2_GRAY_PATH, MUNICH_1_GRAY_PATH)])
def test_make_plot(color_space: str, image_channels: str, source_path: str,
                   reference_path: str) -> None:
    params = Params(
        {
            'color_space': color_space,
            'channels': image_channels,
            'match_proportion': 0.8,
            'src_path': source_path,
            'ref_path': reference_path
        }
    )
    converter = build_cs_converter(params.color_space)

    source = read_image(params.src_path)
    reference = read_image(params.ref_path)

    source = converter.convert(source)
    reference = converter.convert(reference)

    channels = tuple(int(c) for c in params.channels.split(','))
    hist_match = HistogramMatching(channels, params.match_proportion)
    result = hist_match(source, reference)

    file_name = os.path.join(TEST_DIR, 'histogram_matching_plot.png')
    images = hm_plot.Images(source, reference, result)
    hm_plot.make_plot(file_name, images, converter, params.color_space,
                      params.channels)

    assert os.path.exists(file_name) is True

    os.remove(file_name)
