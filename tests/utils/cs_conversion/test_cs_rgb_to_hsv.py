# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import numpy as np
import pytest

from tests import ONES_IMAGE_FLOAT, ColorType
from utils.cs_conversion import ChannelRange
from utils.cs_conversion.cs_rgb_to_hsv import RgbToHsvConverter

from .fixture_image import \
    fixture_input_image  # noqa pylint: disable=unused-import (W0611)

TEST_DATA = [((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
             ((0.0, 1.0, 0.0), (120.0, 1.0, 1.0)),
             ((0.0, 0.0, 1.0), (240.0, 1.0, 1.0)),
             ((1.0, 1.0, 0.0), (60.0, 1.0, 1.0)),
             ((1.0, 0.0, 1.0), (300.0, 1.0, 1.0)),
             ((0.0, 1.0, 1.0), (180.0, 1.0, 1.0)),
             ((1.0, 1.0, 1.0), (0.0, 0, 1.0))]

COLOR_DIFF_MAX = 3


@pytest.mark.parametrize('source_rgb, target_hsv', TEST_DATA)
def test_convert(source_rgb: ColorType, target_hsv: ColorType) -> None:
    converter = RgbToHsvConverter()
    image = np.copy(ONES_IMAGE_FLOAT)
    expected = np.copy(ONES_IMAGE_FLOAT)
    for channel in range(image.shape[-1]):
        image[:, :, channel] *= source_rgb[channel]
        expected[:, :, channel] *= target_hsv[channel]
    image_hsv = converter.convert(image)
    np.testing.assert_array_almost_equal(image_hsv, expected, decimal=5)


@pytest.mark.parametrize('target_rgb, source_hsv', TEST_DATA)
def test_convert_back(source_hsv: ColorType, target_rgb: ColorType) -> None:
    converter = RgbToHsvConverter()
    image = np.copy(ONES_IMAGE_FLOAT)
    expected = np.copy(ONES_IMAGE_FLOAT)
    for channel in range(image.shape[-1]):
        image[:, :, channel] *= source_hsv[channel]
        expected[:, :, channel] *= target_rgb[channel]
    image_rgb = converter.convert_back(image)
    np.testing.assert_array_almost_equal(image_rgb, expected)


def test_image_diff(fxt_input_image: np.ndarray) -> None:
    converter = RgbToHsvConverter()
    image_hsv = converter.convert(fxt_input_image)
    assert (np.abs(image_hsv.astype(np.int16) - fxt_input_image.astype(
        np.int16)) > COLOR_DIFF_MAX).any()

    image_rgb = converter.convert_back(image_hsv)
    assert (np.abs(fxt_input_image.astype(np.int16) - image_rgb.astype(
        np.int16)) <= COLOR_DIFF_MAX).all()


def test_channel_ranges() -> None:
    converter = RgbToHsvConverter()
    ranges = converter.target_channel_ranges()
    expected = (ChannelRange(0.0, 360.0),
                ChannelRange(0.0, 1.0),
                ChannelRange(0.0, 1.0))
    assert ranges == expected
