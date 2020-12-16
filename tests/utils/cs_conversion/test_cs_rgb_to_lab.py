# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import numpy as np
import pytest

from tests import ONES_IMAGE_FLOAT, ColorType
from utils.cs_conversion import ChannelRange
from utils.cs_conversion.cs_rgb_to_lab import RgbToLabConverter

from .fixture_image import \
    fixture_input_image  # noqa pylint: disable=unused-import (W0611)

RGB_2_LAB_TEST_DATA = [((1.0, 0.0, 0.0), (53.240967, 80.09375, 67.203125)),
                       ((0.0, 1.0, 0.0), (87.73804, -86.1875, 83.171875)),
                       ((0.0, 0.0, 1.0), (32.2937, 79.1875, -107.859375)),
                       ((1.0, 1.0, 0.0), (97.13745, -21.546875, 94.46875)),
                       ((1.0, 0.0, 1.0), (60.321045, 98.234375, -60.828125)),
                       ((0.0, 1.0, 1.0), (91.11328, -48.09375, -14.125)),
                       ((1.0, 1.0, 1.0), (100., 0., 0.))]

LAB_2_RGB_TEST_DATA = [((0.0, -100.0, 100.0), (0.0, 0.15492299, 0.0)),
                       ((20.0, 75.0, -75.0), (0.31290624, 0.0, 0.64186966)),
                       ((40.0, 30., 60.0), (0.59813994, 0.28067374, 0.0)),
                       ((100.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                       ((60.0, -30.0, -60.0), (0.0, 0.6443579, 0.98083043)),
                       ((80.0, -80.0, 50.0), (0.0, 0.9053117, 0.38394532)),
                       ((100.0, 127.0, 127.0), (1.0, 0.27524692, 0.0))]


@pytest.mark.parametrize('source_rgb, target_lab', RGB_2_LAB_TEST_DATA)
def test_convert(source_rgb: ColorType, target_lab: ColorType) -> None:
    converter = RgbToLabConverter()
    image = np.copy(ONES_IMAGE_FLOAT)
    expected = np.copy(ONES_IMAGE_FLOAT)
    for channel in range(image.shape[-1]):
        image[:, :, channel] *= source_rgb[channel]
        expected[:, :, channel] *= target_lab[channel]
    image_lab = converter.convert(image)
    np.testing.assert_array_almost_equal(image_lab, expected)


@pytest.mark.parametrize('source_lab, target_rgb', LAB_2_RGB_TEST_DATA)
def test_convert_back(source_lab: ColorType, target_rgb: ColorType) -> None:
    converter = RgbToLabConverter()
    image = np.copy(ONES_IMAGE_FLOAT)
    expected = np.copy(ONES_IMAGE_FLOAT)
    for channel in range(image.shape[-1]):
        image[:, :, channel] *= source_lab[channel]
        expected[:, :, channel] *= target_rgb[channel]
    image_rgb = converter.convert_back(image)
    np.testing.assert_array_almost_equal(image_rgb, expected)


def test_channel_ranges() -> None:
    converter = RgbToLabConverter()
    ranges = converter.target_channel_ranges()
    expected = (ChannelRange(0.0, 100.0),
                ChannelRange(-127.0, 127.0),
                ChannelRange(-127.0, 127.0))
    assert ranges == expected
