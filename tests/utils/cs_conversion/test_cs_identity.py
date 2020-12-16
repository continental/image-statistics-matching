# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import numpy as np
import pytest

from core import DIM_1, DIM_3
from utils.cs_conversion.cs_identity import RGB_MAX, RGB_MIN, IdentityConverter

from .fixture_image import \
    fixture_input_image  # noqa pylint: disable=unused-import (W0611)


def test_convert(fxt_input_image: np.ndarray) -> None:
    converter = IdentityConverter()
    image = converter.convert(fxt_input_image)
    np.testing.assert_array_equal(image, fxt_input_image)


def test_convert_back(fxt_input_image: np.ndarray) -> None:
    converter = IdentityConverter()
    image = converter.convert_back(fxt_input_image)
    np.testing.assert_array_equal(image, fxt_input_image)


@pytest.mark.parametrize('dim', [DIM_3, DIM_1])
def test_channel_ranges(dim: int) -> None:
    converter = IdentityConverter(dim)
    assert converter._dim == dim  # pylint: disable=protected-access (W0212)
    ranges = converter.target_channel_ranges()
    assert len(ranges) == dim
    for interval in ranges:
        assert interval.min == RGB_MIN
        assert interval.max == RGB_MAX
