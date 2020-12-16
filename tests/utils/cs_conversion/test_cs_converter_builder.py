# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import pytest

from core import GRAY, HSV, LAB, RGB
from utils.cs_conversion.cs_converter_builder import build_cs_converter
from utils.cs_conversion.cs_identity import IdentityConverter
from utils.cs_conversion.cs_rgb_to_hsv import RgbToHsvConverter
from utils.cs_conversion.cs_rgb_to_lab import RgbToLabConverter


def test_build_cs_converter() -> None:
    conv_gray = build_cs_converter(GRAY)
    assert isinstance(conv_gray, IdentityConverter)
    assert len(conv_gray.target_channel_ranges()) == 1

    conv_hsv = build_cs_converter(HSV)
    assert isinstance(conv_hsv, RgbToHsvConverter)

    conv_lab = build_cs_converter(LAB)
    assert isinstance(conv_lab, RgbToLabConverter)

    conv_rgb = build_cs_converter(RGB)
    assert isinstance(conv_rgb, IdentityConverter)

    with pytest.raises(ValueError):
        build_cs_converter('xyz')
