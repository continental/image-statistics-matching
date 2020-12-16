"""This module contains a function that creates a specified color space
converter"""
from core import DIM_1, GRAY, HSV, LAB, RGB

from . import ColorSpaceConverter
from .cs_identity import IdentityConverter
from .cs_rgb_to_hsv import RgbToHsvConverter
from .cs_rgb_to_lab import RgbToLabConverter


def build_cs_converter(color_space: str) -> ColorSpaceConverter:
    """Creates ColorSpaceConverter for specified color space"""
    target_color_space = color_space.lower()

    if target_color_space == GRAY:
        return IdentityConverter(DIM_1)
    if target_color_space == HSV:
        return RgbToHsvConverter()
    if target_color_space == LAB:
        return RgbToLabConverter()
    if target_color_space == RGB:
        return IdentityConverter()

    raise ValueError(f'there is no color space converter for {color_space}')
