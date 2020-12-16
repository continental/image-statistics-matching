"""This module provides functions to read and write RGB images with openCV"""
import os

import cv2
import numpy as np

from core.constants import DIM_1, DIM_2

MAX_VALUE_8_BIT = 255


def read_image(path: str) -> np.ndarray:
    """ This function reads an image and transforms it to RGB color space """
    if os.path.exists(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32) / MAX_VALUE_8_BIT
        if image.ndim == DIM_2:
            return image[:, :, np.newaxis]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    raise ValueError(f'Invalid image path {path}')


def write_image(image: np.ndarray, path: str) -> None:
    """ This function transforms an image to BGR color space
    and writes it to disk """
    if image.dtype == np.float32:
        image = (image * MAX_VALUE_8_BIT).astype(np.uint8)
    if image.dtype == np.uint8:
        if image.shape[-1] == DIM_1:
            output_image = image[:, :, 0]
        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(path, output_image):
            raise ValueError(
                f'Output directory {os.path.dirname(path)} does not exist')
    else:
        raise TypeError(
            f'Cannot write image with type {image.dtype}')
