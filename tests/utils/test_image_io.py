# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import os

import numpy as np
import pytest
from PIL import Image

from tests import MUNICH_1_GRAY_PATH, MUNICH_1_PATH, TEST_DIR
from utils.image_io import MAX_VALUE_8_BIT, read_image, write_image


def test_read_image() -> None:
    expected = np.asarray(Image.open(MUNICH_1_PATH)).astype(
        np.float32) / MAX_VALUE_8_BIT
    result = read_image(MUNICH_1_PATH)
    np.testing.assert_array_equal(result, expected)

    with pytest.raises(ValueError):
        read_image('no_image.png')


def test_read_image_gray() -> None:
    expected = np.asarray(Image.open(MUNICH_1_GRAY_PATH))[:, :, np.newaxis]
    expected = expected.astype(np.float32) / MAX_VALUE_8_BIT
    result = read_image(MUNICH_1_GRAY_PATH)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('target', ['img1337.png',
                                    os.path.join(TEST_DIR, 'img1337.png')])
def test_write_image_uint8(target: str) -> None:
    image = np.asarray(Image.open(MUNICH_1_PATH))

    test_path = target
    write_image(image, test_path)

    with Image.open(test_path) as test_image:
        result = np.asarray(test_image)
    np.testing.assert_array_equal(result, image)

    os.remove(test_path)


def test_write_image_float() -> None:
    image = np.asarray(Image.open(MUNICH_1_PATH))

    test_path = 'img1337.png'
    pseudo_result = image.astype(np.float32) / MAX_VALUE_8_BIT
    write_image(pseudo_result, test_path)

    with Image.open(test_path) as test_image:
        result = np.asarray(test_image)
    np.testing.assert_array_equal(result, image)

    os.remove(test_path)


def test_write_image_invalid_path() -> None:
    image = np.asarray(Image.open(MUNICH_1_PATH))
    invalid_path = os.path.join('img1337', 'img1337.png')
    with pytest.raises(ValueError):
        write_image(image, invalid_path)


@pytest.mark.parametrize('array_type',
                         [int, float, np.int8, np.int16, np.int32,
                          np.uint16, np.uint32, np.float64])
def test_write_image_invalid_type(array_type: np.dtype) -> None:
    image = np.asarray(Image.open(MUNICH_1_PATH)).astype(array_type)
    path = 'img1337.png'
    with pytest.raises(TypeError):
        write_image(image, path)


def test_write_image_gray() -> None:
    image = np.asarray(Image.open(MUNICH_1_GRAY_PATH))

    test_path = 'img1337.png'
    write_image(image[:, :, np.newaxis], test_path)

    with Image.open(test_path) as test_image:
        result = np.asarray(test_image)
    np.testing.assert_array_equal(result, image)

    os.remove(test_path)


def test_write_image_gray_invalid_path() -> None:
    image = np.asarray(Image.open(MUNICH_1_GRAY_PATH))[:, :, np.newaxis]
    invalid_path = os.path.join('img1337', 'img1337.png')
    with pytest.raises(ValueError):
        write_image(image, invalid_path)
