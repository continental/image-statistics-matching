# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import cv2
import numpy as np
import pytest

from tests import MUNICH_1_PATH


@pytest.fixture(name='fxt_input_image')
def fixture_input_image() -> np.ndarray:
    image_bgr = cv2.imread(MUNICH_1_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb
