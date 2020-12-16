# pylint: disable=missing-module-docstring (C0114)
import os
from typing import Tuple

import numpy as np

from core import DIM_3
from utils.cs_conversion import ChannelRange

DATA_DIR = 'data'
TEST_DIR = 'tests'

MUNICH_1_PATH = os.path.join(DATA_DIR, 'munich_1.png')
MUNICH_1_GRAY_PATH = os.path.join(DATA_DIR, 'munich_1_gray.png')
MUNICH_2_PATH = os.path.join(DATA_DIR, 'munich_2.png')
MUNICH_2_GRAY_PATH = os.path.join(DATA_DIR, 'munich_2_gray.png')
MUNICH_3_PATH = os.path.join(DATA_DIR, 'munich_3.png')
MUNICH_4_PATH = os.path.join(DATA_DIR, 'munich_4.png')

SNOW_1_PATH = os.path.join(DATA_DIR, 'snow_1.png')
SNOW_2_PATH = os.path.join(DATA_DIR, 'snow_2.png')

HEIGHT = 4
WIDTH = 5
CHANNELS = 3
ONES_IMAGE = np.ones((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
ONES_IMAGE_FLOAT = np.ones((HEIGHT, WIDTH, CHANNELS), dtype=np.float32)

ColorType = Tuple[int, int, int]

CHANNELS_DEFAULT = (0, 1, 2)
CHANNEL_RANGES_DEFAULT = tuple([ChannelRange(0., 1.)] * DIM_3)
