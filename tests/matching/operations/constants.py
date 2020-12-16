# pylint: disable=missing-module-docstring (C0114)
import numpy as np

TEST_SRC_IMAGE = np.array([[200, 180, 170, 165],
                           [190, 180, 165, 10],
                           [185, 170, 10, 7],
                           [180, 9, 8, 7]], dtype=np.uint8)

TEST_REF_IMAGE = np.array([[50, 40, 40, 30, 30],
                           [45, 45, 45, 40, 120],
                           [40, 40, 45, 115, 130],
                           [35, 35, 120, 125, 140],
                           [30, 125, 120, 145, 150]], dtype=np.uint8)

# expected resulting image with matching proportion 0.2
TEST_RES_IMAGE_02 = np.array([[190., 168.65625, 154.125, 140.625],
                              [180.4375, 168.65625, 140.625, 15.875],
                              [173.875, 154.125, 15.875, 11.6625],
                              [168.65625, 14.45, 13.24375, 11.6625]],
                             dtype=float)

# expected resulting image with matching proportion 0.5
TEST_RES_IMAGE_05 = np.array([[175., 151.640625, 130.3125, 104.0625],
                              [166.09375, 151.640625, 104.0625, 24.6875],
                              [157.1875, 130.3125, 24.6875, 18.65625],
                              [151.640625, 22.625, 21.109375, 18.65625]],
                             dtype=float)

# expected resulting image with matching proportion 0.8
TEST_RES_IMAGE_08 = np.array([[160., 134.625, 106.5, 67.5],
                              [151.75, 134.625, 67.5, 33.5],
                              [140.5, 106.5, 33.5, 25.65],
                              [134.625, 30.8, 28.975, 25.65]],
                             dtype=float)
