# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import inspect

import cv2
import numpy as np
import pytest
from skimage.exposure import match_histograms

from core import DIM_3, MATCH_FULL, MATCH_ZERO
from matching import ChannelsType
from matching.operations import HistogramMatching
from tests import CHANNELS_DEFAULT, MUNICH_1_PATH, MUNICH_2_PATH

from . import (TEST_REF_IMAGE, TEST_RES_IMAGE_02, TEST_RES_IMAGE_05,
               TEST_RES_IMAGE_08, TEST_SRC_IMAGE)

# ground truth result image with matching proportion 1.0
TEST_RES_IMAGE = match_histograms(TEST_SRC_IMAGE, TEST_REF_IMAGE)


@pytest.fixture(name='hist_match')
def fixture_histogram_matching() -> HistogramMatching:
    return HistogramMatching(CHANNELS_DEFAULT, check_input=True)


def test_design() -> None:
    assert inspect.isabstract(HistogramMatching) is False
    assert len(HistogramMatching.__mro__) == 4


@pytest.mark.parametrize('match_prop', [list(), (1, 2, 3)])
def test_hm_match_prop_type_error(match_prop: float) -> None:
    with pytest.raises(TypeError):
        HistogramMatching(CHANNELS_DEFAULT, check_input=True,
                          match_prop=match_prop)


@pytest.mark.parametrize('match_prop', ['abc'])
def test_hm_match_prop_value_error(match_prop: float) -> None:
    with pytest.raises(ValueError):
        HistogramMatching(CHANNELS_DEFAULT, check_input=True,
                          match_prop=match_prop)


@pytest.mark.parametrize('match_prop', [0, False, True, 1, '0.5', '1.0'])
def test_hm_match_prop(match_prop: float) -> None:
    hist_match = HistogramMatching(CHANNELS_DEFAULT, check_input=True,
                                   match_prop=match_prop)
    assert hist_match.match_prop == float(match_prop)


@pytest.mark.parametrize('match_prop',
                         [MATCH_ZERO, 0.2, 0.6, MATCH_FULL, '0.3', 1])
def test_hm_match_prop_valid_value(match_prop: float) -> None:
    hist_match = HistogramMatching(CHANNELS_DEFAULT, check_input=True,
                                   match_prop=match_prop)
    assert hist_match.match_prop == float(match_prop)


@pytest.mark.parametrize('match_prop', [-5., -0.2, 1.1, 8.0, '-10', '4.5', 14])
def test_hm_match_prop_invalid_value(match_prop: float) -> None:
    with pytest.raises(ValueError):
        HistogramMatching(CHANNELS_DEFAULT, check_input=True,
                          match_prop=match_prop)


def test_apply(hist_match: HistogramMatching) -> None:
    # pylint: disable=protected-access (W0212)
    hist_match.channels = (0,)
    source = TEST_SRC_IMAGE[:, :, np.newaxis].astype(float)
    reference = TEST_REF_IMAGE[:, :, np.newaxis].astype(float)
    result = hist_match(source, reference)
    expected_result = hist_match._apply(source, reference)
    np.testing.assert_array_equal(result, expected_result)


def test_match_channel(hist_match: HistogramMatching) -> None:
    # pylint: disable=protected-access (W0212)
    result = hist_match._match_channel(TEST_SRC_IMAGE, TEST_REF_IMAGE)

    # we test against scikit image histogram matching
    assert result.shape == TEST_SRC_IMAGE.shape
    np.testing.assert_array_equal(result, TEST_RES_IMAGE)


@pytest.mark.parametrize('match_prop, expected_result',
                         [(MATCH_ZERO, TEST_SRC_IMAGE),
                          (0.2, TEST_RES_IMAGE_02),
                          (0.5, TEST_RES_IMAGE_05),
                          (0.8, TEST_RES_IMAGE_08),
                          (MATCH_FULL, TEST_RES_IMAGE)])
def test_match_channel_prop(match_prop: float,
                            expected_result: np.ndarray) -> None:
    # pylint: disable=protected-access (W0212)
    hist_match = HistogramMatching(CHANNELS_DEFAULT, check_input=True,
                                   match_prop=match_prop)
    result = hist_match._match_channel(TEST_SRC_IMAGE, TEST_REF_IMAGE)

    assert result.shape == TEST_SRC_IMAGE.shape
    np.testing.assert_array_almost_equal(result, expected_result)


@pytest.mark.parametrize('source_path, reference_path',
                         [(MUNICH_1_PATH, MUNICH_2_PATH),
                          (MUNICH_2_PATH, MUNICH_1_PATH),
                          (MUNICH_1_PATH, MUNICH_1_PATH)])
def test_match_channel_images(source_path: str, reference_path: str,
                              hist_match: HistogramMatching) -> None:
    # pylint: disable=protected-access (W0212)
    source = cv2.imread(source_path)
    reference = cv2.imread(reference_path)

    for channel in range(source.shape[-1]):
        source_c = source[:, :, channel]
        reference_c = reference[:, :, channel]
        result = hist_match._match_channel(source_c, reference_c)

        # we test against scikit image histogram matching
        expected_result = match_histograms(source_c, reference_c)

        assert result.shape == source_c.shape
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize('channels',
                         [CHANNELS_DEFAULT, (0, 1), (1,), (0, 2), (1, 2)])
def test_apply_channels(channels: ChannelsType) -> None:
    source = cv2.imread(MUNICH_1_PATH)
    reference = cv2.imread(MUNICH_2_PATH)

    original_source = np.copy(source)
    original_reference = np.copy(reference)

    hist_match = HistogramMatching(channels, check_input=True)
    result = hist_match(source.astype(float), reference.astype(float))

    # check channels to be matched
    for channel in channels:
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(source[:, :, channel],
                                          result[:, :, channel])

    # check skipped channels
    skipped_channels = tuple(set(CHANNELS_DEFAULT) - set(channels))
    for channel in skipped_channels:
        np.testing.assert_array_equal(source[:, :, channel],
                                      result[:, :, channel])

    assert result.shape == source.shape
    assert result.dtype == np.float32

    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(reference, original_reference)


def test_apply_2d_image(hist_match: HistogramMatching) -> None:
    hist_match.channels = (0,)
    source = TEST_SRC_IMAGE[:, :, np.newaxis]
    reference = TEST_REF_IMAGE[:, :, np.newaxis]
    original_source = np.copy(source)
    original_reference = np.copy(reference)

    result = hist_match(source.astype(float), reference.astype(float))

    assert result.shape == original_source.shape
    assert result.dtype == np.float32
    assert result.ndim == DIM_3

    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(reference, original_reference)

    # we test against scikit image histogram matching
    np.testing.assert_array_equal(result, TEST_RES_IMAGE[:, :, np.newaxis])
