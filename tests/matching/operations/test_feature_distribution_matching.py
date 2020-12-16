# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
# pylint: disable=protected-access (W0212)
import inspect
from typing import Callable, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from matching import ChannelsType
from matching.operations import \
    FeatureDistributionMatching as FeatureDistMatching
from tests import (CHANNEL_RANGES_DEFAULT, CHANNELS_DEFAULT, MUNICH_1_PATH,
                   MUNICH_2_PATH)
from utils.cs_conversion import ChannelRange
from utils.image_io import read_image

ImageGenType = Callable[[Tuple[int, int, int], int], np.ndarray]

TEST_IMAGE_423 = np.array([[[0.09634835, 0.67985358, 0.71879272],
                            [0.2746647, 0.55942712, 0.17269985]],
                           [[0.60345517, 0.70931529, 0.14624073],
                            [0.23241476, 0., 0.91803395]],  # 0. entry
                           [[0.58758528, 0.66624122, 0.55438404],
                            [1., 0.26015386, 0.28256821]],  # 1. entry
                           [[0.87368081, 0.85794979, 0.11262025],
                            [0.70622847, 0.9368422, 0.39187311]]])

TEST_IMAGE_421 = TEST_IMAGE_423[:, :, 0:1]

TEST_IMAGE_243 = np.array([[[0.96883339, 0.6280047, 0.55266079],
                            [0.64656832, 0.96452021, 0.01043656],
                            [1., 0.75418668, 0.77878355],  # 1. entry
                            [0.80548047, 0.73128374, 0.72436705]],

                           [[0.81233289, 0.67804285, 0.10213132],
                            [0.3819547, 0.67940864, 0.8103251],
                            [0.30974552, 0.49638342, 0.],  # 0. entry
                            [0.11005092, 0.48727506, 0.31166669]]])

TEST_IMAGE_241 = TEST_IMAGE_243[:, :, 0:1]


def test_channel_ranges() -> None:
    with pytest.raises(TypeError):
        FeatureDistMatching(CHANNELS_DEFAULT, check_input=True,
                            channel_ranges=[1, 2, 3])  # type: ignore
    with pytest.raises(TypeError):
        FeatureDistMatching(CHANNELS_DEFAULT, check_input=True,
                            channel_ranges=(
                                ChannelRange(0.1, 1.7), 1,  # type: ignore
                                ChannelRange(0.1, 1.7)))


@pytest.mark.parametrize('test_image, des_shape',
                         [(TEST_IMAGE_423, (8, 3)),
                          (TEST_IMAGE_421, (8, 1))
                          ])
def test_get_feature_matrix(test_image: np.array,
                            des_shape: Tuple[int, int]) -> None:
    result = FeatureDistMatching._get_feature_matrix(test_image)
    np.testing.assert_array_equal(result, test_image.reshape(des_shape))


def test_center_image() -> None:
    test_image = TEST_IMAGE_423
    test_image_mat = FeatureDistMatching._get_feature_matrix(test_image)
    test_image_mean = np.mean(test_image_mat, axis=0)
    result_image_mat, result_mean = FeatureDistMatching._center_image(
        test_image_mat)
    np.testing.assert_array_equal(result_mean, test_image_mean)
    np.testing.assert_almost_equal(np.mean(result_image_mat, axis=0),
                                   np.zeros(3))


def test_whitening() -> None:
    test_image = TEST_IMAGE_423
    test_image_mat = FeatureDistMatching._get_feature_matrix(test_image)
    FeatureDistMatching._center_image(test_image_mat)

    result = FeatureDistMatching._whitening(test_image_mat)
    np.testing.assert_almost_equal(np.cov(result, rowvar=False),
                                   np.identity(3))


def test_whitening_2d() -> None:
    test_image = TEST_IMAGE_421
    test_image_mat = FeatureDistMatching._get_feature_matrix(test_image)
    FeatureDistMatching._center_image(test_image_mat)

    result = FeatureDistMatching._whitening(test_image_mat)
    np.testing.assert_almost_equal(np.var(result), 1.)


def test_covariance_transformation() -> None:
    test_image = TEST_IMAGE_423
    test_image_mat = FeatureDistMatching._get_feature_matrix(test_image)
    FeatureDistMatching._center_image(test_image_mat)
    test_image_white = FeatureDistMatching._whitening(test_image_mat)

    feature_mat_ref = FeatureDistMatching._get_feature_matrix(TEST_IMAGE_243)

    result = FeatureDistMatching._covariance_transformation(test_image_white,
                                                            feature_mat_ref)
    np.testing.assert_almost_equal(np.cov(result, rowvar=False),
                                   np.cov(feature_mat_ref, rowvar=False))


def test_covariance_transformation_2d() -> None:
    test_image = TEST_IMAGE_421
    test_image_mat = FeatureDistMatching._get_feature_matrix(test_image)
    FeatureDistMatching._center_image(test_image_mat)
    test_image_white = FeatureDistMatching._whitening(test_image_mat)

    feature_mat_ref = FeatureDistMatching._get_feature_matrix(
        TEST_IMAGE_243[:, :, 0])

    result = FeatureDistMatching._covariance_transformation(test_image_white,
                                                            feature_mat_ref)
    np.testing.assert_almost_equal(np.var(result),
                                   np.var(feature_mat_ref))


@pytest.fixture(name='feature_dist_matching')
def fixture_feature_dist_matching() -> FeatureDistMatching:
    return FeatureDistMatching(CHANNELS_DEFAULT,
                               CHANNEL_RANGES_DEFAULT,
                               check_input=True)


def test_design() -> None:
    assert inspect.isabstract(FeatureDistMatching) is False
    assert len(FeatureDistMatching.__mro__) == 4


@pytest.mark.parametrize('channel_range_max', [1.0, 255.])
def test_matching(channel_range_max: float) -> None:
    source = TEST_IMAGE_423
    reference = TEST_IMAGE_243
    source[:, :, 1] *= channel_range_max
    reference[:, :, 1] *= channel_range_max
    result = FeatureDistMatching._matching(source, reference)

    feature_mat_result = FeatureDistMatching._get_feature_matrix(result)
    feature_mat_ref = FeatureDistMatching._get_feature_matrix(reference)

    result_mean = np.mean(feature_mat_result, axis=0)
    reference_mean = np.mean(feature_mat_ref, axis=0)
    np.testing.assert_almost_equal(result_mean, reference_mean)

    feature_mat_result -= result_mean
    feature_mat_ref -= reference_mean
    np.testing.assert_almost_equal(np.cov(feature_mat_result, rowvar=False),
                                   np.cov(feature_mat_ref, rowvar=False))


@pytest.mark.parametrize('channel_range_max', [1.0, 255.])
def test_matching_2d(channel_range_max: float) -> None:
    source = TEST_IMAGE_421 * channel_range_max
    reference = TEST_IMAGE_421 * channel_range_max
    result = FeatureDistMatching._matching(source, reference)

    feature_mat_result = FeatureDistMatching._get_feature_matrix(result)
    feature_mat_ref = FeatureDistMatching._get_feature_matrix(reference)

    result_mean = np.mean(feature_mat_result, axis=0)
    reference_mean = np.mean(feature_mat_ref, axis=0)

    np.testing.assert_almost_equal(result_mean, reference_mean)

    feature_mat_result -= result_mean
    feature_mat_ref -= reference_mean

    np.testing.assert_almost_equal(np.var(feature_mat_result),
                                   np.var(feature_mat_ref))


@pytest.mark.parametrize('source, reference',
                         [(TEST_IMAGE_423, TEST_IMAGE_243),
                          (TEST_IMAGE_421, TEST_IMAGE_241)])
def test_apply(feature_dist_matching: FeatureDistMatching,
               source: np.array,
               reference: np.array) -> None:
    feature_dist_matching.channels = CHANNELS_DEFAULT[0:source.shape[-1]]
    feature_dist_matching.channel_ranges = CHANNEL_RANGES_DEFAULT[
        0:source.shape[-1]]
    source_copy = np.copy(source)
    reference_copy = np.copy(reference)
    result = feature_dist_matching._apply(source, reference)

    # Check modification of input:
    np.testing.assert_array_equal(source, source_copy)
    np.testing.assert_array_equal(reference, reference_copy)

    # Check result:
    assert result.shape == source.shape
    assert np.max(result) <= 1.
    assert 0.1 < np.mean(result) < 0.9
    assert np.min(result) >= 0.
    assert result.dtype == np.float32


@pytest.mark.parametrize('channels',
                         [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_apply_channels(channels: ChannelsType,
                        feature_dist_matching: FeatureDistMatching) -> None:
    source = TEST_IMAGE_423.astype(np.float32)
    reference = TEST_IMAGE_243.astype(np.float32)
    feature_dist_matching.channels = channels
    result = feature_dist_matching._apply(source, reference)

    for untouched_channel in {0, 1, 2} - set(channels):
        np.testing.assert_array_equal(source[:, :, untouched_channel],
                                      result[:, :, untouched_channel])


def test_apply_channel_range(
        feature_dist_matching: FeatureDistMatching) -> None:
    source = TEST_IMAGE_423
    reference = TEST_IMAGE_243

    source[:, :, 1] *= 255.
    source[:, :, 2] *= 254.
    source[:, :, 2] -= 127.
    reference[:, :, 1] *= 255.
    reference[:, :, 2] *= 254.
    reference[:, :, 2] -= 127.
    feature_dist_matching.channel_ranges = tuple([ChannelRange(0., 1.),
                                                  ChannelRange(0., 255.),
                                                  ChannelRange(-127.0, 127.0)])

    result = feature_dist_matching._apply(source, reference)

    np.testing.assert_array_equal(source, TEST_IMAGE_423)
    np.testing.assert_array_equal(reference, TEST_IMAGE_243)
    assert result.shape == source.shape
    assert result.dtype == np.float32

    assert np.max(result[:, :, 0]) <= 1.
    assert 0.1 < np.mean(result[:, :, 0]) < 0.9
    assert np.max(result[:, :, 0]) >= 0.

    assert np.max(result[:, :, 1]) <= 255.
    assert 5. < np.mean(result[:, :, 1]) < 250.
    assert np.min(result[:, :, 1]) >= 0.

    assert np.max(result[:, :, 2]) <= 127.
    assert -120. < np.mean(result[:, :, 2]) < 120.
    assert np.min(result[:, :, 2]) >= -127.


@patch.object(FeatureDistMatching,
              FeatureDistMatching._get_feature_matrix.__name__)
@patch.object(FeatureDistMatching,
              FeatureDistMatching._center_image.__name__,
              return_value=(np.ones((2, 3)), np.ones((2, 3))))
@patch.object(FeatureDistMatching, FeatureDistMatching._whitening.__name__)
@patch.object(FeatureDistMatching,
              FeatureDistMatching._covariance_transformation.__name__)
def test_matching_order(covariance_transformation: MagicMock,
                        whitening: MagicMock,
                        center_image: MagicMock,
                        get_feature_matrix: MagicMock,
                        feature_dist_matching: FeatureDistMatching) -> None:
    mock = Mock()
    mock_get_feature_matrix = 'mock_get_feature_matrix'
    mock_center_image = 'mock_center_image'
    mock_whitening = 'mock_whitening'
    mock_covariance_transformation = 'mock_covariance_transformation'

    mock.attach_mock(get_feature_matrix, mock_get_feature_matrix)
    mock.attach_mock(center_image, mock_center_image)
    mock.attach_mock(whitening, mock_whitening)
    mock.attach_mock(covariance_transformation, mock_covariance_transformation)

    source = TEST_IMAGE_423
    reference = TEST_IMAGE_243
    feature_dist_matching._matching(source, reference)

    # check the call order
    expected_calls = [mock_get_feature_matrix,
                      mock_get_feature_matrix,
                      mock_center_image,
                      mock_center_image,
                      mock_whitening,
                      mock_covariance_transformation,
                      mock_covariance_transformation + '().__iadd__',
                      mock_covariance_transformation + '().__iadd__().reshape']

    calls = [call[0] for call in mock.mock_calls]
    assert calls == expected_calls


@pytest.mark.parametrize('source_path, reference_path',
                         [(MUNICH_1_PATH, MUNICH_2_PATH),
                          (MUNICH_2_PATH, MUNICH_1_PATH),
                          (MUNICH_1_PATH, MUNICH_1_PATH)])
def test_apply_images(source_path: str, reference_path: str,
                      feature_dist_matching: FeatureDistMatching) -> \
        None:
    source = read_image(source_path)
    reference = read_image(reference_path)
    result = feature_dist_matching._apply(source, reference)
    assert result.shape == source.shape
    assert np.max(result) <= 1.
    assert 0.1 < np.mean(result) < 0.9
    assert np.min(result) >= 0.
    assert result.dtype == np.float32
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(result, source)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(result, reference)
