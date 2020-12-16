"""This module implements Feature Distribution Matching operation"""
from typing import Tuple

import numpy as np

from matching import ChannelsType, Operation
from utils.cs_conversion import ChannelRange


class FeatureDistributionMatching(Operation):
    """Feature Distribution Matching operation class"""

    # pylint: disable=too-few-public-methods (R0903)

    def __init__(self, channels: ChannelsType,
                 channel_ranges: Tuple[ChannelRange, ...],
                 check_input: bool = False):
        super().__init__(channels, check_input)
        self.channel_ranges = channel_ranges

    @property
    def channel_ranges(self) -> Tuple[ChannelRange, ...]:
        """ Returns the channels of the color space """
        return self._channel_ranges

    @channel_ranges.setter
    def channel_ranges(self, channel_ranges: Tuple[ChannelRange, ...]) -> None:
        if not isinstance(channel_ranges, tuple):
            raise TypeError(
                f'channel ranges has to be of type '
                f'{repr(Tuple[ChannelRange, ...])}')

        for channel_range in channel_ranges:
            if not isinstance(channel_range, ChannelRange):
                raise TypeError(
                    f'Channel range has to be of {repr(ChannelRange)}')
        self._channel_ranges = channel_ranges

    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:

        matching_result = self._matching(source[:, :, self.channels],
                                         reference[:, :, self.channels])

        result = np.copy(source)
        # Replace selected channels with matching result
        result[:, :, self.channels] = matching_result

        # Replace selected channels
        for channel in self.channels:
            result[:, :, channel] = np.clip(result[:, :, channel],
                                            self.channel_ranges[
                                                channel].min,
                                            self.channel_ranges[
                                                channel].max)

        return result.astype(np.float32)

    @staticmethod
    def _matching(source: np.ndarray,
                  reference: np.ndarray) -> np.ndarray:
        """ Run all transformation steps """
        # 1.) reshape to feature matrix (H*W,C)
        feature_mat_src = FeatureDistributionMatching._get_feature_matrix(
            source)
        feature_mat_ref = FeatureDistributionMatching._get_feature_matrix(
            reference)

        # 2.) center (subtract mean)
        feature_mat_src, _ = FeatureDistributionMatching._center_image(
            feature_mat_src)
        feature_mat_ref, reference_mean = \
            FeatureDistributionMatching._center_image(feature_mat_ref)

        # 3.) whitening: cov(feature_mat_src) = I
        feature_mat_src_white = FeatureDistributionMatching._whitening(
            feature_mat_src)

        # 4.) transform covariance: cov(feature_mat_ref) = covariance_ref
        feature_mat_src_transformed = \
            FeatureDistributionMatching._covariance_transformation(
                feature_mat_src_white, feature_mat_ref)

        # 5.) Add reference mean
        feature_mat_src_transformed += reference_mean

        # 6.) Reshape
        result = feature_mat_src_transformed.reshape(source.shape)

        return result

    @staticmethod
    def _get_feature_matrix(image: np.ndarray) -> np.ndarray:
        """ Reshapes an image (H, W, C) to
        a feature vector (H * W, C)
        :param image: H x W x C image
        :return feature_matrix: N x C matrix with N samples and C features
        """
        feature_matrix = np.reshape(image, (-1, image.shape[-1]))
        return feature_matrix

    @staticmethod
    def _center_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Centers the image by removing mean
        :returns centered image and original mean
        """
        image = np.copy(image)
        image_mean = np.mean(image, axis=0)
        image -= image_mean
        return image, image_mean

    @staticmethod
    def _whitening(feature_mat: np.ndarray) -> np.ndarray:
        """
        Transform the feature matrix so that cov(feature_map) = Identity or
        if the feature matrix is one dimensional so that var(feature_map) = 1.
        :param feature_mat: N x C matrix with N samples and C features
        :return feature_mat_white: A corresponding feature vector with an
        identity covariance matrix or variance of 1.
        """
        if feature_mat.shape[1] == 1:
            variance = np.var(feature_mat)
            feature_mat_white = feature_mat / np.sqrt(variance)
        else:
            data_cov = np.cov(feature_mat, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(data_cov)
            sqrt_s = np.diag(np.sqrt(s_vec))
            feature_mat_white = (feature_mat @ u_mat) @ np.linalg.inv(sqrt_s)
        return feature_mat_white

    @staticmethod
    def _covariance_transformation(feature_mat_white: np.ndarray,
                                   feature_mat_ref: np.ndarray) -> np.ndarray:
        """
        Transform the white (cov=Identity) feature matrix so that
        cov(feature_mat_transformed) = cov(feature_mat_ref). In the 2d case
        this becomes:
        var(feature_mat_transformed) = var(feature_mat_ref)
        :param feature_mat_white: input with identity covariance matrix
        :param feature_mat_ref: reference feature matrix
        :return: feature_mat_transformed with cov == cov(feature_mat_ref)
        """
        if feature_mat_white.shape[1] == 1:
            variance_ref = np.var(feature_mat_ref)
            feature_mat_transformed = feature_mat_white * np.sqrt(variance_ref)
        else:
            covariance_ref = np.cov(feature_mat_ref, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(covariance_ref)
            sqrt_s = np.diag(np.sqrt(s_vec))

            feature_mat_transformed = (feature_mat_white @ sqrt_s) @ u_mat.T
        return feature_mat_transformed
