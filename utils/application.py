"""This module provides a function to perform the matching operation"""

import numpy as np

from core import DIM_1, GRAY, HM, HM_PLOT_FILE, Params
from matching.operation_context_builder import build_operation_context
from utils.visu import histogram_matching_plot as hm_plot

from .image_io import read_image, write_image


def run(operation_type: str, params: Params) -> None:
    """
    This function gets the input images, creates the specified
    OperationContext, uses it and saves the resulting
    image
    """
    if params.color_space == GRAY and params.channels != '0':
        raise ValueError(f'{params.channels} is no valid channel selection '
                         f'for color space {GRAY}.')

    source = read_image(params.source_path)
    reference = read_image(params.reference_path)
    color_check(params.color_space, source, reference)

    op_ctx = build_operation_context(operation_type, params)
    result = op_ctx(source, reference)

    write_image(result, params.result_path)

    if params.plot:
        if operation_type == HM:
            result = read_image(params.result_path)
            images = hm_plot.Images(source, reference, result)
            hm_plot.make_plot(HM_PLOT_FILE, images, op_ctx.converter,
                              params.color_space, params.channels)
        else:
            print(
                f'Plotting is not implemented for operation {operation_type}!')


def color_check(color_space: str, source: np.ndarray,
                reference: np.ndarray) -> None:
    """
    This function checks if the input images fit to the selected color space.
    """
    if color_space == GRAY and source.shape[-1] != DIM_1:
        raise ValueError(
            f'Parameter color space is {GRAY}, but source image is colored.')
    if color_space == GRAY and reference.shape[-1] != DIM_1:
        raise ValueError(
            f'Parameter color space is {GRAY}, but reference image is '
            f'colored.')
    if color_space != GRAY and source.shape[-1] == DIM_1:
        raise ValueError(
            f'Selected color space {color_space}, but source image is '
            f'grayscale.')
    if color_space != GRAY and reference.shape[-1] == DIM_1:
        raise ValueError(
            f'Selected color space {color_space}, but reference image is '
            f'grayscale.')
