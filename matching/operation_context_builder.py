"""This module contains a function that creates the operation context"""
from core import FDM, HM, Params
from utils.cs_conversion.cs_converter_builder import build_cs_converter

from . import Operation
from .operation_context import OperationContext
from .operations import FeatureDistributionMatching, HistogramMatching


def build_operation_context(matching_type: str,
                            params: Params) -> OperationContext:
    """Creates OperationContext for specified matching type and params"""
    converter = build_cs_converter(params.color_space)
    channel_ranges = converter.target_channel_ranges()

    channels = tuple(int(c) for c in params.channels.split(','))

    if matching_type == HM:
        operation: Operation = \
            HistogramMatching(channels,
                              check_input=params.verify_input,
                              match_prop=params.match_proportion)
    elif matching_type == FDM:
        operation = \
            FeatureDistributionMatching(channels,
                                        check_input=params.verify_input,
                                        channel_ranges=channel_ranges)
    else:
        raise ValueError(f'there is no matching operation for {matching_type}')

    return OperationContext(converter, operation)
