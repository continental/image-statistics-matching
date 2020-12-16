# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import pytest

from core import FDM, HM, RGB, Params
from matching import operation_context_builder
from matching.operations import FeatureDistributionMatching, HistogramMatching
from utils.cs_conversion.cs_converter import ColorSpaceConverter


def test_operation_context_builder() -> None:
    params = Params({'color_space': RGB,
                     'channels': '0,1,2',
                     'match_proportion': 1.0,
                     'verify_input': True})
    op_ctx = operation_context_builder.build_operation_context(HM, params)
    assert isinstance(op_ctx.operation, HistogramMatching)
    assert isinstance(op_ctx.converter, ColorSpaceConverter)

    op_ctx = operation_context_builder.build_operation_context(FDM, params)
    assert isinstance(op_ctx.operation, FeatureDistributionMatching)
    assert isinstance(op_ctx.converter, ColorSpaceConverter)

    with pytest.raises(ValueError):
        operation_context_builder.build_operation_context('1337', params)
