# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
from unittest.mock import MagicMock, Mock, patch

import pytest

from matching.operation_context import OperationContext
from tests import CHANNELS_DEFAULT, ONES_IMAGE
from tests.utils.cs_conversion import StubConverter
from utils.cs_conversion.cs_rgb_to_lab import RgbToLabConverter

from .operations.stub_operation import StubOperationMul, StubOperationSum


@pytest.fixture(name='context')
def fixture_context() -> OperationContext:
    return OperationContext(StubConverter(),
                            StubOperationMul(CHANNELS_DEFAULT, True))


def test_converter_property_getter(context: OperationContext) -> None:
    assert isinstance(context.converter, StubConverter)


def test_converter_property_setter(context: OperationContext) -> None:
    # try to pass a wrong data type to constructor
    with pytest.raises(TypeError):
        OperationContext(dict(),  # type: ignore
                         StubOperationMul(CHANNELS_DEFAULT, True))

    # call setter with a wrong data type
    with pytest.raises(TypeError):
        context.converter = list()  # type: ignore

    context.converter = RgbToLabConverter()
    assert isinstance(context.converter, RgbToLabConverter)


def test_operation_property_getter(context: OperationContext) -> None:
    assert isinstance(context.operation, StubOperationMul)


def test_operation_property_setter(context: OperationContext) -> None:
    # try to pass a wrong data type to constructor
    with pytest.raises(TypeError):
        OperationContext(StubConverter(), dict())  # type: ignore

    # call setter with a wrong data type
    with pytest.raises(TypeError):
        context.operation = list()  # type: ignore

    context.operation = StubOperationSum(CHANNELS_DEFAULT, True)
    assert isinstance(context.operation, StubOperationSum)


@patch.object(StubConverter, StubConverter.convert.__name__)
@patch.object(StubOperationMul, StubOperationMul.__call__.__name__)
@patch.object(StubConverter, StubConverter.convert_back.__name__)
def test_call_order(convert_back: MagicMock,
                    call: MagicMock,
                    convert: MagicMock,
                    context: OperationContext) -> None:
    # create a mock object
    mock = Mock()
    mock_convert = 'mock_convert'
    mock_call = 'mock_call'
    mock_convert_back = 'mock_convert_back'

    mock.attach_mock(convert, mock_convert)
    mock.attach_mock(call, mock_call)
    mock.attach_mock(convert_back, mock_convert_back)

    # run the function to be tested
    context(ONES_IMAGE, ONES_IMAGE)

    # check the call order
    expected_calls = [mock_convert,
                      mock_convert,
                      mock_call,
                      mock_convert_back]

    calls = [call[0] for call in mock.mock_calls]
    assert calls == expected_calls
