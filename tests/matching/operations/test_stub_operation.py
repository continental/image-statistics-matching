# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
# pylint: disable=protected-access (W0212)
import inspect
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from tests import CHANNELS_DEFAULT, ONES_IMAGE

from .stub_operation import \
    fixture_stub_operation  # noqa pylint: disable=unused-import (W0611)
from .stub_operation import StubOperationMul, StubOperationSum


def test_design() -> None:
    assert inspect.isabstract(StubOperationMul) is False
    assert len(StubOperationMul.__mro__) == 4


def test_stub_operation_mul_apply(stub_operation: StubOperationMul) -> None:
    source = 0.1 * ONES_IMAGE
    reference = 0.2 * ONES_IMAGE

    original_source = np.copy(source)
    original_reference = np.copy(reference)

    result = stub_operation._apply(source, reference)

    expected_result = source * reference

    np.testing.assert_array_equal(result, expected_result)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(reference, original_reference)


def test_stub_operation_sum_apply() -> None:
    stub_operation = StubOperationSum(CHANNELS_DEFAULT, True)
    source = 0.1 * ONES_IMAGE
    reference = 0.2 * ONES_IMAGE

    original_source = np.copy(source)
    original_reference = np.copy(reference)

    result = stub_operation._apply(source, reference)

    expected_result = source + reference

    np.testing.assert_array_equal(result, expected_result)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(reference, original_reference)


def test_channel_property_getter(stub_operation: StubOperationMul) -> None:
    assert stub_operation.channels == CHANNELS_DEFAULT


def test_channel_property_setter(stub_operation: StubOperationMul) -> None:
    # try to pass a wrong data type to constructor
    with pytest.raises(TypeError):
        StubOperationMul(dict(), True)  # type: ignore

    # call setter with a wrong data type
    with pytest.raises(TypeError):
        stub_operation.channels = list()  # type: ignore

    with pytest.raises(TypeError):
        stub_operation.channels = tuple([1, 2.5, 3])  # type: ignore

    with pytest.raises(TypeError):
        stub_operation.channels = tuple([0, '1', 2])  # type: ignore

    channels = (0, 1)
    stub_operation.channels = channels
    assert stub_operation.channels == channels


def test_check_input_property_getter(stub_operation: StubOperationMul) -> None:
    assert stub_operation.check_input


def test_check_input_property_setter(stub_operation: StubOperationMul) -> None:
    stub_operation.check_input = False
    assert not stub_operation.check_input

    default_op = StubOperationMul(CHANNELS_DEFAULT)
    assert not default_op.check_input


def test_verify_input(stub_operation: StubOperationMul) -> None:
    data_234 = np.ones((2, 3, 4))
    data_2345 = np.ones((2, 3, 4, 5))
    with pytest.raises(ValueError):
        stub_operation._verify_input(data_234, data_2345)

    with pytest.raises(ValueError):
        stub_operation._verify_input(data_2345, data_2345)

    data_235 = np.ones((2, 3, 5))
    with pytest.raises(ValueError):
        stub_operation._verify_input(data_234, data_235)


def test_verify_input_channels() -> None:
    stub_operation = StubOperationMul((0, 1, 4), True)
    data_234 = np.ones((2, 3, 4))
    data_564 = np.ones((5, 6, 4))
    with pytest.raises(IndexError):
        stub_operation._verify_input(data_234, data_564)

    stub_operation.channels = (0, 1, -4)
    with pytest.raises(IndexError):
        stub_operation._verify_input(data_234, data_564)


@pytest.mark.parametrize('data_type',
                         [float, np.float, np.float32, np.float64])
def test_verify_input_type_valid(data_type: np.dtype,
                                 stub_operation: StubOperationMul) -> None:
    data_233 = np.ones((2, 3, 3), dtype=data_type)
    stub_operation._verify_input(data_233, data_233)


@pytest.mark.parametrize('data_type',
                         [np.int64, np.int32, np.uint8, np.short, np.bool])
def test_verify_input_type_invalid(data_type: np.dtype,
                                   stub_operation: StubOperationMul) -> None:
    data_233 = np.ones((2, 3, 3), dtype=float)
    source = reference = data_233

    # test source data type
    source = source.astype(data_type)
    with pytest.raises(TypeError):
        stub_operation._verify_input(source, reference)

    # test reference data type
    source = data_233
    reference = reference.astype(data_type)
    with pytest.raises(TypeError):
        stub_operation._verify_input(source, reference)


@patch.object(StubOperationMul, StubOperationMul._apply.__name__)
@patch.object(StubOperationMul, StubOperationMul._verify_input.__name__)
def test_call(verify_input: MagicMock,
              apply: MagicMock,
              stub_operation: StubOperationMul) -> None:
    # create a mock object
    mock = Mock()
    mock_verify_input = 'mock_verify_input'
    mock_apply = 'mock_apply'

    mock.attach_mock(verify_input, mock_verify_input)
    mock.attach_mock(apply, mock_apply)

    source = reference = np.ones((2, 3, 4))

    # run the function to be tested with input verification
    stub_operation(source, reference)

    # check the call order
    expected_calls = [mock_verify_input, mock_apply]
    calls = [call[0] for call in mock.mock_calls]

    assert calls == expected_calls

    # reset the call history
    mock.reset_mock()

    # run the function to be tested without input verification
    stub_operation.check_input = False
    stub_operation(source, reference)

    expected_calls = [mock_apply]
    calls = [call[0] for call in mock.mock_calls]

    assert calls == expected_calls
