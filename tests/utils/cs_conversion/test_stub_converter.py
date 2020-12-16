# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import numpy as np
import pytest

from .stub_converter import StubConverter

HEIGHT = WIDTH = 3
TEST_IMAGE = 0.123 * np.ones((HEIGHT, WIDTH)).astype(np.uint8)


@pytest.fixture(name='stub_converter')
def fixture_stub_operation() -> StubConverter:
    return StubConverter()


def test_convert(stub_converter: StubConverter) -> None:
    result = stub_converter.convert(TEST_IMAGE)
    expected_result = TEST_IMAGE * 0.21
    np.testing.assert_array_equal(result, expected_result)


def test_convert_back(stub_converter: StubConverter) -> None:
    result = stub_converter.convert_back(TEST_IMAGE)
    expected_result = TEST_IMAGE * 0.22
    np.testing.assert_array_equal(result, expected_result)


def test_channel_ranges(stub_converter: StubConverter) -> None:
    ranges = stub_converter.target_channel_ranges()
    for interval in ranges:
        assert interval.min == 0.0
        assert interval.max == 1.0
