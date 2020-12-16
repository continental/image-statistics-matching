# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import numpy as np
import pytest

from matching import Operation
from tests import CHANNELS_DEFAULT


class StubOperationMul(Operation):
    """
    Stub Operation class for stub testing
    """

    # pylint: disable=too-few-public-methods (R0903)

    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        return source * reference


class StubOperationSum(Operation):
    """
    Stub Operation class for stub testing
    """

    # pylint: disable=too-few-public-methods (R0903)

    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        return source + reference


@pytest.fixture(name='stub_operation')
def fixture_stub_operation() -> StubOperationMul:
    return StubOperationMul(CHANNELS_DEFAULT, True)
