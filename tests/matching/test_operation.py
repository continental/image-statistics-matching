# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
# pylint: disable=protected-access (W0212)
import abc
import inspect

from matching import Operation


def test_design() -> None:
    assert issubclass(Operation, abc.ABC) is True
    assert inspect.isabstract(Operation) is True
    assert len(Operation.__mro__) == 3
