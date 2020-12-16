"""This module defines Context for launching matching operations"""
import numpy as np

from utils.cs_conversion import ColorSpaceConverter

from . import Operation


class OperationContext:
    """ This class executes a given matching operation with a given
    color space converter """

    def __init__(self, converter: ColorSpaceConverter,
                 operation: Operation) -> None:
        self.converter = converter
        self.operation = operation

    @property
    def converter(self) -> ColorSpaceConverter:
        """ Returns the color converter """
        return self._converter

    @converter.setter
    def converter(self, converter: ColorSpaceConverter) -> None:
        if isinstance(converter, ColorSpaceConverter):
            self._converter = converter
        else:
            raise TypeError(
                f'converter has to be of {repr(ColorSpaceConverter)} type')

    @property
    def operation(self) -> Operation:
        """ Returns the matching operation """
        return self._operation

    @operation.setter
    def operation(self, operation: Operation) -> None:
        if isinstance(operation, Operation):
            self._operation = operation
        else:
            raise TypeError(
                f'converter has to be of {repr(Operation)} type')

    def __call__(self, source: np.ndarray,
                 reference: np.ndarray) -> np.ndarray:
        """ Operation process flow """
        source = self.converter.convert(source)
        reference = self.converter.convert(reference)
        result = self.operation(source, reference)
        result = self.converter.convert_back(result)
        return result
