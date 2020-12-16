"""This module defines class for storing command line parameters"""
from keyword import iskeyword
from typing import Any, Dict


class Params:
    """ Params class stores command line parameters as dynamic attributes """

    def __init__(self, mapping: Dict[str, Any]):
        self.__data = dict()

        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f'parameter name must be {repr(str)}')

            # check whether key is a valid Python identifier
            if not str.isidentifier(key):
                raise NameError(f'wrong name for an attribute: {key}')

            # check whether key does not collide with python keywords
            if iskeyword(key):
                raise NameError(f'Python keyword {key} cannot be used')

            self.__data[key] = value

    def __getattr__(self, name: str) -> Any:
        if name in self.__data:
            return self.__data[name]
        raise AttributeError(f'there is no attribute {name}')

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f'{class_name}({self.__data})'

    def __str__(self) -> str:
        params_str = str()
        for key, value in self.__data.items():
            params_str += f'{key} : {value}\n'
        return params_str
