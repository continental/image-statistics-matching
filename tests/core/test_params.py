# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
# pylint: disable=eval-used (W0123)
import pytest

from core import Params


@pytest.fixture(name='params')
def fixture_params() -> Params:
    mapping = {'param_1': 10, 'param_2': 'abc', 'param_3': 2.45}
    return Params(mapping)


@pytest.fixture(name='data_member_name')
def fixture_data_member_name() -> str:
    return '_' + Params.__name__ + '__data'


def test_params(params: Params) -> None:
    assert params.param_1 == 10
    assert params.param_2 == 'abc'
    assert params.param_3 == 2.45

    with pytest.raises(AttributeError):
        assert params.non_existing == 12


def test_params_wrong_name_type() -> None:
    mapping = {'a': 1, 'b': 2, 3: 'xyz'}
    with pytest.raises(TypeError):
        assert Params(mapping)  # type: ignore


def test_params_wrong_identifier() -> None:
    mapping = {'a': 1, 'b': 2, '3.14x': 3}
    with pytest.raises(NameError):
        assert Params(mapping)


def test_params_python_keyword() -> None:
    mapping = {'a': 1, 'b': 2, 'import': 3}
    with pytest.raises(NameError):
        assert Params(mapping)


def test_params_dict(data_member_name: str) -> None:
    mapping_1 = {'a': 1, 'b': 2, 'c': 3}
    mapping_2 = {'a': 1, 'b': 2, 'c': 3}

    params_1 = Params(mapping_1)
    params_2 = Params(mapping_2)

    assert len(params_1.__dict__[data_member_name]) == \
        len(params_2.__dict__[data_member_name])
    assert params_1.__dict__[data_member_name] == \
        params_2.__dict__[data_member_name]

    mapping_2 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    params_2 = Params(mapping_2)

    assert len(params_1.__dict__[data_member_name]) != \
        len(params_2.__dict__[data_member_name])
    assert params_1.__dict__[data_member_name] != \
        params_2.__dict__[data_member_name]


def test_params_len() -> None:
    mapping_1 = {'a': 1, 'b': 2, 'c': 3}
    mapping_2 = {'a': 1, 'b': 2, 'c': 3}

    params_1 = Params(mapping_1)
    params_2 = Params(mapping_2)

    assert len(params_1) == len(params_2)
    assert len(params_1) == len(mapping_1)
    assert len(params_2) == len(mapping_2)

    mapping_2 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    params_2 = Params(mapping_2)

    assert len(params_1) != len(params_2)
    assert len(params_2) == len(mapping_2)


def test_params_repr(params: Params) -> None:
    params_repr = eval(repr(params))

    assert len(params_repr) == len(params)
    assert params_repr.param_1 == params.param_1
    assert params_repr.param_2 == params.param_2
    assert params_repr.param_3 == params.param_3


def test_params_str() -> None:
    mapping = {'a': 10, 'b': 'abc'}
    params = Params(mapping)

    # dict is a hash table, so there is no order guarantee
    params_str_1 = 'a : 10\nb : abc\n'
    params_str_2 = 'b : abc\na : 10\n'
    assert str(params) == params_str_1 or str(params) == params_str_2
