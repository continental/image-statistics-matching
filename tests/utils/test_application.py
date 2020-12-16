# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import io
import os
import sys

import pytest

from core import FDM, GRAY, HSV, MATCH_FULL, RGB, Params
from tests import (MUNICH_1_GRAY_PATH, MUNICH_1_PATH, MUNICH_2_GRAY_PATH,
                   MUNICH_2_PATH)
from utils import application as app
from utils.image_io import read_image


@pytest.fixture(name='params')
def fixture_params() -> Params:
    return Params(
        {
            'color_space': GRAY,
            'channels': '0,1,2',
            'match_proportion': MATCH_FULL,
            'verify_input': True,
            'plot': False,
            'source_path': MUNICH_1_PATH,
            'reference_path': MUNICH_2_PATH,
            'result_path': 'application_run_test.png'
        }
    )


def test_run_not(params: Params) -> None:
    with pytest.raises(ValueError):
        app.run(FDM, params)


def test_no_plot_msg(params: Params) -> None:
    params.color_space = RGB  # type: ignore
    params.plot = True  # type: ignore
    captured_output = io.StringIO()
    sys.stdout = captured_output
    app.run(FDM, params)
    sys.stdout = sys.__stdout__
    expected = 'Plotting is not implemented for operation fdm!\n'
    assert captured_output.getvalue() == expected

    os.remove(params.result_path)


@pytest.mark.parametrize('color_space,source,reference',
                         [[GRAY, MUNICH_1_PATH, MUNICH_2_GRAY_PATH],
                          [GRAY, MUNICH_1_GRAY_PATH, MUNICH_2_PATH],
                          [RGB, MUNICH_1_GRAY_PATH, MUNICH_2_PATH],
                          [HSV, MUNICH_1_PATH, MUNICH_2_GRAY_PATH]])
def test_color_check_fail(color_space: str, source: str,
                          reference: str) -> None:
    source_image = read_image(source)
    reference_image = read_image(reference)
    with pytest.raises(ValueError):
        app.color_check(color_space, source_image, reference_image)
