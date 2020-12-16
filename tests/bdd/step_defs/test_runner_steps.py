# pylint: disable=missing-module-docstring (C0114)
# pylint: disable=missing-function-docstring (C0116)
import os
from typing import Any, Dict

from click.testing import CliRunner
from pytest import fixture
from pytest_bdd import given, scenarios, then, when

import main
from core import HM_PLOT_FILE

OUTPUT_FILE = 'output.png'


@fixture(name='click_runner')
def runner() -> CliRunner:
    return CliRunner()


scenarios('../scenarios/runner.feature',
          example_converters=dict(matching_algorithm=str,
                                  color_space_param=str,
                                  channels_param=str,
                                  match_proportion_param=str,
                                  verify_input_param=str,
                                  plot_param=str, ))


@fixture(name='params')
def test_params() -> Dict[str, Any]:
    return dict()


@given('source image path')
def given_source_path(params: Dict[str, Any]) -> None:
    params['source_path'] = 'data/munich_1.png'


@given('gray source image path')
def given_gray_source_path(params: Dict[str, Any]) -> None:
    params['source_path'] = 'data/munich_1_gray.png'


@given('reference image path')
def given_reference_path(params: Dict[str, Any]) -> None:
    params['reference_path'] = 'data/munich_2.png'


@given('gray reference image path')
def given_gray_reference_path(params: Dict[str, Any]) -> None:
    params['reference_path'] = 'data/munich_2_gray.png'


@given('resulting image path')
def given_result_path(params: Dict[str, Any]) -> None:
    params['result_path'] = OUTPUT_FILE


@given('<matching_algorithm>')
def given_matching_algorithm(params: Dict[str, Any],
                             matching_algorithm: str) -> None:
    params['matching_algorithm'] = matching_algorithm


@given('<color_space_param>')
def given_color_space_param(params: Dict[str, Any],
                            color_space_param: str) -> None:
    params['color_space_param'] = color_space_param


@given('<channels_param>')
def given_channels_param(params: Dict[str, Any], channels_param: str) -> None:
    params['channels_param'] = channels_param


@given('<match_proportion_param>')
def given_match_proportion_param(params: Dict[str, Any],
                                 match_proportion_param: str) -> None:
    params['match_proportion_param'] = match_proportion_param


@given('<verify_input_param>')
def given_verify_input_param(params: Dict[str, Any],
                             verify_input_param: bool) -> None:
    params['verify_input_param'] = verify_input_param


@given('<plot_param>')
def given_plot_param(params: Dict[str, Any],
                     plot_param: bool) -> None:
    params['plot_param'] = plot_param


@when('calling main')
def calling_main(params: Dict[str, Any], click_runner: CliRunner) -> None:
    command = [params['matching_algorithm'],
               params['source_path'],
               params['reference_path'],
               params['result_path']]

    if 'color_space_param' in params:
        command += params['color_space_param'].split(' ')
    if 'channels_param' in params:
        command += params['channels_param'].split(' ')
    if 'match_proportion_param' in params:
        command += params['match_proportion_param'].split(' ')
    if 'verify_input_param' in params:
        command += params['verify_input_param'].split(' ')
    if 'plot_param' in params:
        command += params['plot_param'].split(' ')

    params['exception'] = click_runner.invoke(main.main, command).exception


@then('no exception is raised')
def no_error(params: Dict[str, Any]) -> None:
    assert params['exception'] is None


@then('an exception is raised')
def error(params: Dict[str, Any]) -> None:
    assert params['exception'] is not None


@then('an output file is created')
def output_exists(params: Dict[str, Any]) -> None:
    assert os.path.exists(params['result_path'])


@then('an HM plot file is created')
def hm_plot_exists() -> None:
    assert os.path.exists(HM_PLOT_FILE)


def teardown_module() -> None:
    os.remove(HM_PLOT_FILE)
    os.remove(OUTPUT_FILE)
