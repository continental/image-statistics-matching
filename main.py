"""This module creates a command line interface and is an entry point"""
from functools import wraps
from typing import Any, Callable

import click

from core import (FDM, GRAY, HM, HSV, IMAGE_CHANNELS, LAB, MATCH_FULL,
                  MATCH_ZERO, RGB, Params)
from utils import application


def command_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Command wrapper, parameters that are common for all commands are set here
    :param func: command function to be wrapped
    :return: command function wrapped / decorated with the common parameters
    """

    @wraps(func)
    @click.option('--color-space', '-s', 'color_space', default=RGB,
                  type=click.Choice([GRAY, HSV, LAB, RGB],
                                    case_sensitive=False),
                  help='color space')
    @click.option('--channels', '-c', 'channels', default=IMAGE_CHANNELS,
                  type=click.Choice(
                      ['0', '1', '2', '0,1', '0,2', '1,2', '0,1,2']),
                  help='comma-separated list of channels to matching '
                       f'(supported channels are {IMAGE_CHANNELS})')
    @click.option('--plot', '-p', 'plot', is_flag=True, default=False,
                  help='creates the visualization for an applied operation')
    @click.option('--verify-input', '-v', 'verify_input', default=False,
                  type=bool, help='input data verification')
    @click.argument('source_path', type=click.Path(exists=True))
    @click.argument('reference_path', type=click.Path(exists=True))
    @click.argument('result_path', type=click.Path(exists=False))
    def wrapper(ctx: click.core.Context, **kwargs: Any) -> Any:
        for name, value in kwargs.items():
            ctx.obj[name] = value
        run(ctx)
        return func()

    return wrapper


# pylint: disable=no-value-for-parameter (E1120)
@click.group()
@click.pass_context
def main(ctx: click.core.Context) -> None:
    """ Defines a group of commands """
    ctx.ensure_object(dict)


@main.command(name=FDM, help='Feature Distribution Matching')
@click.pass_context
@command_wrapper
def command_fdm() -> None:
    """ Feature Distribution Matching command function """


@main.command(name=HM, help='Histogram Matching')
@click.option('--match-proportion', '-m', 'match_proportion',
              default=MATCH_FULL,
              type=click.FloatRange(MATCH_ZERO, MATCH_FULL),
              help=f'interpolation strength between source and reference '
                   f'histograms, {MATCH_FULL} (default) is full matching, '
                   f'{MATCH_ZERO} is no matching')
@click.pass_context
@command_wrapper
def command_hm() -> None:
    """ Histogram Matching command function """


def run(ctx: click.core.Context) -> None:
    """ Calls the main program with the parameters available in the context """
    application.run(ctx.command.name, Params(ctx.obj))


if __name__ == '__main__':
    main()
