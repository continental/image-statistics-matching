"""This module contains code analysis and testing tools for development"""
# pylint: disable=missing-function-docstring (C0116)
import os

COMMANDS = {'autopep8': 'autopep8 --in-place --recursive '
                        '--max-line-length 79 .',
            'flake8': 'flake8',
            'isort': 'isort .',
            'mypy': 'mypy .',
            'pylint': 'find . -name \'*.py\' -print0 | xargs -0 pylint',
            'pytest': 'pytest'}


def main() -> None:
    for key, val in COMMANDS.items():
        print(f'---> CODE CHECK: {key}\n')
        os.system(val)


if __name__ == '__main__':
    main()
