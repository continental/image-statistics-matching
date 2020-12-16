# image-statistics-matching

## Preface

This repository contains a Python implementation of **Feature Distribution Matching** and **Histogram Matching** methods published in
[Keep it Simple: Image Statistics Matching for Domain Adaptation](https://arxiv.org/abs/2005.12551) at the CVPR workshop on Scalability
in Autonomous Driving 2020. Both methods are based on the alignment of global image statistics and were originally aimed at unsupervised
Domain Adaptation for object detection (see the paper for more details). They also can be considered as data augmentation techniques.

All software components in this repository were designed with a clear focus on scalability and extensibility, so that new image matching
operations can be added with minimal effort.

## Installation

### Requirements

- Ubuntu Linux 16.04, 18.04, 20.04, Windows or MacOS
- Python version >= 3.6

### Install image-statistics-matching

We encourage you to do the installation within a `conda` environment with the latest supported Python version. In this guidance we presume that you
already have `Anaconda` or `miniconda` installed. For reference, we run all commands on Ubuntu Linux and use a Python 3.8 `conda` environment.

1. create a conda virtual environment and activate it:

```sh
>>> conda create -n <ENV_NAME> python=3.8 -y
>>> conda activate <ENV_NAME>
```

2. clone the `image-statistics-matching` repository:

```sh
>>> git clone https://github.com/continental/image-statistics-matching.git
>>> cd image-statistics-matching
```

3. install all dependencies:

```sh
>>> pip install -U -r requirements.txt
```

4. run all tests and make sure all of them are passed:

```sh
>>> pytest
```

## Run Operations

### Available Operations

All image matching operations are implemented as commands in [Click](https://palletsprojects.com/p/click/) command line interface.
You can list commands for all available image matching operations by

```sh
>>> python main.py --help
```

| Acronym | Operation | Command name | Command options |
|:-:|:-:|:-:|:-:|
| **FDM** | Feature Distribution Matching | `fdm` | `python main.py fdm --help` |
| **HM** | Histogram Matching | `hm` | `python main.py hm --help` |

Each command has the following format:

```sh
>>> python main.py <COMMAND> <COMMAND OPTIONS> <SOURCE IMAGE> <REFERENCE IMAGE> \
                   <RESULT IMAGE>
```

`<COMMAND>` - command name from the table above

`<COMMAND OPTIONS>` - options for `<COMMAND>`

`<SOURCE IMAGE>` - path to a source image (mandatory)

`<REFERENCE IMAGE>` - path to a reference image (mandatory), in the context of Domain Adaptation it is an image from a target domain (see [the paper](#citation)
for more details)

`<RESULT IMAGE>` - path to a resulting image (mandatory)

All image matching operations can run in various color spaces, the supported color spaces are

- [x] **GRAY**: grayscale, this color space should be used in case source and reference images are grayscale
- [x] **HSV**: **H**ue, **S**aturation (shades of the color), **V**alue (intensity)
- [x] **LAB**: **L**ightness (intensity), **A** – color from Green to Magenta, **B** – color from Blue to Yellow
- [x] **RGB**: additive color space where colors are obtained by a linear combination of **R**ed, **G**reen, and **B**lue values

### Feature Distribution Matching operation

Feature Distribution Matching (**FDM**) transforms a source image in such a way that it obtains the color mean and covariance of the reference image, while retaining the source
image content. Instead of a transformation in homogeneous coordinates, **FDM** generalizes the transformation to the c-dimensional Euclidean space (see [the paper](#citation)
for more details).

Apply **FDM** in the **RGB** color space to all channels of a source image [data/munich_1.png](data/munich_1.png) taking [data/munich_2.png](data/munich_2.png)
as a reference image:

```sh
>>> python main.py fdm --color-space rgb --channels 0,1,2 data/munich_1.png data/munich_2.png \
                       output.png
```

or using its shorter form:

```sh
>>> python main.py fdm -s rgb -c 0,1,2 data/munich_1.png data/munich_2.png output.png
```

![FeatureDistributionMatching Image](/docs/fdm_rgb_012.png)

Matching feature distributions directly in the default **RGB** color space does not always give the desired results due to the strong correlation between
luminance and color information in all three channels. On the other hand, in the **LAB** color space **L**ightness is independent of color information,
so that we can apply **FDM** in the **LAB** color space to **A** and **B** channels of a source image [data/snow_1.png](data/snow_1.png) taking
[data/munich_3.png](data/munich_3.png) as a reference image:

```sh
>>> python main.py fdm --color-space lab --channels 1,2 data/snow_1.png data/munich_3.png \
                       output.png
```

or using its shorter form:

```sh
>>> python main.py fdm -s lab -c 1,2 data/snow_1.png data/munich_3.png output.png
```

![FeatureDistributionMatching Image](/docs/fdm_lab_12.png)

### Histogram Matching operation

**H**istogram **M**atching (**HM**) is a common approach in image processing for finding a monotonic mapping between a pair of image histograms.
It manipulates pixels of a source image in such a way that its histogram matches that of a reference image (see [the paper](#citation) for more details).

Apply **HM** in the **RGB** color space to all channels of a source image [data/snow_2.png](data/snow_2.png) taking [data/munich_2.png](data/munich_2.png)
as a reference image with a matching strength of `1.0` (`1.0` is full match, `0.0` is no match) and plot the results:

```sh
>>> python main.py hm --match-proportion 1.0 --color-space rgb --channels 0,1,2 --plot \
                      data/snow_2.png data/munich_2.png output.png
```

or using its shorter form:

```sh
>>> python main.py hm -m 1.0 -s rgb -c 0,1,2 -p data/snow_2.png data/munich_2.png output.png
```

![HistogramMatching Image](/docs/hm_plot_rgb_012_1_0.png)

Matching histograms directly in the default **RGB** color space does not always give the desired results due to the strong correlation between
luminance and color information in all three channels. On the other hand, in the **LAB** color space **L**ightness is independent of color information,
so that we can apply **HM** in the **LAB** color space to **A** and **B** channels of a source image [data/munich_2.png](data/munich_2.png) taking
[data/munich_4.png](data/munich_4.png) as a reference image:

```sh
>>> python main.py hm --color-space lab --channels 1,2 --plot data/munich_2.png \
                      data/munich_4.png output.png
```

or using its shorter form:

```sh
>>> python main.py hm -s lab -c 1,2 -p data/munich_2.png data/munich_4.png output.png
```

![HistogramMatching Image](/docs/hm_plot_lab_12.png)

## Contributing

All kinds of contributions are kindly welcome:

* fixes (typos, bugs)
* new matching operations and image converters

If you find a bug or have a feature request, post an issue at [image-statistics-matching/issues](https://github.com/continental/image-statistics-matching/issues).

### Workflow

1. fork the repository
2. clone it
3. install `pre-commit` hook and initialize it from the directory with the repository:

```sh
>>> pre-commit install
```

4. make desired changes to the code and provide tests that assure the correctness of new features and modules

5. run tests:

```sh
>>> pytest
```

6. run `code_checker.py` to assure the code quality every time you apply or add some changes:

```sh
>>> python code_checker.py
```

7. push code to your forked repository

8. create a pull request and request a review

### Code Style

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting, formatting and testing:
- [autopep8](https://github.com/hhatto/autopep8): formatter
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [isort](https://github.com/timothycrosley/isort): sort imports
- [mypy](https://github.com/python/mypy): static type checker
- [pylint](https://github.com/PyCQA/pylint/): linter
- [pytest](https://docs.pytest.org/en/stable/): for testing
- [pytest-bdd](https://github.com/pytest-dev/pytest-bdd) for testing in Behavioural Driven Development (BDD) manner
- [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) for producing test coverage reports

The [pre-commit hook](https://pre-commit.com/) checks and formats for `autopep8`, `flake8`, `isort`, `mypy`, `pylint`, `check-added-large-files`,
`check-docstring-first`, `check-yaml`, `debug-statements`, `double-quote-string-fixer`, `end-of-file-fixer`, `trailing whitespaces`, `requirements-txt-fixer`,
`end-of-files`, runs unit tests with `pytest` automatically on every commit. The config for a pre-commit hook is stored in [.pre-commit-config](.pre-commit-config.yaml).

### Software Design

![Architecture Image](/docs/image-statistic-matching_uml.png)

#### Implementing New Matching Operation

Image matching operations are located in [matching/operations](matching/operations) and implement `Operation` interface from [matching/operation.py](matching/operation.py).
To implement a new matching operation `NewOperation` you need to create a new Python module:

```sh
>>> cd matching/operations
>>> echo > new_operation.py
```

and implement the abstract method `_apply` from the `Operation` interface:


```python
import numpy as np

from matching import ChannelsType, Operation


class NewOperation(Operation):

    def __init__(self, channels: ChannelsType, check_input: bool = True,
                 a: A = DEFAULT_A, b: B = DEFAULT_B, ...):
        # base class (Operation) constructor
        super().__init__(channels, check_input)

        # parameters specific for NewOperation
        self.a = a
        self.b = b
        ...

    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        # matching operation implementation
        ...
```

During the development and testing we strongly encourage you to set `check_input` flag to `True`. This will call the `_verify_input` function from
[matching/operation.py](matching/operation.py) every time you apply the operation in order to ensure that your input data has correct types and dimensionality.

Unit tests for image matching operations are located in [tests/matching/operations](tests/matching/operations). Put all necessary tests assuring the correctness of `NewOperation`
into a separate test module:

```sh
>>> cd tests/matching/operations
>>> echo > test_new_operation.py
```

Add a command name for `NewOperation` to [core/constants.py](core/constants.py):

```python
...
NEW_OP = 'new_op'
...
```

Add [Click](https://palletsprojects.com/p/click/) command for `NewOperation` to the command line interface in [main.py](main.py):

```python
...
from core import NEW_OP
...
@main.command(name=NEW_OP, help='New Operation')
@click.option(...)
...
@click.pass_context
@command_wrapper
def command_new_operation() -> None:
    """ New Operation command function """
```

Add `NewOperation` to [matching/operation_context_builder.py](matching/operation_context_builder.py):

```python
...
from core import NEW_OP
...
from .operations import NewOperation
...
    elif matching_type == NEW_OP:
        operation = \
            NewOperation(channels,
                         check_input=params.verify_input,
                         a=params.a,
                         b=params.b,
                         ...)
...
```

Now you should be able to run `NewOperation` from the [Click](https://palletsprojects.com/p/click/) command line interface:

```sh
>>> python main.py new_op data/munich_1.png data/munich_2.png output.png
```

For more implementation details see the existing image matching operations:

* [FeatureDistributionMatching](matching/operations/feature_distribution_matching.py)
  * [Tests for FeatureDistributionMatching](tests/matching/operations/test_feature_distribution_matching.py)
* [HistogramMatching](matching/operations/histogram_matching.py)
  * [Tests for HistogramMatching](tests/matching/operations/test_histogram_matching.py)
* [StubOperationMul and StubOperationSum](tests/matching/operations/stub_operation.py) - only for testing
  * [Tests for StubOperationMul and StubOperationSum](tests/matching/operations/test_stub_operation.py)

#### Implementing New Color Space Converter

Color space converters are located in [utils/cs_conversion](utils/cs_conversion) and implement `ColorSpaceConverter` interface from [utils/cs_conversion/cs_converter.py](utils/cs_conversion/cs_converter.py).
To implement a new color space converter `RgbToNewColorSpaceConverter` you need to create a new Python module:

```sh
>>> cd utils/cs_conversion
>>> echo > cs_rgb_to_new_color_space.py
```

and implement the abstract methods `convert`, `convert_back`, and `target_channel_ranges` from the `ColorSpaceConverter` interface:

```python
from typing import Tuple

import numpy as np

from . import ChannelRange, ColorSpaceConverter


class RgbToNewColorSpaceConverter(ColorSpaceConverter):

    def convert(self, image: np.ndarray) -> np.ndarray:
        # image conversion implementation
        ...

    def convert_back(self, image: np.ndarray) -> np.ndarray:
        # back image conversion implementation
        ...

    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        # return the ranges of the target color space (your new color space)
        ...
```

Unit tests for color space converters are located in [tests/utils/cs_conversion](tests/utils/cs_conversion). Put all necessary tests assuring the correctness of `RgbToNewColorSpaceConverter`
into a separate test module:

```sh
>>> cd tests/utils/cs_conversion
>>> echo > test_cs_rgb_to_new_color_space.py
```

Add `NewColorSpace` to the [Click](https://palletsprojects.com/p/click/) command line interface in [main.py](main.py):

```python
...
from core import NewColorSpace
...
    @click.option('--color-space', '-s', 'color_space', default=RGB,
                  type=click.Choice([GRAY, HSV, LAB, RGB, NewColorSpace],
                                    case_sensitive=False),
                  help='color space')
...
```

Add `NewColorSpace` to [utils/cs_conversion/cs_converter_builder.py](utils/cs_conversion/cs_converter_builder.py):

```python
...
from core import NewColorSpace
...
    if target_color_space == NewColorSpace:
        return RgbToNewColorSpaceConverter()
...
```

For more implementation details see the existing color space converters:

* [IdentityConverter](utils/cs_conversion/cs_identity.py)
  * [Tests for IdentityConverter](tests/utils/cs_conversion/test_cs_identity.py)
* [RgbToHsvConverter](utils/cs_conversion/cs_rgb_to_hsv.py)
  * [Tests for RgbToHsvConverter](tests/utils/cs_conversion/test_cs_rgb_to_hsv.py)
* [RgbToLabConverter](utils/cs_conversion/cs_rgb_to_lab.py)
  * [Tests for RgbToLabConverter](tests/utils/cs_conversion/test_cs_rgb_to_lab.py)
* [StubConverter](tests/utils/cs_conversion/stub_converter.py) - only for testing
  * [Tests for StubConverter](tests/utils/cs_conversion/test_stub_converter.py)

## Citation

Please cite [Keep it Simple: Image Statistics Matching for Domain Adaptation](https://arxiv.org/abs/2005.12551) if you use `image-statistics-matching`:

```bib
@inproceedings{AbramovBayerHeller2020,
    author    = {Alexey Abramov and Christopher Bayer and Claudio Heller},
    title     = {Keep it Simple: Image Statistics Matching for Domain Adaptation},
    booktitle = {Scalability in Autonomous Driving, CVPR workshop},
    year      = {2020},
}
```

## License

`image-statistics-matching` is [MIT](LICENSE) licensed.

## Disclaimer

Links to websites of third parties are provided only for your convenience.
These websites are completely independent and outside the control of
Continental AG and in no-way related to Continental AG and/or its subsidiaries
(Together called as ‘Continental AG’). Continental AG is not liable for the
content of any of these third-party websites that are accessed from the
Continental websites, and assumes no responsibility and no liability for the
content, data protection provisions or use of such websites.

## Authors

Alexey Abramov [@aabramovrepo](https://github.com/aabramovrepo)

Christopher Bayer [@BayerC](https://github.com/BayerC)

Claudio Heller [@claudio-h](https://github.com/claudio-h)

## Maintainers

Alexey Abramov [@aabramovrepo](https://github.com/aabramovrepo)

Claudio Heller [@claudio-h](https://github.com/claudio-h)
