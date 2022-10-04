# Project Setup

GrainLearning is setup using poetry. 

## Python versions (to check)

This repository is set up with Python versions:

- 3.7
- 3.8
- 3.9


## Package management and dependencies

### Install using poetry

1. Install poetry following [these instructions](https://python-poetry.org/docs/#installation).
1. Clone the repository: https://github.com/GrainLearning/grainLearning.git
1. `cd grainLearning`
1. `poetry shell`
1. `poetry install`
1. You can either run your code via: `poetry run python example.py` or `python example.py`

### Install using pip

1. Clone the repository: ` git clone https://github.com/GrainLearning/grainLearning.git`
1. `cd grainLearning`
1. We recomend to work on an environment conda or any other python environment manager
  `conda create --name grainlearning python=3.8`
  `conda activate grainlearning`
1. `pip install .`
1. Additionally you'll need to install matplotlib to do some plotting: `pip install matplotlib`

## Packaging/One command install
TODO
```shell
pip install grainlearning
```

## Testing and code coverage

To run the tests:
``` shell
poetry run pytest -v
```

To create a file coverage.xml with the information of the code coverage:
``` shell
poetry run coverage xml
```

To create a more complete output of tests and coverage:
``` shell
poetry run pytest --cov --cov-report term --cov-report xml --junitxml=xunit-result.xml tests/ 
```

## Documentation

### Online:
You can check the documentation [here](https://grainlearning.readthedocs.io/en/latest/)

### Create the documentation using poetry
1. You need to be in the same `poetry shell` used to install grainlearning, or repeat the process to install using poetry.
1. `cd docs`
1. `poetry run make html`


## Coding style conventions and code quality

- [Relevant section in the NLeSC guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=coding-style-conventions) and [README.dev.md](README.dev.md).

## Continuous code quality

TODO: integration of sonarcloud

## Package version number

TODO: update

## CHANGELOG.md

- In this file we document changes to our software package

## CODE_OF_CONDUCT.md

- Information about how to behave professionally
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/documentation?id=code-of-conduct)

## CONTRIBUTING.md

- Information about how to contribute to this software package
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/documentation?id=contribution-guidelines)

## MANIFEST.in

- List non-Python files that should be included in a source distribution
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=building-and-packaging-code)

## NOTICE
TODO (also modify/check NOTICE)
- List of attributions of this project and license dependencies
