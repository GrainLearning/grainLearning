## Installation of grainlearning
Install grainlearning with the posibility to build documentation and check tests.

### Using poetry
1. Install poetry following [these instructions](https://python-poetry.org/docs/#installation).
1. Clone the repository: `git clone git@github.com:GrainLearning/grainLearning.git`
1. Go to the source code directory: `cd grainLearning`
1. Activate the virtual environment: `poetry shell`
1. Install GrainLearning and its dependencies: `poetry install -E docs -E dev` or `poetry install --extras "docs dev"`
1. Run all self-tests of GrainLearning with pytest: `poetry run pytest -v`

### Using pip
1. Clone the repository: `git clone git@github.com:GrainLearning/grainLearning.git`
1. Go to the source code directory: `cd grainLearning`
1. We advise to use a python environment using pyenv or conda.
1. Install GrainLearning and its dependencies: `pip install .[docs,dev]`, if you are using an zsh shell you should run `pip install .'[docs,dev]'` instead, this is because zsh uses square brackets for globbing / pattern matching.

## Making a release
This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
2. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
3. Update the version in different places (Non -exhaustive check-list) :
- CITATION.cff
- docs/source/conf.py
- pyproject.toml
- README.md
4. Run the unit and integration tests with `poetry run pytest -v`

### (2/3) PyPI

#### a. Set up your credentials 

```shell
poetry config http-basic.pypi <username> <password>
```

Those are your credentials to your account on pypi.org, which you do have to create if you don't have one.

#### b. Deploying to test-pypi

Before publishing unpolished package on PyPI, you can test it on a test version of PyPI.
This test PyPI will allow us to mimic updating to PyPI, then pip installing our own package to see if it works as expected.
For that you need another account on test.pypi.org (same username and password is fine).

You can then add the repository to poetry:

```shell
poetry config repositories.testpypi https://test.pypi.org/legacy/
```

Create a token following [this link](https://test.pypi.org/manage/account/token/)

```shell
poetry config pypi-token.test-pypi <your-token>
```

In a new terminal, without an activated virtual environment or an env directory:

```shell
# prepare a new directory
cd $(mktemp -d grainlearning.XXXXXX)

# fresh git clone ensures the release has the state of origin/main branch
git clone https://github.com/GrainLearning/grainlearning .

# prepare a clean virtual environment and activate it or skip it if you have poetry installed in your system.
# (venv version)
python3 -m venv env
source env/bin/activate
pip install poetry

# clean up any previously generated artefacts (if there are)
rm -rf grainlearning.egg-info
rm -rf dist

# Become a poet
poetry shell
poetry install
poetry build 

# This generates folder dist that has the wheel that is going to be distributed on test-pypi.
poetry publish --build -r test-pypi

```
This will by default register the package to test-pypi.
Visit [https://test.pypi.org/project/grainlearning](https://test.pypi.org/project/grainlearning)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

#### c. Testing the deployed package to test-pypi
In a new terminal, without an activated virtual environment or an env directory:

```shell
cd $(mktemp -d grainlearning-test.XXXXXX)

# prepare a clean virtual environment and activate it
python3 -m venv env
source env/bin/activate

# install from test pypi instance:
# TODO: why `pip install -i https://test.pypi.org/simple/ grainlearning` does not work
python3 -m pip -v install --no-cache-dir \
--index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple grainlearning
```

Check that the package works as it should when installed from test-pypi. For example run:
``` shell
python3 tests/integration/test_lengreg.py 
```

#### d. Uploading to pypi

If you are happy with the package on test-pypi, deploy it on pypi.
Once a version of GrainLearning is uploaded. It cannot be removed.

```shell
# Go back to the first terminal in step b,
# FINAL STEP: upload to PyPI
poetry publish --build -r test-pypi
```

Visit [https://pypi.org/project/grainlearning/](https://pypi.org/project/grainlearning/)
and verify that your package was deployed successfully.

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/GrainLearning/grainlearning/releases/new). Check that this release also triggers Zenodo into making a snapshot of your repository and sticking a DOI on it.

## Documentation

### Online:
You can check the documentation [here](https://grainlearning.readthedocs.io/en/latest/)

### Create the documentation locally using poetry
1. You need to be in the same `poetry shell` used to install grainlearning, or repeat the process to install using poetry and doc extras: `poetry install -E docs` or `poetry install --extras "docs"`. Alternatively you can install via pip: `pip install .[docs]`
1. `cd docs`
1. `poetry run make html`

## Testing and code coverage
You must have had installed grainlearning development dependencies: `poetry install -E dev` or `poetry install --extras "dev"` or `pip install .[dev]`

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

## Linter
Run several Python analysis tools to ensure that your contributions are following standards and best practices.

1. You must have had installed grainlearning with dev dependencies either  
a). `poetry install -E dev` or b). `pip install .[dev]`.
1. While being in the main directory grainlearning, at the same level as .prospector.yaml, run prospector. Depending on how you have installed grainlearning you can either run
a). `poetry run prospector` or b). `prospector`
