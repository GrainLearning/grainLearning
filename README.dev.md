## Making a release
This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
2. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
3. Update the version: Bumping the version across all files is done with [bumpversion](https://github.com/c4urself/bump2version), e.g.

```shell
bumpversion major
bumpversion minor
bumpversion patch
```
4. Run the unit tests with `poetry run pytest -v`

### (2/3) PyPI

#### a. Deploying to test-pypi
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
pip install twine

# This generates folder dist that has the wheel that is going to be distributed. Install twine in your system or in an environment.
twine upload --repository-url https://test.pypi.org/legacy/ dist/* 

```

Alternatively, you can try instead of the last command (twine) use Poetry :
```shell
poetry publish --build
```
This will by default register the package to pypi. You'll need to put your pypi username and password, alternatively you can pass the as -u and -p, respectively. More info [here](https://python-poetry.org/docs/cli/#build).

Visit [https://test.pypi.org/project/grainlearning](https://test.pypi.org/project/grainlearning)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

#### b. Testing the deployed package to test-pypi
In a new terminal, without an activated virtual environment or an env directory:

```shell
cd $(mktemp -d grainlearning-test.XXXXXX)

# prepare a clean virtual environment and activate it
python3 -m venv env
source env/bin/activate

# install from test pypi instance:
python3 -m pip -v install --no-cache-dir \
--index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple grainlearning
```

Check that the package works as it should when installed from pypitest. For example run:
``` shell
python3 tests/integration/test_14sep_lengreg.py 
```

#### c. Uploading to pypi

```shell
# Back to the first terminal,
# FINAL STEP: upload to PyPI (requires credentials)
twine upload dist/*
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/GrainLearning/grainlearning/releases/new). Check that this release also triggers Zenodo into making a snapshot of your repository and sticking a DOI on it.

## Documentation

### Online:
You can check the documentation [here](https://grainlearning.readthedocs.io/en/latest/)

### Create the documentation using poetry
1. You need to be in the same `poetry shell` used to install grainlearning, or repeat the process to install using poetry.
1. `cd docs`
1. `poetry run make html`


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
