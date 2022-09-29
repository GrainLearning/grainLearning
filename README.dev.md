## Making a release
This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
2. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
3. Make sure the [version has been updated](#versioning).
4. Run the unit tests with `pytest -v`

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

# This generates folder dist that has the wheel that is going to be distributed. Install twine in your system or in an environment.
twine upload --repository-url https://test.pypi.org/legacy/ dist/* 

```

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

Check that the package works as it should when installed from pypitest.

#### c. Uploading to pypi

```shell
# Back to the first terminal,
# FINAL STEP: upload to PyPI (requires credentials)
twine upload dist/*
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/GrainLearning/grainlearning/releases/new). Check that this release also triggers Zenodo into making a snapshot of your repository and sticking a DOI on it.
