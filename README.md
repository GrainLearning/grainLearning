# Welcome to GrainLearning!
| fair-software.eu recommendations  | Badges |
|:---  | :--| 
| code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/GrainLearning/grainlearning) |
| license                      |  [![github license badge](https://img.shields.io/github/license/GrainLearning/grainlearning)](https://github.com/GrainLearning/grainlearning)|
| community registry           |  [![RSD](https://img.shields.io/badge/rsd-grainlearning-00a3e3.svg)](https://research-software-directory.org/projects/granular-materials) [![workflow pypi badge](https://img.shields.io/pypi/v/grainlearning.svg?colorB=blue)](https://pypi.python.org/project/grainlearning/)|
| citation                     | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7123966.svg)](https://doi.org/10.5281/zenodo.7123966)|
| Best practices checklist     | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/6533/badge)](https://bestpractices.coreinfrastructure.org/projects/6533)|
| howfairis                    | [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)|
| Documentation                | [![Documentation Status](https://readthedocs.org/projects/grainlearning/badge/?version=latest)](https://grainlearning.readthedocs.io/en/latest/?badge=latest)|
| Code Quality                | [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=GrainLearning_grainLearning&metric=coverage)](https://sonarcloud.io/summary/new_code?id=GrainLearning_grainLearning) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=GrainLearning_grainLearning&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=GrainLearning_grainLearning) |


Bayesian uncertainty quantification for discrete and continuum numerical models of granular materials,
developed by various projects of the University of Twente (NL), the Netherlands eScience Center (NL), University of
Newcastle (AU), and Hiroshima University (JP).
Browse to the [GrainLearning documentation](https://grainlearning.readthedocs.io/en/latest/) to get started.

## Features

- Infer and update model parameters using "time" series (sequence) data
  via [Sequential Monte Carlo filtering](https://en.wikipedia.org/wiki/Particle_Filter)
- Uniform, quasi-random sampling using [low-discrepancy sequences](https://en.wikipedia.org/wiki/Halton_sequence)
- Iterative sampling by training a
  nonparametric [Gaussian mixture model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
- [Surrogate modeling](https://grainlearning.readthedocs.io/en/latest/rnn.html) capability for "time" series data

[//]: # (using [recurrent neural networks]&#40;https://en.wikipedia.org/wiki/Recurrent_neural_network&#41;)

[//]: # (- Hybrid physics-based and data-driven model evaluation strategy)

## Installation

### Install using poetry

1. Install poetry following [these instructions](https://python-poetry.org/docs/#installation).
1. Clone the repository: `git clone https://github.com/GrainLearning/grainLearning.git`
1. Go to the source code directory: `cd grainLearning`
1. Activate the virtual environment: `poetry shell`
1. Install GrainLearning and its dependencies: `poetry install`

### Install using pip

1. Clone the repository: `git clone https://github.com/GrainLearning/grainLearning.git`
1. Go to the source code directory: `cd grainLearning`
1. Activate the virtual environment: `conda create --name grainlearning python=3.8 && conda activate grainlearning`
1. Install GrainLearning and its dependencies: `pip install .`

### For Windows users

- Installation using Windows Subsystem for Linx (WSL)
  - Enable WSL1 or WSL2 according to the
    instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install-manual)
  - Install GrainLearning using [poetry](#install-using-poetry) or [pip](#install-using-pip)
- Installation using anaconda (if no WSLs are available on your Windows system)
  - Open Anaconda Prompt and install GrainLearning using [pip](#install-using-pip). This should create a virtual
    environment, named GrainLearning.
  - Choose that environment from your anaconda navigator: click `Environments` and select `grainlearning` from the
    drop-down menu

### One command installation

Stable versions of GrainLearning can be installed via `pip install grainlearning`
However, you still need to clone the GrainLearning repository to run the tutorials.

## Tutorials

1. Linear regression with
   the [`run_sim`](https://github.com/GrainLearning/grainLearning/blob/main/tutorials/linear_regression/python_linear_regression_solve.py#L14)
   callback function of the [`DynamicSystem`](https://github.com/GrainLearning/grainLearning/blob/main/grainlearning/dynamic_systems.py)
   class,
   in [python_linear_regression_solve.py](https://github.com/GrainLearning/grainLearning/blob/main/tutorials/linear_regression/python_linear_regression_solve.py)

2. Nonlinear, multivariate regression

3. Interact with the numerical model of your choice
   via [`run_sim`](https://github.com/GrainLearning/grainLearning/blob/main/tutorials/linear_regression/linear_regression_solve.py#L11)
   ,
   in [linear_regression_solve.py](https://github.com/GrainLearning/grainLearning/blob/main/tutorials/linear_regression/linear_regression_solve.py)

4. Load existing simulation data and run GrainLearning for one iteration,
   in [oedo_load_and_resample.py](https://github.com/GrainLearning/grainLearning/blob/main/tutorials/oedo_compression/oedo_load_and_resample.py)
5. RNN module tutorials:
    - [Train your RNN](tutorials/rnn/train_rnn.ipynb)
    - [Predict using an RNN](tutorials/rnn/predict.ipynb)
    - [Use an RNN in the calibration workflow](tutorials/rnn/rnn_calibration_GL.ipynb)

## Citing GrainLearning

Please choose from the following:

- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7123966.svg)](https://doi.org/10.5281/zenodo.7123966) A DOI for
  citing the software
- H. Cheng, T. Shuku, K. Thoeni, P. Tempone, S. Luding, V. Magnanimo. **An iterative Bayesian filtering framework for
  fast and automated calibration of DEM models**. _Comput. Methods Appl. Mech. Eng.,_ 350 (2019), pp.
  268-294, [10.1016/j.cma.2019.01.027](https://doi.org/10.1016/j.cma.2019.01.027)

## Software using GrainLearning

- YADE: http://yade-dem.org/
- MercuryDPM: https://www.mercurydpm.org/

## Community

The original development of `GrainLearning` is done by [Hongyang Cheng](https://hongyangcheng.weebly.com), in collaboration
with [Klaus Thoeni](https://www.newcastle.edu.au/profile/klaus-thoeni)
, [Philipp Hartmann](https://www.newcastle.edu.au/profile/philipp-hartmann),
and [Takayuki Shuku](https://sites.google.com/view/takayukishukuswebsite/home).
The software is currently maintained by [Hongyang Cheng](https://hongyangcheng.weebly.com) and [Stefan Luding](https://www2.msm.ctw.utwente.nl/sluding/) with the help
of [Luisa Orozco](https://www.esciencecenter.nl/team/dr-luisa-orozco/)
and [Retief Lubbe](https://tusail.eu/projects/esr-12.html).
The GrainLearning project receives contributions from students and collaborators. For an exhaustive list,
see [CONTRIBUTORS.md]().

## Help and Support

For assistance with the GrainLearning software, please raise an issue on the GitHub Issues page.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and
the [NLeSC/python-template](https://github.com/NLeSC/python-template).
