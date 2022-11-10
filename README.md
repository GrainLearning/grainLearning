
# Welcome to GrainLearning!

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/GrainLearning/grainlearning) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/GrainLearning/grainlearning)](https://github.com/GrainLearning/grainlearning) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-grainlearning-00a3e3.svg)](https://research-software-directory.org/projects/granular-materials) [![workflow pypi badge](https://img.shields.io/pypi/v/grainlearning.svg?colorB=blue)](https://pypi.python.org/project/grainlearning/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7123966.svg)](https://doi.org/10.5281/zenodo.7123966) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/6533/badge)](https://bestpractices.coreinfrastructure.org/projects/6533) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/grainlearning/badge/?version=latest)](https://grainlearning.readthedocs.io/en/latest/?badge=latest) |

Bayesian uncertainty quantification for discrete and continuum numerical models of granular materials,
developed by various projects of the University of Twente (NL), the Netherlands eScience Center (NL), University of Newcastle (AU), and Hiroshima University (JP).
Browse to the [GrainLearning documentation](https://grainlearning.readthedocs.io/en/latest/) to get started. 

## Features
- Infer and update model parameters using "time" series (sequence) data via [Sequential Monte Carlo filtering](https://en.wikipedia.org/wiki/Particle_Filter)
- Uniform, quasi-random sampling using [low-discrepancy sequences](https://en.wikipedia.org/wiki/Halton_sequence) 
- Iterative sampling by training a nonparametric [Gaussian mixture model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
- Surrogate modeling capability for "time" series data

[//]: # (- Hybrid physics-based and data-driven model evaluation strategy)

## Installation

### Install using poetry (recommended)

1. Install poetry following [these instructions](https://python-poetry.org/docs/#installation).
1. Clone the repository: `git clone https://github.com/GrainLearning/grainLearning.git`
1. Go to the source code directory: `cd grainLearning`
1. Activate the virtual environment: `poetry shell`
1. Install GrainLearning and its dependencies: `poetry install`
1. Run all self-tests of GrainLearning with pytest: `poetry run pytest -v`

For windows users, click [here](https://grainlearning.readthedocs.io/en/latest/installation.html#for-windows-users) to check other installation options.

## Tutorials

1. Linear regression with the [`run_sim`](https://github.com/GrainLearning/grainLearning/blob/main/tutorials/linear_regression/linear_reg_solve.py#L13) callback function of the [`Model`](https://github.com/GrainLearning/grainLearning/blob/main/grainlearning/models.py) class
2. Nonlinear, multivariate regression
3. Interact with the numerical model of your choice
4. Load existing simulation data and run GrainLearning for one iteration 

[//]: # (5. Can you extend tutorial 1 to interactions between two particles?)

## Citing GrainLearning

Please choose from the following:
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7123966.svg)](https://doi.org/10.5281/zenodo.7123966) A DOI for citing the software 
- H. Cheng, T. Shuku, K. Thoeni, P. Tempone, S. Luding, V. Magnanimo. **An iterative Bayesian filtering framework for fast and automated calibration of DEM models**. _Comput. Methods Appl. Mech. Eng.,_ 350 (2019), pp. 268-294, [10.1016/j.cma.2019.01.027](https://doi.org/10.1016/j.cma.2019.01.027)

## Software using GrainLearning

- YADE: http://yade-dem.org/
- MercuryDPM: https://www.mercurydpm.org/

## Community

The original development of `GrainLearning` is done by [Hongyang Cheng](hongyangcheng.weebly.com), in collaboration with [Klaus Thoeni](https://www.newcastle.edu.au/profile/klaus-thoeni), [Philipp Hartmann](https://www.newcastle.edu.au/profile/philipp-hartmann), and [Takayuki Shuku](https://sites.google.com/view/takayukishukuswebsite/home).
The software is currently maintained with the help of [Luisa Orozco](https://www.esciencecenter.nl/team/dr-luisa-orozco/), [Retief Lubbe](https://tusail.eu/projects/esr-12.html), and [Aron Jansen](https://www.esciencecenter.nl/team/dr-aron-jansen/).
The GrainLearning project receives contributions from students and collaborators. For an exhaustive list, see [CONTRIBUTORS.md]().

## Help and Support

For assistance with the GrainLearning software, please raise an issue on the GitHub Issues page.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
