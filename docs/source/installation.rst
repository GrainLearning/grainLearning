Installation
============

GrainLearning is setup using poetry. 

Python versions
---------------

This repository is set up with Python versions:

- 3.7
- 3.8
- 3.9


Package management and dependencies
-----------------------------------

Install using poetry
````````````````````

1. Install poetry following `these instructions <https://python-poetry.org/docs/#installation>`_.

2. `git clone https://github.com/GrainLearning/grainLearning.git`

3. `cd grainLearning`

4. `poetry shell`

5. `poetry install`

6. You can either run your code via: `poetry run python <example.py>` or `python <example.py>`

Install using pip
`````````````````

1. `git clone https://github.com/GrainLearning/grainLearning.git`

2. `cd grainLearning`

3. We recommend to work on an environment conda or any other python environment manager

  `conda create --name grainlearning python=3.8`

  `conda activate grainlearning`

4. `pip install .`

5. You may need to install matplotlib to do some plotting: `pip install matplotlib`

Packaging/One command install
-----------------------------

TODO

`pip install grainlearning`

Documentation
-------------

Online
``````

You can check the documentation `here <https://grainlearning.readthedocs.io/en/latest/>`_

Create the documentation using poetry
`````````````````````````````````````

1. You need to be in the same `poetry shell` used to install grainlearning, or repeat the process to install using poetry.

1. `cd docs`

1. `poetry run make html`
