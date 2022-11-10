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

5. You may need to install matplotlib to do some plotting: `conda install matplotlib`

6. To run the self-tests of GrainLearning, do `conda install pytest` and run `pytest -v`

For Windows users
`````````````````
1. Enable WSL1 or WSL2 following `this link <https://learn.microsoft.com/en-us/windows/wsl/install-manual>`_ 

2. Install GrainLearning following the instructions above
 
3. Use anaconda if there are no Windows Subsystem for Linux available on your operating system

4. Open Anaconda Prompt and install GrainLearning with pip

5. Now you should have a virtual environment, possibly called GrainLearning. Choose that environment from your anaconda navigator: click Environments and select grainlearning from the drop-down menu

6. Open any editor, for example, spider and run the examples in grainLearning/tutorials.


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
