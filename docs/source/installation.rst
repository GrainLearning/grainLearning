Installation
============

GrainLearning can be installed with poetry and pip, on Windows, Mac, and Linux OS, with a Python version higher than 3.8.
We have tested GrainLearning with Python versions 3.9, 3.10, and 3.12

Package management and dependencies
-----------------------------------

Install using poetry
````````````````````

First, install poetry following `these instructions <https://python-poetry.org/docs/#installation>`_.
 
.. code-block:: bash
  
   # Clone the repository
   git clone https://github.com/GrainLearning/grainLearning.git 

   # Activate the virtual environment
   cd grainLearning
   poetry shell

   # Install GrainLearning and its dependencies
   poetry install

   # Run the self-tests
   poetry run pytest -v  

   # You are done. Try any examples in the ./tutorials directory
   poetry run python <example.py>

Install using pip
`````````````````

.. code-block:: bash
  
   # clone the repository
   git clone https://github.com/GrainLearning/grainLearning.git 
   cd grainLearning

   # We recommend working in a virtual environment using conda or any other python environment manager.
   # for example, with anaconda
   conda create --name grainlearning python=3.8
   conda activate grainlearning

   # Install GrainLearning and its dependencies 
   pip install .

   # You may need to install matplotlib and pytest
   conda install matplotlib # for visualization
   conda install pytest # optional

   # Run the self-tests
   pytest -v  

   # You are done. Try any examples in the ./tutorials directory
   python <example.py>


For Windows users
`````````````````

- Installation using Windows Subsystem for Linx (WSL)

  - Enable WSL1 or WSL2 according to `these instructions <https://learn.microsoft.com/en-us/windows/wsl/install-manual>`_ 
  - Install GrainLearning using :ref:`poetry<installation:Install using poetry>` or :ref:`pip<installation:Install using pip>`

- Installation using anaconda (if no WSLs are available on your Windows system)

  - Open Anaconda Prompt and :ref:`install GrainLearning using pip<installation:Install using pip>`. This should create a virtual environment, named GrainLearning.
  - Choose that environment from your anaconda navigator: click `Environments` and select `grainlearning` from the drop-down menu
  - Open any editor, for example, spider, and run the examples in grainLearning/tutorials.

Packaging/One command install
`````````````````````````````

Stable versions of GrainLearning can be installed via `pip install grainlearning`.
However, you would still need to clone the GrainLearning repository to run the tutorials. 

.. code-block:: bash

   # create a virtual environment
   python3 -m venv env
   source env/bin/activate

   # install GrainLearning
   pip install grainlearning

   # Clone the repository
   git clone https://github.com/GrainLearning/grainLearning.git 

   # run a simple linear regression test
   python3 grainLearning/tests/integration/test_lenreg.py

   # deactivate virtual environment
   deactivate
   rm -r env

Documentation
-------------

Online
``````

You can check the online documentation `here <https://grainlearning.readthedocs.io/en/latest/>`_.

Build the documentation locally
```````````````````````

.. code-block:: bash
  
   # You need to be in the same `poetry shell` used for installing grainlearning
   $ poetry shell
   $ cd docs
   $ poetry run make html
