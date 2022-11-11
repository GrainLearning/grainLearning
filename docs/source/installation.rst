Installation
============

GrainLearning is set up using poetry, with a Python version higher than 3.7.
We have tested GrainLearning with Python verions 3.7, 3.8, and 3.9.

Package management and dependencies
-----------------------------------

Install using poetry
````````````````````

First, install poetry following `these instructions <https://python-poetry.org/docs/#installation>`_.
 
.. code-block:: bash
  
   # Clone the repository
   $ git clone https://github.com/GrainLearning/grainLearning.git 

   # Activate a poetry environment
   $ cd grainLearning
   $ poetry shell

   # Install the dependencies of GrainLearning 
   $ poetry install

   # Run the self-tests
   $ poetry run pytest -v  

   # You are done. Try any examples in the ./tutorials directory
   $ poetry run python <example.py>

Install using pip
`````````````````

.. code-block:: bash
  
   # clone the repository
   $ git clone https://github.com/GrainLearning/grainLearning.git 
   $ cd grainLearning

   # We recommend to work on an environment conda or any other python environment manager,
   # for example, with anaconda
   $ conda create --name grainlearning python=3.8
   $ conda activate grainlearning`

   # Install GrainLearning 
   $ pip install .

   # You may need to install matplotlib and pytest
   $ conda install matplotlib # for visualization
   $ conda install pytest # optional

   # Run the self-tests
   $ pytest -v  

   # You are done. Try any examples in the ./tutorials directory
   $ python <example.py>


For Windows users
`````````````````
1. Enable Windows Subsystem for Linux (WSL1 or WSL2) following `this link <https://learn.microsoft.com/en-us/windows/wsl/install-manual>`_ 

2. Install GrainLearning following the instructions above
 
3. Use anaconda if there are no WSL available on your operating system

4. Open Anaconda Prompt and install GrainLearning with pip

5. Now you should have a virtual environment, possibly called GrainLearning. Choose that environment from your anaconda navigator: click Environments and select grainlearning from the drop-down menu

6. Open any editor, for example, spider and run the examples in grainLearning/tutorials.


Packaging/One command install
`````````````````````````````

TODO

`pip install grainlearning`

Documentation
-------------

Online
``````

You can check the online documentation `here <https://grainlearning.readthedocs.io/en/latest/>`_.

Build the documentation
```````````````````````

.. code-block:: bash
  
   # You need to be in the same `poetry shell` used for installing grainlearning
   $ poetry shell
   $ cd docs
   $ poetry run make html
