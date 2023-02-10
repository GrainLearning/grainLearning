.. GrainLearning documentation master file, created by
   sphinx-quickstart on Mon Aug 29 15:19:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GrainLearning's documentation!
=========================================

GrainLearning is a Bayesian uncertainty quantification and propagation toolbox for computer simulations
of granular materials.
The software is primarily used to infer and quantify parameter uncertainties in computational models of granular materials
from observation data, which is also known as inverse analyses or data assimilation. 
Implemented in Python, GrainLearning can be loaded into a Python environment to process the simulation and observation data,
or alternatively, as an independent tool where simulation runs are done separately, e.g., via a shell script.

If you use GrainLearning, please cite `the version of the GrainLearning software you used <https://zenodo.org/record/7123966>`_ and the following paper:

H. Cheng, T. Shuku, K. Thoeni, P. Tempone, S. Luding, V. Magnanimo.
An iterative Bayesian filtering framework for fast and automated calibration of DEM models. *Comput. Methods Appl. Mech. Eng., 350 (2019)*, pp. 268-294,
`10.1016/j.cma.2019.01.027 <https://doi.org/10.1016/j.cma.2019.01.027>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   dynamic_systems
   inference
   sampling
   iterative_bayesian_filter
   rnn
   tutorials
   examples
   how_to_contribute
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
