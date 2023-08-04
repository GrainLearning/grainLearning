.. GrainLearning documentation master file, created by
   sphinx-quickstart on Mon Aug 29 15:19:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GrainLearning's documentation!
=========================================

GrainLearning is a Bayesian uncertainty quantification toolbox for computer simulations of granular materials.
The software is primarily used to infer model parameter distributions from observation or reference data,
a process also known as inverse analyses or data assimilation. 
Implemented in Python, GrainLearning can be loaded into a Python environment to process your simulation and observation data,
or alternatively, used as an independent tool where simulations are run separately, e.g., from the command line.

If you use GrainLearning, please cite `the version of the GrainLearning software you used <https://zenodo.org/record/7123966>`_.
If you want to know more about how the method works, the following papers can be interesting:

- H. Cheng, T. Shuku, K. Thoeni, P. Tempone, S. Luding, V. Magnanimo. An iterative Bayesian filtering framework for fast and automated calibration of DEM models. *Comput. Methods Appl. Mech. Eng., 350 (2019)*, pp. 268-294, `10.1016/j.cma.2019.01.027 <https://doi.org/10.1016/j.cma.2019.01.027>`_
- P. Hartmann, H. Cheng, K. Thoeni. Performance study of iterative Bayesian filtering to develop an efficient calibration framework for DEM. *Computers and Geotechnics 141*, 104491,  `10.1016/j.compgeo.2021.104491 <https://doi.org/10.1016/j.compgeo.2021.104491>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   dynamic_systems
   bayesian_filtering
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
