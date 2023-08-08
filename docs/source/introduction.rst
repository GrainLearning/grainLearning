Introduction
============

Welcome to GrainLearning!

GrainLearning is a toolbox for Bayesian calibration and uncertainty quantification of 
computer simulations of granular materials.
It is designed to make uncertainty quantification, sensitivity analysis, and calibration simple,
for computational models of granular materials, such as soils, powders, and food grains.
GrainLearning is written in Python and is built on top of the `NumPy <https://numpy.org/>`_ and
`SciPy <https://www.scipy.org/>`_ libraries.

GrainLearning was initially developed for inferring particle- or microstructure-scale parameters
in **discrete element method (DEM)** simulations of granular materials.
It is designed to solve **inverse** or **data assimilation** problems, with two meanings behind its name:

1. physical particle properties are learned from limited observations,

2. statistical "particles" are used to achieve inference via `sequential Monte Carlo filtering <https://en.wikipedia.org/wiki/Particle_filter>`_.

GrainLearning is non-intrusive. Computer models used by GrainLearning are blackboxes and therefore can be theoretical, numerical, or data-driven.

The most important modules of GrainLearning are :doc:`dynamic systems <dynamic_systems>`, :ref:`inference <bayesian_filtering:The inference module>`,
and :ref:`sampling <bayesian_filtering:The sampling module>`.
An unique feature of GrainLearning is the machine learning (ML) capabilities of the :ref:`samplers <bayesian_filtering:The sampling module>`
to learn complex **probability distributions** and the :ref:`data-driven **emulators** <rnn:Train a RNN with your own data>` to learn the physics.
Users can combine different classes in these modules to create a variety of Bayesian calibration tools.
Currently, only the :ref:`iterative Bayesian filter <bayesian_filtering:Iterative Bayesian filter>` has been implemented, and tested
with `DEM models <https://www.sciencedirect.com/science/article/pii/S0045782519300520>`_
and `theoretical constitutive models <https://link.springer.com/chapter/10.1007/978-3-030-64514-4_90>`_.
The available data-driven emulator is `Long Short-Term Memory (LSTM) <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_ networks
widely used for predicting time series.
However, the modular design of GrainLearning allows for the implementation of other inference algorithms and models.
The following figure shows the main modules of GrainLearning and their interconnections.

.. figure:: ./figs/gl_modules.png
  :width: 400
  :alt: GrainLearning modules

Users only need to write a :attr:`callback function <.IODynamicSystem.callback>`
where parameter samples are sent to an external "software" or Python model.
Examples of this can be found for :ref:`YADE <examples:Run DEM simulations guided by GrainLearning>`
(`Yet Another Dynamic Engine <http://yade-dem.org/>`_).
For more information on how to integrate a predictive model into GrainLearning, see :doc:`dynamic systems <dynamic_systems>`.

.. note:: 
  While reading this documentation, keep in mind the following definitions, as they will appear frequently in various sections.

  - **Prior**: the initial knowledge about model state and/or parameter before any data is taken into account, expressed as a probability distribution.
  - **Posterior**: the updated knowledge after the evidence is taken into account via Bayes' theorem.
  - **Proposal**: the distribution from which we draw new samples. It can be assumed or learned.
  - **Samples**: the combination of model parameters drawn from a distribution, leading to certain "randomly" distributed model states.
  - **Model evaluation**: the execution of the model at the sample points.
  - **Ensemble**: the probability distribution of the model state, represented as a collection of state vectors and their associated weights.
  - **Filtering**: the process of updating the ensemble as new data is taken into account.

We draw inspiration from the `UQpy <https://uqpyproject.readthedocs.io/en/latest/index.html>`_ and `MUSCLE3 <https://muscle3.readthedocs.io/en/latest/index.html>`_ libraries when designing the GrainLearning software and its documentation.
