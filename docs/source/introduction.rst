Introduction
============

Welcome to GrainLearning!

GrainLearning is a Bayesian uncertainty quantification and propagation (UQ+P) toolbox
for computer simulations of granular materials.
It is designed to make uncertainty quantification, sensitivity analysis, and calibration simple,
for the computational models of granular materials, such as soils, rocks, pharmaceutical powders, and food grains.

GrainLearning was initially developed for the inference of particle- or microstructure-scale parameters
in the **discrete element method (DEM)** simulations of granular materials, formulated as an inverse or data assimilation problem.
The meaning of "GrainLearning" is twofold: (1) the properties of physical particles are learned from limited observations,
and (2) statistical "particles" are used to achieve the inference via `sequential Monte Carlo filtering <https://en.wikipedia.org/wiki/Particle_filter>`_.
GrainLearning can be also used with other empirical, numerical, or data-driven models 
that are strongly nonlinear in time.

The most important modules of GrainLearning are :doc:`dynamic_systems <dynamic_systems>`, :doc:`inference <inference>`, and :doc:`sampling <sampling>`.
Different classes in these modules can be mixed by the user to provide a variety of Bayesian calibration tools.
Currently, only the :doc:`iterative Bayesian filter <iterative_bayesian_filter>` is implemented and has been tested
with the DEM and continuum constitutive models.
GrainLearning can be used together with external DEM software packages, e.g., `YADE <http://yade-dem.org/>`_, and
`MercuryDPM <https://www.mercurydpm.org/>`_, and other numerical methods beyond DEM.
Integrating an external (software) model into GrainLearning can be easily done
via the callback function :attr:`.IODynamicSystem.callback` of the :class:`.IODynamicSystem` class.

It is important to bear in mind the following definitions, as they will appear frequently in this documentation.

- `Prior <https://en.wikipedia.org/wiki/Prior_probability>`_: the initial knowledge about model state and/or parameter before any data is taken into account, expressed as a probability distribution.
- `Posterior <https://en.wikipedia.org/wiki/Posterior_probability>`_: the updated knowledge after the data is taken into account via Bayes' theorem.
- Proposal: the distribution from which we draw new samples. It can be assumed or learned.
- Samples: the combination of model parameters drawn from a distribution, leading to certain "randomly" distributed model states.
- Model evaluation: the execution of the model at the sample points.
- Ensemble: the probability distribution of the model state, represented as a collection of state vectors and their associated weights.

We draw inspiration from the `UQpy <https://uqpyproject.readthedocs.io/en/latest/index.html>`_ and `MUSCLE3 <https://muscle3.readthedocs.io/en/latest/index.html>`_ libraries when designing the GrainLearning software and its documentation.
