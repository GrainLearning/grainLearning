Introduction
============

Welcome to GrainLearning!

GrainLearning is a Bayesian uncertainty quantification and propagation (UQ+P) toolbox
for computer simulations of granular materials.
It is designed to make uncertainty quantification, sensitivity analysis, and calibration simple,
for computational models of granular materials, such as soils, rocks, pharmaceutical powders, and food grains.

GrainLearning was initially developed for the inference of the particle- or microstructure-scale parameters
of **discrete element method (DEM)** models of granular media, formulated as a inverse or data asimilation problem.
We name the software "GrainLearning" because not only the properties of physical particles
(e,g., powders and grains) are learned from limited observations for accurate simulations,
but also statistical "paricles" are used in the context of sequential Monte Carlo filtering.
Nevertheless, GrainLearning can be used with other empirical, numerical, or data-driven models
that are stronly nonlinear in time.

The most important modules of GrainLearning are: **models**,  **inference**, amd **sampling**.
Different classes in these modules can be mixed to provide the user a variety of Bayesian UQ+P tools.
Currently, only the iterative Bayesian filter is provided and has been tested
with discrete particle and continuum constitutive models.
GrainLearning can be used together `YADE <http://yade-dem.org/>`_,
`MercuryDPM <https://www.mercurydpm.org/>`_, and other numerical methods beyond DEM.
Integration another model or software into GrainLearning can be easily done
in the **models** module, via a callback function.

It is important to bear in mind the following definitions as they will appear frequently in this documentation.

- `Prior <https://en.wikipedia.org/wiki/Prior_probability>`_: the initial knowledge about model state or parameter before any data is taken into account, expressed as a probability distribution.
- `Posterior <https://en.wikipedia.org/wiki/Posterior_probability>`_: the updated knowledge after the data is taken into account via the Bayes' theorem.
- Proposal: the distribution where we drawn new samples. It can be assumed or learned.
- Samples: the combination of model parameters drawn from a distribution, leading certain "randomly" distributed model states.
- Model evaluation: the execution of the model at the sample points.
- Ensemble: the probability distribution of the model state, represented as a collection of state vectors and their associated weights.

We draw inspiration from the `UQpy <https://uqpyproject.readthedocs.io/en/latest/index.html>`_ and `MUSCLE3 <https://muscle3.readthedocs.io/en/latest/index.html>`_ libraries when designing the GrainLearning software and its documentation.
