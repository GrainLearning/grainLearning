Introduction
============

Welcome to GrainLearning!

GrainLearning is a Bayesian uncertainty quantification and propagation (UQ+P) toolbox
for computer simulations of granular materials.
It is designed to make uncertainty quantification, sensitivity analysis, and calibration simple,
for computational models of granular materials, such as soils, rocks, pharmaceutical powders, and food grains.

GrainLearning was initially developed for the inference of the particle- or microstructure-scale parameters
of **discrete element method (DEM)** models of granular media, formulated as a inverse or data asimilation problem.
We name the software ''GrainLearning'' because not only the properties of physical particles
(e,g., powders and grains) are learned from limited observations for accurate simulations,
but also statistical ''paricles'' are used in the context of sequential Monte Carlo filtering.
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

Prior
Posterior
Proposal
Samples
Model evaluation

