Inference
=========

The inference module
--------------------

The :mod:`.inference` module contains classes that infer the probability
distribution of model parameters from observation data,
also known as `inverse analysis <https://en.wikipedia.org/wiki/Inverse_problem>`_ or `data assimilation <https://en.wikipedia.org/wiki/Data_assimilation>`_, in many engineering disciplines.
The output of the :mod:`.inference` module is the probability distribution of model states :math:`\vec{x}_t`, 
sometimes augmented by the parameters :math:`\vec{\Theta}`, conditioned on observation data :math:`\vec{y}_t`.

Sequential Monte Carlo
----------------------

The class currently available for statistical inference is :class:`.SMC`.
It recursively updates the probability distribution of the augmented model state 
:math:`\hat{\vec{x}}_T=(\vec{x}_T, \vec{\Theta})` from the sequences of observation data
:math:`\hat{\vec{y}}_{0:T}` from time :math:`t = 0` to time :math:`T`.

Via Bayes' rule, the posterior distribution of the augmented model state reads

.. math::

   p(\hat{\vec{x}}_{0:T}|\hat{\vec{y}}_{1:T}) \propto \prod_{t_i=1}^T p(\hat{\vec{y}}_{t_i}|\hat{\vec{x}}_{t_i}) p(\hat{\vec{x}}_{t_i}|\hat{\vec{x}}_{{t_i}-1}) p(\hat{\vec{x}}_0),

Where :math:`p(\hat{\vec{x}}_0)` is the so-called prior distribution.
We can rewrite this equation in the recursive form, so that the posterior distribution gets updated
at every time step :math:`t`, according to

.. math::

   p(\hat{\vec{x}}_{0:t}|\hat{\vec{y}}_{1:t}) \propto p(\hat{\vec{y}}_t|\hat{\vec{x}}_t)p(\hat{\vec{x}}_t|\hat{\vec{x}}_{t-1})p(\hat{\vec{x}}_{1:t-1}|\hat{\vec{y}}_{1:t-1}),

Where :math:`p(\hat{\vec{y}}_t|\hat{\vec{x}}_t)` and :math:`p(\hat{\vec{x}}_t|\hat{\vec{x}}_{t-1})`
are the `likelihood <https://en.wikipedia.org/wiki/Likelihood_function>`_ distribution
and the (deterministic) transition distribution.

Importance sampling
```````````````````

These distributions can be efficiently evaluated via `importance sampling <https://en.wikipedia.org/wiki/Importance_sampling>`_.
The idea is to have samples that are more important than others when approximating a target distribution.
The measure of this importance is the so-called **importance weight**.

.. image:: ../figs/sis.png
  :width: 400
  :alt: Sequential Importance Sampling
  Illustration of importance sampling.

Therefore, we draw samples :attr:`.Model.param_data`, :math:`\vec{\Theta}^{(i)} \ (i=1,...,N_p)`,
from a proposal density, leading to an ensemble of model states :attr:`.Model.sim_data` :math:`\vec{x}_t^{(i)}`,
and recursively update the importance weights :attr:`.SMC.posteriors` :math:`w_t^{(i)}`, via

.. math::

   w_t^{(i)} \propto p(\hat{\vec{y}}_t|\hat{\vec{x}}_t^{(i)})p(\hat{\vec{x}}_t^{(i)}|\hat{\vec{x}}_{t-1}^{(i)}) w_{t-1}^{(i)}.

The likelihood function :attr:`.SMC.likelihoods` :math:`p(\hat{\vec{x}}_t|\hat{\vec{x}}_{t-1})`
can be simply a multivariate Gaussian, which is used by the function :attr:`.SMC.get_likelihoods`.

.. math::

   p(\hat{\vec{y}}_t|\hat{\vec{x}}_t^{(i)}) \propto \exp \{-\frac{1}{2}[\hat{\vec{y}}_t-\mathbf{H}(\vec{x}^{(i)}_t)]^T {\mathbf{\Sigma}_t^D}^{-1} [\hat{\vec{y}}_t-\mathbf{H}(\vec{x}^{(i)}_t)]\},

where :math:`\mathbf{H}` is the observation model that reduces to a diagonal matrix for uncorrelated observables,
and :math:`\mathbf{\Sigma}_t^D` is the covariance matrix :attr:`.SMC.cov_matrices`
calculated from :math:`\hat{\vec{y}}_t` and the user-defined normalized variance :attr:`.Model.sigma_max`, in :attr:`.SMC.get_covariance_matrices`.

By making use of importance sampling, the posterior distribution
:math:`p(\hat{\vec{y}}_t|\hat{\vec{x}}_t^{(i)})` gets updated over time in :attr:`.SMC.data_assimilation_loop`
--- this is known as `Bayesian updating <https://statswithr.github.io/book/the-basics-of-bayesian-statistics.html#bayes-updating>`_.

.. image:: ../figs/bayesian_updating.png
  :width: 400
  :alt: Sequential Importance Sampling
  Time evolution of the importance weights over a model parameter.

Ensemble predictions
````````````````````

Since the importance weight on each sample :math:`\vec{\Theta}^{(i)}` is discrete
and the sample :math:`\vec{\Theta}^{(i)}` and model state :math:`\vec{x}_t^{(i)}` are in a one-to-one relationship,
the ensemble mean :attr:`.SMC.ips` and variance :attr:`.SMC.covs` can be computed as 

.. math::

   \mathrm{\widehat{E}}[f_t(\hat{\vec{x}}_t)|\hat{\vec{y}}_{1:t}] & = \sum_{i=1}^{N_p} w_t^{(i)} f_t(\hat{\vec{x}}_t^{(i)}),
   
   \mathrm{\widehat{Var}}[f_t(\hat{\vec{x}}_t)|\hat{\vec{y}}_{1:t}] & = \sum_{i=1}^{N_p} w_t^{(i)} (f_t(\hat{\vec{x}}_t^{(i)})-\mathrm{\widehat{E}}[f_t(\hat{\vec{x}}_t)|\hat{\vec{y}}_{1:t}])^2.

where :math:`f_t` describes an arbitrary quantity of interest as a function of the model's state and parameters :math:`\hat{\vec{x}}_t^{(i)}`.
