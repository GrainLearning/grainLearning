Models
======

The models module
-----------------

The :mod:`.models` module is essential for GrainLearning to execute computational models
and encapsulate simulation and observation (reference) data in a single :class:`.Model` class
Currently, the :mod:`.models` module contains

- a :class:`.Model` class that execute simulations handles the simulation and observation data in a *Python environment*,
- a :class:`.IOModel` class that sends instructions to a *third-party software* (from the command line) and retrieve simulation data from the output files of the software.

Note that the models defined in GrainLearning are state space models
consist of both numerical predictions :math:`\vec{x}_t` (:attr:`.Model.sim_data`) and experimental observables :math:`\vec{y}_t` (:attr:`.Model.obs_data`).

.. math::

	\begin{align}
	\vec{x}_t & =\mathbb{F}(\vec{x}_{t-1})+\vec{\nu}_t
	\label{eq:dynaModel},\\
	\vec{y}_t & =\mathbb{H}(\vec{x}_t)+\vec{\omega}_t
	\label{eq:obsModel}
	\end{align}

where :math:`\mathbb{F}` represents the *third-party software* model that
takes the previous model state :math:`\vec{x}_{t-1}` to make predict for time :math:`t`. 
If all observables :math:`\vec{y}_t` are independent and have one-to-one relationship with :math:`\vec{y}_t`,
the observation model :math:`\mathbb{H}` reduces to the identity matrix :math:`\mathbf{I}_d`, 
with :math:`d` being the number of independent observables.

The simulation and observation errors :math:`\vec{\nu}_t` and :math:`\vec{\omega}_t`
are random variables and assumed to be normally distributed with zero means.
We consider both errors together in the covariance matrix with :attr:`.SequentialMonteCarlo.cov_matrices`.
In fact, :math:`\vec{x}_t` and :math:`\vec{y}_t` are also random variables
whose distributions are updated by the :mod:`.inference` module.

Interact with third-party software
----------------------------------

Interaction with external "software" models can be done via the callback of the :class:`.Model` class.
You can define your own :attr:`.Model.callback`
and pass samples (combinations of parameters) to the **model implemented in Python** or to the software from the **command line**.
The following gives an example of the callback where the "software" model :math:`\mathbb{F}` is a Python function. 

.. code-block:: python
   :caption: A callback function implemented in Python

   def run_sim(model: Type["Model"], **kwargs):
       data = []
       for params in model.param_data:
           # a linear function y = a*x + b
           y_sim = params[0] * model.ctrl_data + params[1]
           data.append(np.array(y_sim, ndmin=2))
       # assign model ouput to the model class
       model.sim_data = np.array(data)


The IOModel class
`````````````````

The :class:`.IOModel` inherits from :class:`.Model` and is intended to work with third-party software packages.
The :attr:`.IOModel.run` function overrides the :attr:`.Model.run` function of the :class:`.Model`.
to write samples in a text file and run the :attr:`.IOModel.callback` to execute the third-party software model at all sample points.
Below is an example of the callback where parameter samples are passed as command-line arguments to an external executable.

.. code-block:: python
   :caption: A callback function that interact with external software

   executable = './software'

   def run_sim(model, **kwargs):
       from math import floor, log
       import os
       # keep the naming convention consistent between iterations
       magn = floor(log(model.num_samples, 10)) + 1
       curr_iter = kwargs['curr_iter']
       # check the software name and version
       print("*** Running external software... ***\n")
       # loop over and pass parameter samples to the executable
       for i, params in enumerate(model.param_data):
           description = 'Iter'+str(curr_iter)+'-Sample'+str(i).zfill(magn)
           print(" ".join([executable, '%.8e %.8e'%tuple(params), description]))
           os.system(' '.join([executable, '%.8e %.8e'%tuple(params), description]))
