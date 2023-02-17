Dynamic systems
===============

The dynamic system module
-------------------------

The :mod:`.dynamic_systems` module is essential for GrainLearning to execute computational models
and encapsulate simulation and observation (reference) data in a single :class:`.DynamicSystem` class.
Currently, the :mod:`.dynamic_systems` module contains

- a :class:`.DynamicSystem` class that executes simulations handles the simulation and observation data in a *Python environment*,
- an :class:`.IODynamicSystem` class that sends instructions to *third-party software* (from the command line) and retrieves simulation data from the output files of the software.

Note that the dynamic system classes defined in GrainLearning are also known as state-space models
that consist of the predicted states :math:`\vec{x}_t` (:attr:`.DynamicSystem.sim_data`) and experimental observables :math:`\vec{y}_t` (:attr:`.DynamicSystem.obs_data`).

.. math::

	\begin{align}
	\vec{x}_t & =\mathbb{F}(\vec{x}_{t-1})+\vec{\nu}_t
	\label{eq:dynaModel},\\
	\vec{y}_t & =\mathbb{H}(\vec{x}_t)+\vec{\omega}_t
	\label{eq:obsModel}
	\end{align}

where :math:`\mathbb{F}` represents the *third-party software* model that
takes the previous model state :math:`\vec{x}_{t-1}` to make predictions for time :math:`t`. 
If all observables :math:`\vec{y}_t` are independent and have a one-to-one relationship with :math:`\vec{y}_t`,
the observation model :math:`\mathbb{H}` reduces to the identity matrix :math:`\mathbf{I}_d`, 
with :math:`d` being the number of independent observables.

The simulation and observation errors :math:`\vec{\nu}_t` and :math:`\vec{\omega}_t`
are random variables and assumed to be normally distributed with zero means.
We consider both errors together in the covariance matrix with :attr:`.SMC.cov_matrices`.
In fact, :math:`\vec{x}_t` and :math:`\vec{y}_t` are also random variables
whose distributions are updated by the :mod:`.inference` module.

Interact with third-party software
----------------------------------

Interaction with external "software" models can be done via the callback of the :class:`.DynamicSystem` class.
You can define your own :attr:`.DynamicSystem.callback`
and pass samples (combinations of parameters) to the **model implemented in Python** or to the software from the **command line**.
The following gives an example of the callback where the "software" model :math:`\mathbb{F}` is a Python function. 

.. code-block:: python
   :caption: A callback function implemented in Python

   def run_sim(system: Type["DynamicSystem"], **kwargs):
       data = []
       for params in system.param_data:
           # a linear function y = a*x + b
           y_sim = params[0] * system.ctrl_data + params[1]
           data.append(np.array(y_sim, ndmin=2))
       # assign model output to the dynamic system class
       model.set_sim_data(data) 


The IODynamicSystem class
`````````````````````````

The :class:`.IODynamicSystem` inherits from :class:`.DynamicSystem` and is intended to work with third-party software packages.
The :attr:`.IODynamicSystem.run` function overrides the :attr:`.DynamicSystem.run` function of the :class:`.DynamicSystem`.
to write samples in a text file and run the :attr:`.IODynamicSystem.callback` to execute the third-party software model at all sample points.
Below is an example of the callback where parameter samples are passed as command-line arguments to an external executable.

.. code-block:: python
   :caption: A callback function that interacts with external software

   executable = './software'

   def run_sim(system, **kwargs):
       from math import floor, log
       import os
       # keep the naming convention consistent between iterations
       mag = floor(log(system.num_samples, 10)) + 1
       curr_iter = kwargs['curr_iter']
       # loop over and pass parameter samples to the executable
       for i, params in enumerate(system.param_data):
           description = 'Iter'+str(curr_iter)+'-Sample'+str(i).zfill(mag)
           print(" ".join([executable, '%.8e %.8e'%tuple(params), description]))
           os.system(' '.join([executable, '%.8e %.8e'%tuple(params), description]))


Data format and directory structure
```````````````````````````````````

GrainLearning can read .npy (for backward compatibility) and plain text formats.
When using :class:`.IODynamicSystem`, the directory :attr:`.IODynamicSystem.sim_data_dir` must exist and contains the observation data file :attr:`.IODynamicSystem.obs_data_file`.
Subdirectories with name `iter<curr_iter>` will be created in :attr:`.IODynamicSystem.sim_data_dir`.
In these subdirectories, you find

- simulation data file: `<sim_name>_Iter<curr_iter>-Sample<sample_ID>_sim.txt`
- parameter data file: `<sim_name>_Iter<curr_iter>-Sample<sample_ID>_param.txt`,

where <sim_name> is :attr:`.IODynamicSystem.sim_name`, <curr_iter> is :attr:`.BayesianCalibration.curr_iter`,
and <sample_ID> is the index of the :attr:`.IODynamicSystem.param_data` sequence.

For example, the observation data stored in a text file :attr:`.IODynamicSystem.obs_data_file` should look like this.

.. code-block:: text

	# u f
	0		5.0
	1		5.2
	2		5.4
	3		5.6
	4		5.8
	5		6.0

Similarly, in a simulation data file `linear_Iter0-Sample00_sim.txt`, you find

.. code-block:: text

	# f
	0.741666667
	1.023635417
	1.3056041669999998
	1.587572917
	1.869541667
	2.151510417

Note the simulation data doesn't contain the :attr:`DynamicSystem.ctrl_data` sequence.

Therefore, when using :class:`.IODynamicSystem` the user needs to provide the keys to the data sequence
of the **control** and **observation** group.
These keys are also used to extract the corresponding data from the simulation data files.

.. code-block:: python

    # name of the control variable
    "ctrl_name": 'u',
    # name of the output variables of the model
    "obs_names": ['f'],
