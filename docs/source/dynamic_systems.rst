Dynamic systems
===============

The dynamic system module
-------------------------

The :mod:`.dynamic_systems` module is essential for GrainLearning to run the predictive model(s)
and encapsulate simulation and observation (or reference) data in a single :class:`.DynamicSystem` class.
Currently, the :mod:`.dynamic_systems` module contains

- a :class:`.DynamicSystem` class that handles the simulation and observation data within a *Python environment*,
- an :class:`.IODynamicSystem` class that sends instructions to external *third-party software* (e.g., via the command line) and retrieves simulation data from the output files of the software.

.. note:: A dynamic system is also known as a state-space model in the literature.
  It describes the time evolution of the state of the model :math:`\vec{x}_t` (:attr:`.DynamicSystem.sim_data`)
  and the state of the observables :math:`\vec{y}_t` (:attr:`.DynamicSystem.obs_data`).
  Both :math:`\vec{x}_t` and :math:`\vec{y}_t` are random variables
  whose distributions are updated by the :mod:`.inference` module.

.. math::

	\begin{align}
	\vec{x}_t & =\mathbb{F}(\vec{x}_{t-1})+\vec{\nu}_t
	\label{eq:dynaModel},\\
	\vec{y}_t & =\mathbb{H}(\vec{x}_t)+\vec{\omega}_t
	\label{eq:obsModel}
	\end{align}

where :math:`\mathbb{F}` represents the **third-party software** model that
takes the previous model state :math:`\vec{x}_{t-1}` to make predictions for time :math:`t`. 
If all observables :math:`\vec{y}_t` are independent and have a one-to-one correspondence with :math:`\vec{x}_t`,
(meaning you predict what you observe),
the observation model :math:`\mathbb{H}` reduces to the identity matrix :math:`\mathbb{I}_d`, 
with :math:`d` being the number of independent observables.

The simulation and observation errors :math:`\vec{\nu}_t` and :math:`\vec{\omega}_t`
are random variables and assumed to be normally distributed around zero means.
We consider both errors together in the covariance matrix :attr:`.SMC.cov_matrices`.

Interact with third-party software via callback function
--------------------------------------------------------

Interaction with an external "software" model can be done via the callback function of :class:`.DynamicSystem` or :class:`.IODynamicSystem`.
You can define your own callback function
and pass *samples* (combinations of parameters) to the **model implemented in Python** or to the software from the **command line**.
The figure below shows how the callback function is called in the execution loop of :class:`.BayesianCalibration`. 

.. _execution_loop:
.. image:: ./figs/execution_loop.png
  :width: 400
  :alt: How a callback function gets executed

Interact with Python software
`````````````````````````````

Let us first look at an example where the predictive model :math:`\mathbb{F}` is implemented in Python.
The following code snippet shows how to define a callback function that runs a linear model. 

.. code-block:: python
   :caption: A linear function implemented in Python

   def run_sim(system, **kwargs):
       data = []
       # loop over parameter samples
       for params in system.param_data:
           # Run the model: y = a*x + b
           y_sim = params[0] * system.ctrl_data + params[1]
           # Append the simulation data to the list
           data.append(np.array(y_sim, ndmin=2))
       # pass the simulation data to the dynamic system
       system.set_sim_data(data)


The function `run_sim` is assigned to the :attr:`.DynamicSystem.callback` attribute of the :class:`.DynamicSystem` class
and is is called every time the :attr:`.DynamicSystem.run` function is called (see :ref:`the figure <execution_loop>` above).


Interact with non-Python software
`````````````````````````````````

The :class:`.IODynamicSystem` class inherits from :class:`.DynamicSystem` and is intended to work with external software packages
via the command line.
The :attr:`.IODynamicSystem.run` function overrides the :attr:`.DynamicSystem.run` function of the :class:`.DynamicSystem` class.
Parameter samples are written into a text file and used by :attr:`.IODynamicSystem.callback` to execute the third-party software.
Users only need to write a for-loop to pass each parameter sample to this external software, e.g., as command-line arguments (see the example below).

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
           description = 'Iter'+str(curr_iter)+'_Sample'+str(i).zfill(mag)
           os.system(' '.join([executable, '%.8e %.8e'%tuple(params), description]))


.. note:: This code snippet can be used as a template to interact with any third-party software.
  The only thing you need to do is to replace the executable name and the command-line arguments.
  The command-line arguments are passed to the software in the order of the parameter names in :attr:`.IODynamicSystem.param_names`.
  The last argument (optional) is a description of the current simulation, which is used to tag the output files.
  In this example, the description is `Iter<curr_iter>_Sample<sample_ID>`.
  The output files are read into :attr:`.IODynamicSystem.sim_data` by the function :attr:`.IODynamicSystem.load_sim_data`.

Data format and directory structure
:::::::::::::::::::::::::::::::::::

GrainLearning can read plain text and .npy formats (for backward compatibility).
When using :class:`.IODynamicSystem`, the directory :attr:`.IODynamicSystem.sim_data_dir` must exist and contains the observation data file :attr:`.IODynamicSystem.obs_data_file`.
Subdirectories with name `iter<curr_iter>` will be created in :attr:`.IODynamicSystem.sim_data_dir`.
In these subdirectories, you find

- simulation data file: `<sim_name>_Iter<curr_iter>_Sample<sample_ID>_sim.txt`
- parameter data file: `<sim_name>_Iter<curr_iter>_Sample<sample_ID>_param.txt`,

where <sim_name> is :attr:`.IODynamicSystem.sim_name`, <curr_iter> is :attr:`.BayesianCalibration.curr_iter`,
and <sample_ID> is the index of the :attr:`.IODynamicSystem.param_data` sequence.

For example, the observation data stored in a text file :attr:`.IODynamicSystem.obs_data_file` should look like this.

.. code-block:: text

	# u f
	0	5.0
	1	5.2
	2	5.4
	3	5.6
	4	5.8
	5	6.0

Similarly, in a simulation data file `linear_Iter0_Sample00_sim.txt`, you will find

.. code-block:: text

	# f
	5.0
	5.2
	5.4
	5.6
	5.8
	6.0

.. note:: The simulation data doesn't contain the sequence of :attr:`DynamicSystem.ctrl_data` at which the outputs are stored.
  Therefore, when initializing :class:`.IODynamicSystem` the user needs to provide the keys to the data sequences
  that belong to the **control** and the **observation** group.

  .. code-block:: python
  
      # name of the control variable
      "ctrl_name": 'u',
      # name of the output variables of the model
      "obs_names": ['f'],
