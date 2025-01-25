Tutorials
=========

In this section, we demonstrate the functionalities GrainLearning via a simple example of linear regression.

.. _sec-use-GL:

Three ways of using GrainLearning
---------------------------------

We show in the following three different ways of using GrainLearning,

1. with :ref:`a Python model <sec-python-model>`,
2. with :ref:`a "software" model via the command line <sec-external-model>`,
3. and as :ref:`a postprocessing tool <sec-standalone>` to generate new parameter samples.

.. _sec-python-model:

Linear regression with a Python model
`````````````````````````````````````

To work with GrainLearning in the Python environment,
we need to write a callback function of the :class:`.BayesianCalibration` class where the model :math:`y = a\times{x}+b` is called.

.. code-block:: python

    # define a callback function to be passed to the system object
    def run_sim(system):
        data = []
        # loop over the parameter samples
        for params in system.param_data:
            # run the model y = a*x + b
            y_sim = params[0] * system.ctrl_data + params[1]
            # append the data to the list
            data.append(np.array(y_sim, ndmin=2))
        # pass the data to the system object
        system.set_sim_data(data)
 
Let us use :math:`a=0.2` and :math:`b=5.0` to generate some synthetic data and add Gaussian noise to it.

.. code-block:: python

    # define the true parameters
    a = 0.2
    b = 5.0
    # define the control data in the range [0, 100)
    x_obs = np.arange(100)
    # generate the synthetic data
    y_obs = a * x_obs + b

    # add Gaussian noise (optional)
    y_obs += np.random.rand(100) * 2.5

A calibration tool can be initialized by defining all the necessary input in a dictionary
and passing it to the constructor of :class:`.BayesianCalibration`.
Check out the documentation of :class:`.BayesianCalibration` for more details.

.. code-block:: python

    import numpy as np
    from grainlearning import BayesianCalibration

    # create a calibration tool
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 10,
            "callback": run_sim,
            "system": {
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
            },
          "inference": {
              "Bayes_filter": {"ess_target": 0.3},
              "sampling": {
                  "max_num_components": 1,
                }
            }            
            "save_fig": 0,
        }
    )

    # run the calibration tool    
    calibration.run()
    print(f'The most probable parameter values are {calibration.get_most_prob_params()}')

Note that it is important to keep the effective sample size large enough (e.g., `ess_target = 0.3`)
such that the parameter distribution is not sampled only at the optima.
Another key parameter for the sampling algorithm is `max_num_components`,
which controls the upper bound for the number of components in the parameter distribution.

Click :download:`here <../../tutorials/simple_regression/linear_regression/python_linear_regression_solve.py>` to download the full script.

.. attention::
  Play with `ess_target` and `max_num_components` to see how they affect the identified most probable parameter values
  and the number of iterations needed to reach the termination criterion.

.. _sec-external-model:

Linear regression with a "software" model
`````````````````````````````````````````

Linking GrainLearning with external software is done with the :class:`.IODynamicSystem`
Now let us look at the same example, using the :class:`.IODynamicSystem` and a linear function implemented in a separate file
:download:`linear_model.py <../../tutorials/simple_regression/linear_regression/linear_model.py>`.
This Python "software" will be run from command line by the callback function of :class:`.BayesianCalibration`
and take the command-line arguments as model parameters.
Download :download:`this script <../../tutorials/simple_regression/linear_regression/linear_model.py>`
and :download:`the observation data <../../tutorials/simple_regression/linear_regression/linear_obs.dat>`,
open a Python console in the same directory, and copy and paste the following code to run the calibration.

First import the necessary modules.

.. code-block:: python

    import os
    from math import floor, log
    from grainlearning import BayesianCalibration
    from grainlearning.dynamic_systems import IODynamicSystem

Then define the callback function to run the external software.

.. code-block:: python

    executable = f'python ./linear_model.py'
        
    def run_sim(system, **kwargs):
        """
        Run the external executable and passes the parameter sample to generate the output file.
        """
        # keep the naming convention consistent between iterations
        mag = floor(log(system.num_samples, 10)) + 1
        curr_iter = kwargs['curr_iter']
        # check the software name and version
        print("*** Running external software... ***\n")
        # loop over and pass parameter samples to the executable
        for i, params in enumerate(system.param_data):
            description = 'Iter' + str(curr_iter) + '_Sample' + str(i).zfill(mag)
            print(" ".join([executable, "%.8e %.8e" % tuple(params), system.sim_name, description]))
            os.system(' '.join([executable, "%.8e %.8e" % tuple(params), system.sim_name, description]))

Now let us define the calibration tool. Note that the system type is changed :class:`.IODynamicSystem`.

.. important::
  Additionally, one has to make sure that `obs_data_file` exist and
  `sim_name`, `obs_names`, `sim_data_dir`, and `ctrl_name` are given
  such that GrainLearning can find the data in the simulation directories.
  Otherwise, an error will be raised.

.. code-block:: python

    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 10,
            "callback": run_sim,            
            "system": {
                "system_type": IODynamicSystem,
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data_file": './linear_obs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "sim_data_dir": './sim_data/',
                "sim_data_file_ext": '.txt',
            },
            "inference": {
                "Bayes_filter": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "random_state": 0,
                }
            },
            "save_fig": 0,
        }
    )
    
    calibration.run()
    print(f'The most probable parameter values are {calibration.get_most_prob_params()}')

For each iteration of `calibration.run()`,
subdirectories with the name `iter<curr_iter>` will be created in :attr:`the simulation data directory <.IODynamicSystem.sim_data_dir>`.
In these subdirectories, you find

- simulation data file: `<sim_name>_Iter<curr_iter>_Sample<sample_ID>_sim.<ext>>`
- parameter data file: `<sim_name>_Iter<curr_iter>_Sample<sample_ID>_param.<ext>>`,

where <sim_name> is :attr:`.IODynamicSystem.sim_name`, <curr_iter> is :attr:`.BayesianCalibration.curr_iter`,
<sample_ID> is the index of the :attr:`.IODynamicSystem.param_data` sequence, and <ext> is :attr:`.IODynamicSystem.sim_data_file_ext`.

Click :download:`here <../../tutorials/simple_regression/linear_regression/linear_regression_solve.py>` to download the full script.

.. _sec-standalone:

GrainLearning as a postprocessing tool
``````````````````````````````````````

Want to work outside the GrainLearning calibration loop?
You can simply use GrainLearning as a postprocessing tool to

1. quantify the posterior distribution from existing simulation data,
2. and draw new samples for the next batch of simulations 

Continuing from :ref:`the previous tutorial  <sec-external-model>`,
there should be subdirectories in `./sim_data` where the simulation data are stored.

Postprocess simulation data using IODynamicSystem
:::::::::::::::::::::::::::::::::::::::::::::::::

The following code snippet shows how to load the simulation data and run Bayesian calibration for one iteration. 
Open a Python console in the same directory where you executed the previous tutorial and copy and paste the following code.

.. note::
    Provide the correct `curr_iter`, `sim_data_dir`, and `param_data_file` to load the simulation data.
    The file extension of the simulation data sim_data_file_ext` must be given to find the data files.

.. code-block:: python

    import os
    from grainlearning import BayesianCalibration
    from grainlearning.dynamic_systems import IODynamicSystem

    # user input
    curr_iter = 0
    sim_data_dir = './sim_data/'
    param_data_file = sim_data_dir + f'/iter{curr_iter}/linear_Iter' + str(curr_iter) + '_Samples.txt'
    sim_data_file_ext = '.txt'

    # create a calibration tool
    calibration = BayesianCalibration.from_dict(
        {
            "curr_iter": curr_iter,
            "num_iter": 0,
            "system": {
                "system_type": IODynamicSystem,
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "obs_data_file": './linear_obs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "param_data_file": param_data_file,
                "sim_data_dir": sim_data_dir,
                "sim_data_file_ext": sim_data_file_ext,
                "param_names": ['a', 'b'],
            },
            "inference": {
                "Bayes_filter": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                },
            },
            "save_fig": 0,            
        }
    )
    
    # run GrainLearning for one iteration and generate the resampled parameter values
    calibration.load_and_run_one_iteration()

This will create and store new parameter samples in a text file named 'linear_Iter<curr_iter+1>_Samples.txt'.
The user may want to continue running the software model using the parameter values stored in this file.

.. attention::
  Change `curr_iter` to 1, 2, ..., 4 and run the above code snippet again to see how different iterations can be loaded.

Postprocess simulation data using DynamicSystem
:::::::::::::::::::::::::::::::::::::::::::::::

It is also possible to use :class:`.DynamicSystem` instead. However, it is crucial to make sure that
the elements in the simulation data array have one-to-one correspondence with the elements in parameter data array.
Otherwise, the probability distribution will be incorrect and therefore the resampled parameter values will be wrong.

Continuing from the previous tutorial, we create a few variables that stores the parameter, observation, and simulation data,
and then create a new `calibration` object using :class:`.DynamicSystem`.

.. code-block:: python

    from grainlearning.dynamic_systems import DynamicSystem

    param_data = calibration.system.param_data
    sim_data = calibration.system.sim_data
    ctrl_data = calibration.system.ctrl_data
    obs_data = calibration.system.obs_data
    
    # recreate a calibration tool
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 0,
            "callback": None,
            "system": {
                "system_type": DynamicSystem,
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "param_data": param_data,
                "num_samples": param_data.shape[0],
                "ctrl_data": ctrl_data,
                "obs_data": obs_data,
                "obs_names": ['f'],
                "sim_name": 'linear',
                "sim_data": sim_data,
            },
            "inference": {
                "Bayes_filter": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                },
            },
            "save_fig": 0,
        }
    )
    
    # run GrainLearning for one iteration and generate the resampled parameter values
    calibration.load_and_run_one_iteration()

Click :download:`here <../../tutorials/simple_regression/linear_regression/linear_reg_one_iteration.py>`
to download the full script for the tutorials in this section.

.. _sec-seed:

Stochastic or deterministic sampling
------------------------------------

The sampling algorithm in GrainLearning is stochastic.
However, the random number generator can be seeded to generate reproducible results.
This can be done by setting :attr:`.GaussianMixtureModel.random_state` to a constant integer. 

.. code-block:: python

    # create a calibration tool
    "inference": {
        "Bayes_filter": {"ess_target": 0.3},
        "sampling": {
            "max_num_components": 1,
            "random_state": 0,
        },
    },

.. note::
  Insert the above code snippet to the declaration of the calibration tool in the previous tutorials;
  observe how the resampled parameter values become deterministic. To visualized the old and new parameter samples,
  set :attr:`.BayesianCalibration.save_fig` to a non-negative integer.

.. _sec-plot:

Visualize the sampling of parameter distribution
------------------------------------------------

Visualizing the sampling of parameter distribution can be done by setting :attr:`.BayesianCalibration.save_fig` to a non-negative integer.

- `save_fig=0` will only show the figures interactively but will not save them
- `save_fig>0` will save the figures in the directory './<sim_name>' but not show them interactively

By default, no plots are shown unless the flag :attr:`.BayesianCalibration.save_fig` is set.
