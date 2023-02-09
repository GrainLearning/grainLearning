Tutorials
=========

In this section, we demonstrate how to use GrainLearning through a simple example of linear regression.
With :class:`.DynamicSystem` and :class:`IODynamicSystem` and the utility functions defined in 
:class:`.BayesianCalibration`, we show different ways of connecting
GrainLearning to Python or third-party software models,

Linear regression with a Python model
-------------------------------------

For demonstrative purposes, we use a linear function :math:`y = a\times{x}+b` as the numerical model,
implemented as the callback function of the :class:`.DynamicSystem`.

First, we create a synthetic dataset from this linear equation.

.. code-block:: python

    x_obs = np.arange(100)
    y_obs = 0.2* x_obs + 5.0

    # add Gaussian noise (optional)
    y_obs += np.random.rand(100) * 2.5

    def run_sim(model, **kwargs):
        data = []
        for params in model.param_data:
            y_sim = params[0] * model.ctrl_data + params[1]
            data.append(np.array(y_sim, ndmin=2))
        
        model.sim_data = np.array(data)

A calibration tool can be initialized by defining all the necessary input in a dictionary
and passing it to the constructor of :class:`.BayesianCalibration`.

.. code-block:: python

    import numpy as np

    from grainlearning import CalibrationToolbox

    calibration = CalibrationToolbox.from_dict(
        {
            "num_iter": 10,
            "model": {
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
                "callback": run_sim,
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "seed": 0,
                }
            }            
            "save_fig": 0,
        }
    )
    
    calibration.run()

Note that we want the SMC algorithm to end with an effective sample size
of no less than 30%. This is achieved by assigning `ess_target = 0.3` when initializing the inference method.
`calibration.run()` will run the Bayesian calibration iteratively
until the termination criterion is met.
By default, no plots are created unless the flag :attr:`.BayesianCalibration.save_fig` is non-negative.

Click :download:`here <../../tutorials/linear_regression/python_linear_regression_solve.py>` to download the full script.

Linear regression with a "software" model
-----------------------------------------

Because most likely the external software reads in and writes out text files,
its interaction with GrainLearning has to be done with the :class:`.IODynamicSystem`
Now let us look at the same example, with the :class:`.IODynamicSystem` and a linear function implemented in a separate file `LinearModel.py`.
For simplicity, we implement this external "software" in Python, which takes the command line arguments as the model parameters.

.. code-block:: python

    #!/usr/bin/env python3
    import sys
    import numpy as np

    def write_dict_to_file(data, file_name):
        """
        write a python dictionary data into a text file 
        """
        with open (file_name,'w') as f: 
            keys = data.keys()
            f.write('# ' + ' '.join(keys) + '\n')
            num = len(data[list(keys)[0]])
            for i in range(num):
                f.write(' '.join([str(data[key][i]) for key in keys]) + '\n')

    # define model parameters
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    description = sys.argv[3]

    x_obs = np.arange(100)
    y_sim = a * x_obs + b

    # write sim data and parameter in text files 
    data_file_name = 'linear_'+ description + '_sim.txt'
    sim_data = {'f': y_sim}
    write_dict_to_file(sim_data, data_file_name)

    data_param_name = 'linear_'+ description + '_param.txt'
    param_data = {'a': [a], 'b': [b]}
    write_dict_to_file(param_data, data_param_name)

This Python script is called by the callback `run_sim` from the command line.

.. code-block:: python

    executable = 'python ./tutorials/linear_regression/LinearModel.py'

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
            description = 'Iter' + str(curr_iter) + '-Sample' + str(i).zfill(magn)
            print(" ".join([executable, '%.8e %.8e' % tuple(params), description]))
            os.system(' '.join([executable, '%.8e %.8e' % tuple(params), description]))

When initializing :class:`.IODynamicSystem`,
one has to make sure that `sim_data_dir` and `obs_data_file` exist, `sim_name`, `obs_names` and `ctrl_name` are given,
and `sim_data_file_ext` is correct such that GrainLearning can find the data in the simulation directories.

.. code-block:: python

    from grainlearning import CalibrationToolbox
    from grainlearning.dynamic_systems import IODynamicSystem

    calibration = CalibrationToolbox.from_dict(
        {
            "num_iter": 10,
            "model": {
                "model_type": IODynamicSystem,
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data_file": 'linearObs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "sim_data_dir": './tutorials/linear_regression/',
                "sim_data_file_ext": '.txt',
                "callback": run_sim,
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "seed": 0,
                }
            },
            "save_fig": 0,
        }
    )
    
    calibration.run()

When running `calibration.run()`, subdirectories with the name `iter<curr_iter>` will be created in :attr:`.IODynamicSystem.sim_data_dir`.
In these subdirectories, you find

- simulation data file: `<sim_name>_Iter<curr_iter>-Sample<sample_ID>_sim.txt`
- parameter data file: `<sim_name>_Iter<curr_iter>-Sample<sample_ID>_param.txt`,

where <sim_name> is :attr:`.IODynamicSystem.sim_name`, <curr_iter> is :attr:`.BayesianCalibration.curr_iter`,
and <sample_ID> is the index of the :attr:`.IODynamicSystem.param_data` sequence.

Click :download:`here <../../tutorials/linear_regression/linear_regression_solve.py>` to download the full script.

GrainingLearning as a postprocessing tool
-----------------------------------------

The previous two examples assume no prior knowledge of the probability distribution of the parameters.
However, if you have some prior knowledge and have drawn samples from it,
you can simply use GrainLearning as a postprocessing tool to

1. quantify the posterior distribution from existing simulation data

2. draw new samples for the next batch of simulations 

The initialization of the calibration tool is the same as before.
However, you can load the simulation data and run Bayesian calibration for one iteration, with 

.. code-block:: python

    calibration.load_and_run_one_iteration()

and store the new parameter table in a text file.

.. code-block:: python

    resampled_param_data = calibration.resample()
    calibration.system.write_to_table(calibration.curr_iter + 1)

The parameter table below can be used to run the software model (e.g., YADE).

.. code-block:: text

	!OMP_NUM_THREADS description key a b 
	 8 Iter1-Sample00         0     5.0000000000e-01     3.3333333333e+00 
	 8 Iter1-Sample01         1     2.5000000000e-01     6.6666666667e+00 
	 8 Iter1-Sample02         2     7.5000000000e-01     1.1111111111e+00 
	 8 Iter1-Sample03         3     1.2500000000e-01     4.4444444444e+00 
	 8 Iter1-Sample04         4     6.2500000000e-01     7.7777777778e+00 
	 8 Iter1-Sample05         5     3.7500000000e-01     2.2222222222e+00 

Click :download:`here <../../tutorials/linear_regression/linear_reg_one_iteration.py>` to download the full script.
