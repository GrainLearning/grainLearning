"""
This tutorial shows how to run one iteration of Bayesian calibration for a linear regression model.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem, DynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
executable = f'python {PATH}/linear_model.py'


def run_sim(calib):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    system = calib.system
    # keep the naming convention consistent between iterations
    mag = floor(log(system.num_samples, 10)) + 1
    # check the software name and version
    print("*** Running external software... ***\n")
    # loop over and pass parameter samples to the executable
    for i, params in enumerate(system.param_data):
        description = 'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)
        linear(executable, params, system.sim_name, description)


def linear(executable, params, sim_name, description):
    print(" ".join([executable, "%.8e %.8e" % tuple(params), sim_name, description]))
    os.system(' '.join([executable, "%.8e %.8e" % tuple(params), sim_name, description]))


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 1,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_data_file": PATH + '/linear_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.01,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "covariance_type": "full",
            },
        },
        "save_fig": -1,
    }
)

# Run one iteration of Bayesian calibration
calibration.run()

# Continue Bayesian calibration from a previous iteration until the total number of iterations or tolerance is reached
curr_iter = 0
calibration_continued = BayesianCalibration.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 3,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "obs_data_file": PATH + '/linear_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": ".txt",
            "param_names": ['a', 'b'],
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "covariance_type": "full",
            },
        },
        "save_fig": 0,
    }
)

calibration_continued.load_and_run_one_iteration()
calibration_continued.run()