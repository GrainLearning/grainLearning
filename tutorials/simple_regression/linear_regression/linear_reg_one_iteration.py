"""
This tutorial shows how to run one iteration of Bayesian calibration for a linear regression model.
"""
import os
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem, DynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
sim_data_dir = os.path.abspath(os.path.join(__file__, "../../../../tests/data/linear_sim_data"))
curr_iter = 0

calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 0,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "obs_data_file": PATH + '/linear_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "param_data_file": f'{sim_data_dir}/iter{curr_iter}/smcTable0.txt',
            "sim_data_dir": sim_data_dir,
            "sim_data_file_ext": ".npy",
            "param_names": ['a', 'b'],
        },
        "inference": {
            "Bayes_filter": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "covariance_type": "full",
            },
        },
        "save_fig": 0,
    }
)

# run GrainLearning for one iteration and generate the resampled parameter values
calibration.load_and_run_one_iteration()

# store the original parameter values and simulation data
param_data = calibration.system.param_data
sim_data = calibration.system.sim_data
ctrl_data = calibration.system.ctrl_data
obs_data = calibration.system.obs_data

# recreate a calibration tool
calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 0,
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
            "callback": None,
        },
        "inference": {
            "Bayes_filter": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
            },
        },
        "save_fig": 1,
    }
)

# run GrainLearning for one iteration and generate the resampled parameter values
calibration.load_and_run_one_iteration()
