"""
This tutorial shows how to run one iteration of Bayesian calibration for a linear regression model.
"""
import os
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem, DynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))

calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 10,
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
        "inference": {
            "Bayes_filter": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 2,
                "n_init": 1,
                "random_state": 0,
                "covariance_type": "full",
            }
        },
        "save_fig": 0,
    }
)

# run GrainLearning for one iteration and generate the resampled parameter values
calibration.load_all()
