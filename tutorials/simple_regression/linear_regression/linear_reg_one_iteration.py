"""
This tutorial shows how to run one iteration of Bayesian calibration for a linear regression model.
"""
import os
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

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

# %%
# reproduce the result with a given sigma value
calibration.load_and_run_one_iteration()
resampled_param_data = calibration.resample()

# %%
# write new parameter table to the simulation directory
calibration.system.write_params_to_table(calibration.curr_iter + 1)
