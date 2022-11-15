# %%

import numpy as np

from grainlearning import CalibrationToolbox
from grainlearning.models import IOModel

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


calibration = CalibrationToolbox.from_dict(
    {
        "num_iter": 10,
        "model": {
            "param_mins": [0.001, 0.001],
            "param_maxs": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_data_file": 'linearObs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": './tutorials/linear_regression/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.01,
            "callback": run_sim,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 2,
                "n_init": 1,
                "seed": 0,
                "cov_type": "full",
            }
        },
        "save_fig": 0,
        "model_type": IOModel
    }
)

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

error_tolerance = 0.1

error = most_prob_params - [0.2, 5.0]
assert abs(
    error[0]) / 0.2 < error_tolerance, f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
assert abs(
    error[1]) / 5.0 < error_tolerance, f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"
