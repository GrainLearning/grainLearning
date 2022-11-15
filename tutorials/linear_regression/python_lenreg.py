
#%%
 
import numpy as np

from grainlearning import CalibrationToolbox

import matplotlib.pyplot as plt



x_obs = np.arange(100)

y_obs = 0.2* x_obs + 5.0


def run_sim(model, **kwargs):
    data = []
    for params in model.param_data:
        y_sim = params[0] * model.ctrl_data + params[1]
        data.append(np.array(y_sim, ndmin=2))
    
    model.sim_data = np.array(data)

calibration = CalibrationToolbox.from_dict(
    {
        "num_iter": 10,
        "model": {
            "param_mins": [0.001, 0.001],
            "param_maxs": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "sim_name":'linear',
            "sigma_tol": 0.01,
            "callback": run_sim,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "n_init": 1,
                "seed": 0,
            },
            "initial_sampling": "halton",
        },
        "save_fig": 0,
    }
)

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

error_tolerance = 0.1
 
error = most_prob_params - [0.2,5.0]
assert abs(error[0])/0.2 < error_tolerance, f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
assert abs(error[1])/5.0 < error_tolerance, f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"
