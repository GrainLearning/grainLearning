
#%%
 
import numpy as np

from grainlearning import CalibrationToolbox

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse

p1 = 0.2

p2 = 5.0
x_obs = np.arange(100)

y_obs = p1* x_obs + p2

y_obs_w_noise = y_obs + np.random.rand(100) * 2.5


def run_sim(model):
    data = []
    for params in model.param_data:
        y_sim = params[0] * model.ctrl_data + params[1]
        data.append(np.array(y_sim, ndmin=2))
    
    model.sim_data = np.array(data)
    
    # ~ plt.figure()
    # ~ plt.plot(model.ctrl_data, model.obs_data, ls="", marker=".", label="Observation")
    # ~ for i,y_sims in enumerate(model.sim_data):
       # ~ params =  model.param_data[i]
       # ~ print(i,params)
       # ~ plt.plot(model.ctrl_data[0], y_sims[0], label="Simulation")
       # ~ plt.text(model.ctrl_data[0][-1], y_sims[0][-1], "{}".format(i,params[0],params[1]), bbox=dict(facecolor='red', alpha=0.5))
    # ~ plt.show()


def test_step_forward():
    calibration = CalibrationToolbox.from_dict(
        {
            "num_iter": 0,
            "model": {
                "param_mins": [0, 0],
                "param_maxs": [1, 10],
                "num_samples": 13,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
                "callback": run_sim,
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {"max_num_components": 1},
            }
        }
    )

    calibration.run_one_iteration()
    
    most_prob = np.argmax(calibration.calibration.posterior_ibf)

    # most_prob_params = calibration.model.param_data[most_prob] 

    least_err = np.argmin([mse(calibration.model.sim_data[sid,0,:],y_obs) for sid in range(calibration.model.num_samples)])
    
    assert most_prob == least_err, f"most probable does not have the least MAE {most_prob=} {least_err=}"


test_step_forward()