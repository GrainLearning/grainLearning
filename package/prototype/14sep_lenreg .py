
#%%
 
import numpy as np

from grainlearning import CalibrationToolbox

import matplotlib.pyplot as plt


x_obs = np.arange(100)

y_obs = 0.2* x_obs + 5.0

y_obs += np.random.rand(100) * 10


def run_sim(model):
    data = []
    for params in model.param_data:
        y_sim = params[0] * model.ctrl_data + params[1]
        data.append(np.array(y_sim, ndmin=2))

    model.sim_data = np.array(data)
    print(model.sigma_max)

    plt.figure()
    plt.plot(model.ctrl_data, model.obs_data, ls="", marker=".", label="Observation")
    for y_sims in model.sim_data:
        plt.plot(model.ctrl_data[0], y_sims[0], label="Simulation")
    plt.show()


calibration = CalibrationToolbox.from_dict(
    {
        "num_iter": 8,
        "model": {
            "param_mins": [0, 0],
            "param_maxs": [1, 10],
            "num_samples": 20,
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "callback": run_sim,
        },
        "ibf": {
            "inference": {"ess_target": 0.3},
            "sampling": {"max_num_components": 1},
        },
    }
)

calibration.run()

#%%

print(calibration.model.param_data)

#%%
plt.plot( np.arange(calibration.num_iter),calibration.sigma_list)
#%%
# calibration.sigma_list,len(calibration.sigma_list),calibration.num_iter


# %%
most_prob = np.argmax(calibration.calibration.proposal_ibf)

# %%
calibration.model.param_data[most_prob]
# %%
