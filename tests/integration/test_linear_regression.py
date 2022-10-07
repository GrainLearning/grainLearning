
#%%
 
import numpy as np

from grainlearning import CalibrationToolbox

import matplotlib.pyplot as plt



from sklearn.metrics import mean_absolute_error as mae

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
    
    plt.figure()
    plt.plot(model.ctrl_data, model.obs_data, ls="", marker=".", label="Observation")
    for i,y_sims in enumerate(model.sim_data):
       params =  model.param_data[i]
       print(i,params)
       plt.plot(model.ctrl_data[0], y_sims[0], label="Simulation")
       plt.text(model.ctrl_data[0][-1], y_sims[0][-1], "{}".format(i,params[0],params[1]), bbox=dict(facecolor='red', alpha=0.5))
    plt.show()


def test_step_forward():
    calibration = CalibrationToolbox.from_dict(
        {
            "num_iter": 8,
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

    calibration.next()
    
    
    most_prob = np.argmax(calibration.calibration.posterior_ibf)


    # most_prob_params = calibration.model.param_data[most_prob] 

    least_err = np.argmin([mae(calibration.model.sim_data[sid,0,:],y_obs) for sid in range(calibration.model.num_samples)])
    
    assert most_prob == least_err, f"most probable does not have the least MAE {most_prob=} {least_err=}"



test_step_forward()
# #%%
# print(f'All parameter samples at the last iteration:\n {calibration.model.param_data_prev}')

# #%%
# plt.plot( np.arange(calibration.num_iter),calibration.sigma_list)
# #%%
# # calibration.sigma_list,len(calibration.sigma_list),calibration.num_iter
# # print(calibration.sigma_list)

# # %%
# most_prob = np.argmax(calibration.calibration.posterior_ibf)

# # %%
# most_prob_params = calibration.model.param_data_prev[most_prob] 

# print(f'Most probable parameter values: {most_prob_params}')
# # %%

# #tests
# error_tolerance = 0.01

# #1. Testing values of parameters
# error = most_prob_params - [0.2,5.0]
# assert abs(error[0])/0.2 < error_tolerance, f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
# assert abs(error[1])/5.0 < error_tolerance, f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"

# #2. Checking sigma
# assert calibration.sigma_list[-1] < error_tolerance, "Final sigma is bigger than tolerance."

# # %%

# %%
