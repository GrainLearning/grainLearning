import numpy as np
from grainlearning import CalibrationToolbox

x_obs = np.arange(100)
y_obs = 0.2 * x_obs + 5.0
# y_obs += np.random.rand(100) * 2.5

def run_sim(model, **kwargs):
    data = []
    for params in model.param_data:
        y_sim = params[0] * model.ctrl_data + params[1]
        data.append(np.array(y_sim, ndmin=2))

    model.sim_data = np.array(data)

def test_lenreg():
    calibration = CalibrationToolbox.from_dict(
        {
            "num_iter": 10,
            "model": {
                "param_mins": [0.1, 0.1],
                "param_maxs": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
                "sim_name": 'linear',
                "callback": run_sim,
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "n_init": 1,
                    "cov_type": "full",
                    "seed": 0,
                }
            }
        }
    )

    calibration.run()

    # %%
    print(f'All parameter samples at the last iteration:\n {calibration.model.param_data}')

    # %%
    # plt.plot( np.arange(calibration.num_iter),calibration.sigma_list); plt.show()

    # %%
    # calibration.sigma_list,len(calibration.sigma_list),calibration.num_iter
    # print(calibration.sigma_list)

    # %%
    most_prob = np.argmax(calibration.calibration.posterior_ibf)

    # %%
    most_prob_params = calibration.model.param_data[most_prob]

    print(f'Most probable parameter values: {most_prob_params}')
    # %%

    # tests
    error_tolerance = 0.01

    # 1. Testing values of parameters
    error = most_prob_params - [0.2, 5.0]
    assert abs(error[
                   0]) / 0.2 < error_tolerance, f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
    assert abs(error[
                   1]) / 5.0 < error_tolerance, f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"

    # 2. Checking sigma
    assert calibration.sigma_list[-1] < error_tolerance, "Final sigma is bigger than tolerance."


# %%
test_lenreg()
