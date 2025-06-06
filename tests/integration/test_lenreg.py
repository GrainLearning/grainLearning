import numpy as np
from grainlearning import BayesianCalibration

x_obs = np.arange(100)
y_obs = 0.2 * x_obs + 5.0


# y_obs += np.random.rand(100) * 2.5

def run_sim(calib):
    """Run the linear model"""
    data = []
    for params in calib.system.param_data:
        y_sim = params[0] * calib.system.ctrl_data + params[1]
        data.append(np.array(y_sim, ndmin=2))

    calib.system.set_sim_data(data)


def test_lenreg():
    """Test the linear regression example"""
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 10,
            "callback": run_sim,
            "system": {
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
                "sim_name": 'linear',
            },
            "inference": {
                "Bayes_filter": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "n_init": 1,
                    "covariance_type": "full",
                    "random_state": 0,
                }
            }
        }
    )

    calibration.run()

    # %%
    print(f'All parameter samples at the last iteration:\n {calibration.system.param_data}')

    # %%
    # plt.plot( np.arange(calibration.num_iter),calibration.sigma_list); plt.show()

    # %%
    # calibration.sigma_list,len(calibration.sigma_list),calibration.num_iter
    # print(calibration.sigma_list)

    # %%
    most_prob = np.argmax(calibration.inference.posterior)

    # %%
    most_prob_params = calibration.system.param_data[most_prob]

    # tests
    error_tolerance = 0.01

    # 1. Testing values of parameters
    error = most_prob_params - [0.2, 5.0]
    assert abs(error[0]) / 0.2 < error_tolerance, \
        f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
    assert abs(error[1]) / 5.0 < error_tolerance, \
        f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"

    # 2. Checking sigma
    assert calibration.inference.sigma_list[-1] < error_tolerance, "Final sigma is bigger than tolerance."

    # %% Test other stopping criteria
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 10,
            "error_tol": 0.01,
            "callback": run_sim,
            "system": {
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
                "sim_name": 'linear',
            },
            "inference": {
                "Bayes_filter": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "n_init": 1,
                    "covariance_type": "full",
                    "random_state": 0,
                }
            }
        }
    )
    calibration.run()
    assert np.min(calibration.error_array) < 0.01, "Error tolerance is not met."


    # %% Test other stopping criteria
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 10,
            "gl_error_tol": 0.01,
            "callback": run_sim,
            "system": {
                "param_min": [0.1, 0.1],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
                "sim_name": 'linear',
            },
            "inference": {
                "Bayes_filter": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "n_init": 1,
                    "covariance_type": "full",
                    "random_state": 0,
                }
            }
        }
    )
    calibration.run()
    assert calibration.gl_errors[-1] < 0.01, "Error tolerance is not met."

# %%
test_lenreg()
