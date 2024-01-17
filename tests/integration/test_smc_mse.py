"""Test the SMC class and compare the least error"""
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from grainlearning import BayesianCalibration

p1 = 0.2
p2 = 5.0
x_obs = np.arange(100)
y_obs = p1 * x_obs + p2
y_obs_w_noise = y_obs + np.random.rand(100) * 2.5


def run_sim(calib):
    """Run the linear model"""
    data = []
    for params in calib.system.param_data:
        y_sim = params[0] * calib.system.ctrl_data + params[1]
        data.append(np.array(y_sim, ndmin=2))

    calib.system.set_sim_data(data)


def test_smc_mse():
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 0,
            "callback": run_sim,
            "system": {
                "param_min": [0, 0],
                "param_max": [1, 10],
                "num_samples": 13,
                "obs_data": y_obs,
                "ctrl_data": x_obs,
            },
            "calibration": {
                "inference": {
                    "ess_target": 0.3,
                    "scale_cov_with_max": True
                },
                "sampling": {"max_num_components": 1},
            }
        }
    )

    calibration.run_one_iteration()
    most_prob = np.argmax(calibration.calibration.posterior)
    # most_prob_params = calibration.system.param_data[most_prob]
    least_err = np.argmin(
        [mse(calibration.system.sim_data[sid, 0, :], y_obs) for sid in range(calibration.system.num_samples)])

    assert most_prob == least_err, f"most probable does not have the least MAE {most_prob=} {least_err=}"


test_smc_mse()
