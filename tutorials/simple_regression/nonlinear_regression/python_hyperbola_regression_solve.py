"""
This tutorial shows how to link a nonlinear regression model implemented in Python to GrainLearning.
The only changes made to the linear regression case are the definition of
the observed data `y_obs` and the model `nonlinear`.
"""
import numpy as np
from grainlearning import BayesianCalibration

x_obs = np.arange(100)
# hyperbola in a form similar to the Duncan-Chang material model, q = \eps / (a * 100 + b * \eps)
y_obs = x_obs / (0.2 * 100 + 5.0 * x_obs)


def run_sim(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    data = []
    for params in calib.system.param_data:
        # Run the model
        y_sim = nonlinear(calib.system.ctrl_data, params)
        data.append(np.array(y_sim, ndmin=2))
    calib.system.set_sim_data(data)


def nonlinear(x, params):
    """
    A hyperbola in a form similar to the Duncan-Chang material model,

    .. math::
        sig = eps / (a * 100 + b * eps) where s is stress and e is strain.

        y_t & = h(x_t) + r_t

    where
    :math:`sig` is stress,
    :math:`eps` is strain in percent,
    :math:`a` and `b` are the material parameters.
    """
    return x_obs / (params[0] * 100 + params[1] * x_obs)


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 10,
        "callback": run_sim,
        "system": {
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "sim_name": 'nonlinear',
            "sigma_tol": 0.01,
        },
        "inference": {
            "Bayes_filter": {
                "ess_target": 0.3,
                "scale_cov_with_max": True,
            },
            "sampling": {
                "max_num_components": 1,
                "n_init": 1,
                "random_state": 0,
                "slice_sampling": True,
            },
            "initial_sampling": "halton",
        },
        "save_fig": -1,
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
