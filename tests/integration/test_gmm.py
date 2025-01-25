"""Test the Gaussian Mixture Model class."""
import os
import sys
import numpy as np

from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

sys.path.append(os.path.join(os.getcwd(), "grainlearning"))
sim_data_dir = "./tests/data/linear_sim_data"
curr_iter = 0


def test_gmm():
    calibration = BayesianCalibration.from_dict(
        {
            "curr_iter": curr_iter,
            "num_iter": 0,
            "system": {
                "system_type": IODynamicSystem,
                "obs_data_file": f'{sim_data_dir}/linear_obs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "param_data_file": f'{sim_data_dir}/iter{curr_iter}/smcTable0.txt',
                "sim_data_dir": sim_data_dir,
                "param_names": ['a', 'b'],
            },
            "inference": {
                "Bayes_filter": {
                    "ess_target": 0.3,
                    "scale_cov_with_max": True},
                "sampling": {
                    "max_num_components": 1,
                    "covariance_type": "full",
                    "random_state": 0,
                },
            },
        }
    )

    # %%
    # load existing dataset for the test
    file_name = calibration.system.sim_data_dir + f'/iter{calibration.curr_iter}/posterior.npy'
    resampled_param_data_ref, sigma_ref, cov_matrix_ref, posterior_ref = np.load(file_name, allow_pickle=True)
    posterior_ref = posterior_ref.T

    # %%
    # reproduce the result with a given sigma value
    calibration.load_and_process(sigma_ref)
    resampled_param_data = calibration.resample()
    posterior = calibration.inference.Bayes_filter.posteriors

    # %%
    # check (co)variance and posterior distribution
    cov_matrices = calibration.inference.Bayes_filter.get_covariance_matrices(sigma_ref, calibration.system)
    np.testing.assert_allclose(cov_matrix_ref, cov_matrices[-1], err_msg="The (co)variances do not match.")
    np.testing.assert_allclose(posterior, posterior_ref, err_msg="The posterior distributions do not match.")

    # %%
    # write new parameter table to the simulation directory
    calibration.system.write_params_to_table()

    # %%
    check_list = np.isclose(resampled_param_data_ref, resampled_param_data)
    check_list = check_list[:, 0] & check_list[:, 1]
    percentage = len(check_list[check_list == True]) / calibration.system.num_samples
    assert percentage > 0.8, f"Parameter data resampled from the proposal distribution do not match." \
                             f"Mismatch is {100 * (1 - percentage)}%"


test_gmm()
