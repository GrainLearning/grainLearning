"""Test the SMC class."""
import numpy as np
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

sim_data_dir = "./tests/data/linear_sim_data"
curr_iter = 0


def test_smc():
    """Test the SMC class."""
    calibration = BayesianCalibration.from_dict(
        {
            "curr_iter": curr_iter,
            "num_iter": 0,
            "system": {
                "system_type": IODynamicSystem,
                "obs_data_file": f'{sim_data_dir}/linearObs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "sim_data_dir": sim_data_dir,
                "param_data_file": f'{sim_data_dir}/iter{curr_iter}/smcTable0.txt',
                "param_names": ['a', 'b'],
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {"max_num_components": 1},
            },
        }
    )

    # %%
    # load existing dataset for the test
    file_name = calibration.system.sim_data_dir + \
                f'/iter{calibration.curr_iter}/posterior.npy'
    _, sigma_ref, cov_matrix_ref, posterior_ref = np.load(file_name, allow_pickle=True)
    posterior_ref = posterior_ref.T

    # %%
    # reproduce the result with a given sigma value
    calibration.load_and_process(sigma_ref)
    # ~ calibration.load_and_run_one_iteration()
    posterior = calibration.calibration.inference.posteriors

    # %%
    # check (co)variance and posterior distribution
    cov_matrices = calibration.calibration.inference.get_covariance_matrices(sigma_ref, calibration.system)
    np.testing.assert_allclose(cov_matrix_ref, cov_matrices[-1], err_msg="The (co)variances do not match.")
    np.testing.assert_allclose(posterior, posterior_ref, err_msg="The posterior distributions do not match.")


test_smc()
