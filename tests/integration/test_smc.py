
#%%
 
import numpy as np
import matplotlib.pylab as plt

from grainlearning import CalibrationToolbox
from grainlearning.models import IOModel

calibration = CalibrationToolbox.from_dict(
    {
        "curr_iter": 0,
        "num_iter": 0,
        "model": {
            "obs_data_file": 'linearObs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": './tests/data/linear_sim_data',
            "param_data_file": 'smcTable0.txt',
            "param_names": ['a', 'b'],
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {"max_num_components": 1},
        },
        "model_type": IOModel
    }
)

#%%
# load existing dataset for the test
file_name = calibration.model.sim_data_dir +\
	f'/iter{calibration.curr_iter}/posterior.npy'
_, sigma_ref, cov_matrix_ref, posterior_ref = np.load(file_name, allow_pickle=True)
posterior_ref = posterior_ref.T

#%% 
# reproduce the result with a given sigma value
calibration.load_and_process(sigma_ref)
posterior = calibration.calibration.inference.posteriors

#%%
# check (co)variance and posterior distribution 
cov_matrices = calibration.calibration.inference.get_covariance_matrices(sigma_ref, calibration.model)
np.testing.assert_allclose(cov_matrix_ref, cov_matrices[-1], err_msg = "The (co)variances do not match.")
np.testing.assert_allclose(posterior, posterior_ref, err_msg = "The posterior distributions do not match.")
