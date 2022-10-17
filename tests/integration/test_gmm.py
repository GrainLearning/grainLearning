
#%%
 
import numpy as np
import matplotlib.pylab as plt

from grainlearning import CalibrationToolbox
from grainlearning.models import IOModel

import os, sys
sys.path.append(os.getcwd() +'/grainlearning')

from tools import resampledParamsTable

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
            "sampling": {
                "max_num_components": 1,
                "seed": 0,
            },
        },
        "model_type": IOModel
    }
)

#%%
# load existing dataset for the test
file_name = calibration.model.sim_data_dir +\
    f'/iter{calibration.curr_iter}/posterior.npy'
resampled_param_data_ref, sigma_ref, cov_matrix_ref, posterior_ref = np.load(file_name, allow_pickle=True)
posterior_ref = posterior_ref.T

#%% 
# reproduce the result with a given sigma value
calibration.load_and_process(sigma_ref)
resampled_param_data = calibration.resample()
posterior = calibration.calibration.inference.posteriors

#%%
# check (co)variance and posterior distribution 
cov_matrices = calibration.calibration.inference.get_covariance_matrices(sigma_ref, calibration.model)
np.testing.assert_allclose(cov_matrix_ref, cov_matrices[-1], err_msg = "The (co)variances do not match.")
np.testing.assert_allclose(posterior, posterior_ref, err_msg = "The posterior distributions do not match.")

#%%
# compare the existing dataset and the new resampled data using two approaches
paramRanges = {'a': [0, 1], 'b': [0.0, 10]}
newSmcSamples, newparamsFile, gmm, maxNumComponents = resampledParamsTable(keys=calibration.model.param_names,
            smcSamples=calibration.model.param_data,
            proposal=calibration.calibration.posterior_ibf,
            ranges=paramRanges,
            num=calibration.model.num_samples,
            maxNumComponents=1,
            priorWeight=1,
            covType='full',
            tableName='table_new.txt',
            seed=calibration.calibration.sampling.seed,
            simNum=1)            

# ~ plt.plot(resampled_param_data_ref[:,0], resampled_param_data_ref[:,1], 'o', label='ref')
# ~ plt.plot(resampled_param_data[:,0], resampled_param_data[:,1], 'o', label='rewrite')
# ~ plt.plot(newSmcSamples[:,0], newSmcSamples[:,1], 'o', label='rewrite_old_mthod')
# ~ plt.legend()
# ~ plt.show()

#%%
check_list = np.isclose(resampled_param_data_ref, resampled_param_data)
check_list = check_list[:,0] & check_list[:,1]
percentage = len(check_list[check_list==True])/calibration.model.num_samples
assert percentage > 0.8, f"Parameter data resampled from the proposal distribution do not match. Mismatch is {100*(1-percentage)}%"
