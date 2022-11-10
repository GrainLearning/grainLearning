
#%%

import numpy as np

from grainlearning import CalibrationToolbox
from grainlearning.models import IOModel


executable = 'python ./tutorials/linear_regression/LinearModel.py'

def run_sim(model, **kwargs):
	from math import floor, log
	import os
	# keep the naming convention consistent between iterations
	magn = floor(log(model.num_samples, 10)) + 1
	curr_iter = kwargs['curr_iter']
	# check the software name and version
	print("*** Running external software... ***\n")
	# loop over and pass parameter samples to the executable
	for i, params in enumerate(model.param_data):
		description = 'Iter'+str(curr_iter)+'-Sample'+str(i).zfill(magn)
		print(" ".join([executable, '%.8e %.8e'%tuple(params), description]))
		os.system(' '.join([executable, '%.8e %.8e'%tuple(params), description]))


calibration = CalibrationToolbox.from_dict(
    {
        "num_iter": 10,
        "model": {
            "param_mins": [0.001, 0.001],
            "param_maxs": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_data_file": 'linearObs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": './tutorials/linear_regression/',
            "sim_data_file_ext": '.txt',
            "callback": run_sim,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "n_init": 1,
                "seed": 0,
            }
        },
        "save_fig": 0,
        "model_type": IOModel
    }
)

calibration.run()

#%%
print(f'All parameter samples at the last iteration:\n {calibration.model.param_data}')

#%%
# plt.plot( np.arange(calibration.num_iter),calibration.sigma_list); plt.show()

#%%
# calibration.sigma_list,len(calibration.sigma_list),calibration.num_iter
# print(calibration.sigma_list)

# %%
most_prob = np.argmax(calibration.calibration.posterior_ibf)

# %%
most_prob_params = calibration.model.param_data[most_prob]

print(f'Most probable parameter values: {most_prob_params}')
# %%

#tests
error_tolerance = 0.01

#1. Testing values of parameters
error = most_prob_params - [0.2,5.0]
assert abs(error[0])/0.2 < error_tolerance, f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
assert abs(error[1])/5.0 < error_tolerance, f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"

#2. Checking sigma
assert calibration.sigma_list[-1] < error_tolerance, "Final sigma is bigger than tolerance."

# %%
