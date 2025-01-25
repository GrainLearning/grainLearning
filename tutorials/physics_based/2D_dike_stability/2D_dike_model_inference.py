"""
This tutorial shows how to perform iterative Bayesian calibration for a dike regression model
 using GrainLearning.
"""
import os
from math import floor, log

import numpy as np

from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem
from grainlearning.tools import plot_pdf, plot_param_data
import matplotlib.pylab as plt

from joblib import Parallel, delayed

PATH = os.path.abspath(os.path.dirname(__file__))
executable = f'python {PATH}/2D_dike.py'


def run_sim(calib):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    system = calib.system
    # keep the naming convention consistent between iterations
    mag = floor(log(system.num_samples, 10)) + 1
    # check the software name and version
    print("*** Running external software... ***\n")
    # loop over and pass parameter samples to the executable
    Parallel(n_jobs=num_cores)(
        delayed(dike)(params, system.sim_name, 'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)) for
        i, params in enumerate(system.param_data))
    # plot the samples and the comparison between observation and prediction
    if calib.curr_iter == 0:
        return

    # get the path to save the figures
    path = f'{system.sim_data_dir}/iter{calib.curr_iter}' \
        if isinstance(system, IODynamicSystem) \
        else f'./{system.sim_name}/iter{calib.curr_iter}'
    if not os.path.exists(path):
        os.makedirs(path)
    fig_name = f'{path}/{system.sim_name}'

    # get the id of sample that has the highest probability
    most_prob_params_id = np.argmax(calib.inference.posterior)
    sim_data = system.sim_data[most_prob_params_id, :, :]
    obs_data = system.obs_data

    # compute the error norm
    sim_data_norm = np.sqrt(sim_data[0::2] ** 2 + sim_data[1::2] ** 2)
    obs_data_norm = np.sqrt(obs_data[0::2] ** 2 + obs_data[1::2] ** 2)
    error_norm = sim_data_norm - obs_data_norm

    # plot ground truth, inferred field, and error
    lims_truth = dict(cmap='RdBu_r', vmin=obs_data_norm.min(), vmax=obs_data_norm.max())
    lims_error = dict(cmap='RdBu_r', vmin=error_norm.min(), vmax=error_norm.max())
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    im0 = ax[0].scatter(coords[:, 0], coords[:, 1], 100, obs_data_norm, edgecolor='w', lw=0.1, **lims_truth)
    im1 = ax[1].scatter(coords[:, 0], coords[:, 1], 100, sim_data_norm, edgecolor='w', lw=0.1, **lims_truth)
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].scatter(coords[:, 0], coords[:, 1], 100, error_norm, edgecolor='w', lw=0.1, **lims_error)
    fig.colorbar(im2, ax=ax[2])
    for j in range(3):
        ax[j].set_xlabel('x')
        ax[j].set_ylabel('y')
        ax[0].set_title('Ground truth')
        ax[1].set_title('Inferred field')
        ax[2].set_title('Error')

    # plot resampling in parameter space
    plot_param_data(
        fig_name,
        system.param_names,
        calib.inference.param_data_list,
        save_fig=0
    )
    plt.show()


def dike(params, sim_name, description):
    print(" ".join([executable, "%.8e %.8e %.8e" % tuple(params), sim_name, description, str(num_obs)]))
    os.system(' '.join([executable, "%.8e %.8e %.8e" % tuple(params), sim_name, description, str(num_obs)]))


# overwrite a virtual function get_normalization_factor of the smc class
def set_normalization_factor(self):
    self.normalization_factor = 1. / abs(np.mean(self.obs_data, axis=0))


#
IODynamicSystem.set_normalization_factor = set_normalization_factor

# %% Generate synthetic data
sim_name = 'dike'
description = 'synth_data'
true_params = [1e5, 30, 5]
coef_of_variation = 0.2
num_obs = 100
num_cores = 18
param_mins = list((1 - 1 * coef_of_variation) * np.array(true_params))
param_maxs = list((1 + 3 * coef_of_variation) * np.array(true_params))
os.system(' '.join([executable, "%.8e %.8e %.8e" % tuple(true_params), sim_name, description, str(num_obs)]))

# read key of the synthetic data from the first line
with open(f'{sim_name}_{description}_sim.txt', 'r') as file:
    # Read the first line
    first_line = file.readline()
ctrl_name = first_line.split()[1]
obs_names = first_line.split()[2:]
coords = np.load(f'{sim_name}_{description}_input_coords.npy')
n_iter = int(coords.shape[0] / num_obs)
coords = coords[np.arange(0, coords.shape[0], coords.shape[0] / num_obs, dtype=int)]

param_names = ['E', 'phi', 'coh']
num_samples = int(5 * len(param_names) * log(len(param_names)))

# %% Define the calibration
calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 10,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": param_mins,
            "param_max": param_maxs,
            "param_names": param_names,
            "num_samples": num_samples,
            "obs_data_file": PATH + '/dike_synth_data_sim.txt',
            "obs_names": obs_names,
            "ctrl_name": 't',
            "sim_name": sim_name,
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.1,
            "sigma_max": 10,
        },
        "inference": {
            "Bayes_filter": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "random_state": 0,
                "slice_sampling": True,
            },
            "initial_sampling": "sobol",
        },
        "save_fig": -1,
    }
)

# %% Run the calibration
calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

error_tolerance = 0.1
errors = most_prob_params - true_params

for error, true_param, prob_param in zip(errors, true_params, most_prob_params):
    assert abs(
        error) / true_param < error_tolerance, \
        f"Model parameters are not correct, expected {true_param} but got {prob_param}"

plot_pdf('2D_dike', param_names, calibration.inference.param_data_list, save_fig=0,
         true_params=true_params)
plt.show()
