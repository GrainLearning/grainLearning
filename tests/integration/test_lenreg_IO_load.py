"""
This tutorial shows how to run one iteration of Bayesian calibration for a linear regression model.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem, DynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
executable = f'python {PATH}/linear_model.py'


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
    for i, params in enumerate(system.param_data):
        description = 'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)
        print(" ".join([executable, "%.8e %.8e" % tuple(params), description]))
        os.system(' '.join([executable, "%.8e %.8e" % tuple(params), description]))


def test_lenreg_IO_load():

    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 1,
            "callback": run_sim,
            "system": {
                "system_type": IODynamicSystem,
                "param_min": [0.001, 0.001],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 20,
                "obs_data_file": os.path.abspath(
                    os.path.join(__file__, "../..")) + '/data/linear_sim_data/linear_obs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "sim_data_dir": PATH + '/sim_data/',
                "sim_data_file_ext": '.txt',
                "sigma_tol": 0.01,
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "covariance_type": "full",
                },
            },
            "save_fig": -1,
        }
    )

    # Run one iteration of Bayesian calibration
    calibration.run()
    curr_iter = 0

    # Randomly remove a few generated simulation data files the from curr_iter folder to test the robostness
    # get five random indices
    import numpy as np
    np.random.seed(0)
    ids = np.random.choice(calibration.system.num_samples, size=5, replace=False)
    for i in ids:
        file_to_remove = f'{PATH}/sim_data/iter{curr_iter}/linear_Iter{curr_iter}_Sample{str(i).zfill(2)}'
        # Temporarily remove the files to a tmp data
        if os.path.exists(file_to_remove + '_sim.txt'):
            os.rename(file_to_remove + '_sim.txt', file_to_remove + '_sim.txt.tmp')
            os.rename(file_to_remove + '_param.txt', file_to_remove + '_param.txt.tmp')

    # Make a new calibration instance to load previous results and continue
    calibration_continued = BayesianCalibration.from_dict(
        {
            "curr_iter": curr_iter,
            "num_iter": 1,
            "callback": run_sim,
            "system": {
                "system_type": IODynamicSystem,
                "param_min": [0.001, 0.001],
                "param_max": [1, 10],
                "obs_data_file": os.path.abspath(
                    os.path.join(__file__, "../..")) + '/data/linear_sim_data/linear_obs.dat',
                "obs_names": ['f'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "sim_data_dir": PATH + '/sim_data/',
                "sim_data_file_ext": ".txt",
                "param_names": ['a', 'b'],
            },
            "calibration": {
                "inference": {"ess_target": 0.3},
                "sampling": {
                    "max_num_components": 1,
                    "covariance_type": "full",
                },
            },
            "save_fig": 1,
        }
    )
    calibration_continued.load_and_run_one_iteration()
    calibration_continued.run()
        #: Check if the figures are saved
    assert os.path.isfile(
        PATH + '/sim_data/iter0/linear_param_means.png'), "Figure is not saved"
    assert os.path.isfile(
        PATH + '/sim_data/iter0/linear_param_covs.png'), "Figure is not saved"
    assert os.path.isfile(
        PATH + '/sim_data/iter0/linear_posterior_a.png'), "Figure is not saved"
    assert os.path.isfile(
        PATH + '/sim_data/iter0/linear_posterior_b.png'), "Figure is not saved"
    assert os.path.isfile(
        PATH + '/sim_data/iter0/linear_param_space.png'), "Figure is not saved"
    assert os.path.isfile(
        PATH + '/sim_data/iter0/linear_obs_and_sim.png'), "Figure is not saved"

    # Remove the generated files in the folders that have been created
    os.system(f'rm -rf {PATH}/sim_data*')