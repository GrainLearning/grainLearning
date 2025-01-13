"""
 This tutorial shows how to perform iterative Bayesian calibration for a DEM simulation of two particle colliding
 using GrainLearning. The simulation is performed using Yade on a desktop computer.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
curr_iter = 0


calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 0,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [7, 0.0, 0.0, 0.0, 10.0],
            "param_max": [11, 0.5, 1.0, 1.0, 50.0],
            "param_names": ['E_m', 'v', 'kr', 'eta', 'mu'],
            "num_samples": 15,
            "obs_data_file": PATH + '/triax_data_DEM.dat',
            "obs_names": ['e_v', 's33_over_s11'],
            "ctrl_name": 'e_z',
            "sim_name": 'triax',
            "sim_data_dir": PATH + '/sim_data_backup_2025_01_13_20_58_49/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.01,
        },
        "calibration": {
            "inference": {"ess_target": 0.3, "scale_cov_with_max": True},
            "sampling": {
                "max_num_components": 2,
                "n_init": 1,
                "random_state": 0,
                "covariance_type": "full",
            }
        },
        "save_fig": 0,
    }
)

calibration.load_and_run_one_iteration()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')
