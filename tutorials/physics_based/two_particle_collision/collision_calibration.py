"""
 This tutorial shows how to perform iterative Bayesian calibration for a DEM simulation of two particle colliding
 using GrainLearning. The simulation is performed using Yade on a desktop computer.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
executable = 'yade-batch'
yade_script = f'{PATH}/Collision.py'


def run_sim(calib):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    print("*** Running external software YADE ... ***\n")
    os.system(' '.join([executable, calib.system.param_data_file, yade_script]))


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 4,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [7, 0.0],
            "param_max": [11, 0.5],
            "param_names": ['E_m', 'nu'],
            "num_samples": 10,
            "obs_data_file": PATH + '/collision_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'collision',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.01,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
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

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')
