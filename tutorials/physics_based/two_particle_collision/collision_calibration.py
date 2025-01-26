"""
 This tutorial shows how to perform iterative Bayesian calibration for a DEM simulation of two particle colliding
 using GrainLearning. The simulation is performed using Yade on a desktop computer.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
executable = 'yadedaily-batch'
yade_script = f'{PATH}/Collision.py'

# define the callback function which GrainLearning uses to run YADE simulations
def run_sim(calib):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    print("*** Running external software YADE ... ***\n")
    os.system(' '.join([executable, calib.system.param_data_file, yade_script]))

param_names = ['E_m', 'nu']
num_samples = int(5 * len(param_names) * log(len(param_names)))

# define the Bayesian Calibration object
calibration = BayesianCalibration.from_dict(
    {
        # maximum number of GL iterations
        "num_iter": 5,
        # error tolerance to stop the calibration
        "error_tol": 0.1,
        # call back function to run YADE 
        "callback": run_sim,
        # DEM model as a dynamic system
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [7, 0.0],
            "param_max": [11, 0.5],
            "param_names": param_names,
            "num_samples": 10,
            "obs_data_file": PATH + '/collision_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'collision',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.01,
        },
        "inference": {
            "Bayes_filter": {
                # scale the covariance matrix with the maximum observation data or not
                "scale_cov_with_max": False,
            },
            "sampling": {
                # maximum number of distribution components
                "max_num_components": 1,
                # fix the seed for reproducibility
                "random_state": 0,
                # type of covariance matrix
                "covariance_type": "full",
                # use slice sampling (set to False if faster convergence is designed. However, the results could be biased)
                "slice_sampling": True,
            }
        },
        # flag to save the figures (-1: no, 0: yes but only show the figures , 1: yes)
        "save_fig": 0,
        # number of threads to be used by the external software
        "threads": 16,
    }
)

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')
