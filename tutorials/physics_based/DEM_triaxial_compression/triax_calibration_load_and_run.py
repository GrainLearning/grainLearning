"""
 This tutorial shows how to perform iterative Bayesian calibration for a DEM simulation of triaxial compression
 using GrainLearning. The simulations can be performed using Yade on a desktop computer.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration, IODynamicSystem, IterativeBayesianFilter, SMC, GaussianMixtureModel

PATH = os.path.abspath(os.path.dirname(__file__))
executable = 'yade-batch'
yade_script = f'{PATH}/triax_YADE_DEM_model.py'
curr_iter = 0


def run_sim(calib):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    print("*** Running external software YADE ... ***\n")
    os.system(' '.join([executable, calib.system.param_data_file, yade_script]))


param_names = ['kr', 'eta', 'mu']
num_samples = int(5 * len(param_names) * log(len(param_names)))
calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 5,
        "error_tol": 0.1,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_names": param_names,
            "num_samples": num_samples,
            "obs_data_file": PATH + '/triax_DEM_test_run_sim.txt',
            "obs_names": ['e', 's33_over_s11'],
            "ctrl_name": 'e_z',
            "sim_name": 'triax',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
        },
        "inference": {
            "Bayes_filter": {"scale_cov_with_max": True},
            "sampling": {
                "max_num_components": 5,
                "random_state": 0,
                "slice_sampling": True,
            }
        },
        "save_fig": 0,
        "threads": 1,
    }
)

calibration.load_and_run_one_iteration()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

# Turn off the following if you want to continue the calibration. Always increase the current iteration number before continuing the calibration
calibration.increase_curr_iter()
calibration.save_fig = -1
calibration.run()