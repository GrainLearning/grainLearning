"""
Bayesian calibration for the DEM column collapse scenario using GrainLearning.

This mirrors the triaxial calibration driver, using the YADE model in this folder.
"""
import os
from math import log
from grainlearning import BayesianCalibration, IODynamicSystem


PATH = os.path.abspath(os.path.dirname(__file__))
executable = 'yade-batch'
yade_script = f'{PATH}/column_collapse.py'


def run_sim(calib):
    """
    Run the external YADE executable with the parameter table produced by GrainLearning.
    """
    print("*** Running external software YADE ... ***\n")
    os.system(' '.join([executable, calib.system.param_data_file, yade_script]))


# Choose parameters to calibrate; keep consistent with YADE's readParamsFromTable
param_names = ['kr', 'eta', 'mu']
num_samples = int(5 * len(param_names) * log(len(param_names)))

calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": 0,
        "num_iter": 5,
        "error_tol": 0.01,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.0, 0.0, 1.0],
            "param_max": [1.0, 1.0, 60.0],
            "param_names": param_names,
            "num_samples": num_samples,
            # Create this by running column_collapse.py once (non-batch) to generate the reference curve
            "obs_data_file": PATH + '/column_collapse_DEM_test_run_sim.txt',
            "obs_names": ['run_out', 'final_height', 'com_x', 'com_y'],
            "ctrl_name": 't',
            "sim_name": 'column_collapse',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
        },
        "inference": {
            "Bayes_filter": {"scale_cov_with_max": True,
                             "ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "slice_sampling": True,
            },
        },
        "save_fig": 1,
        "threads": 1,
    }
)

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

calibration.save_fig = 1
calibration.plot_uq_in_time()
