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
        "curr_iter": 0,
        "num_iter": 5,
        "error_tol": 0.1,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.0, 0.0, 1.0],
            "param_max": [1.0, 1.0, 60.0],
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
                "covariance_type": "tied",
                "slice_sampling": True,
            },
        },
        "save_fig": -1,
        "threads": 1,
    }
)

# calibration = BayesianCalibration(
#     num_iter=5,
#     error_tol=0.1,
#     callback=run_sim,
#     save_fig=-1,
#     threads=1,
#     system=IODynamicSystem(
#         param_min=[0.0, 0.0, 10.0],
#         param_max=[10.0, 1.0, 60.0],
#         param_names=param_names,
#         num_samples=num_samples,
#         obs_data_file=PATH + '/triax_DEM_test_run_sim.txt',
#         obs_names=['e_v', 's33_over_s11'],
#         ctrl_name='e_z',
#         sim_name='triax',
#         sim_data_dir=PATH + '/sim_data/',
#         sim_data_file_ext='.txt',
#     ),
#     inference=IterativeBayesianFilter(
#         SMC(
#             scale_cov_with_max=True,
#         ),
#         GaussianMixtureModel(
#             max_num_components=5,
#             slice_sampling=True,
#             covariance_type=tied,
#         ),
#     ),    
# )

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

calibration.save_fig = 0
calibration.plot_uq_in_time()