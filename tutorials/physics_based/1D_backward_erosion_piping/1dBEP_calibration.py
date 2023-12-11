"""
This tutorial shows how to perform iterative Bayesian calibration for a linear regression model
 using GrainLearning.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem
from joblib import Parallel, delayed
import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))
executable = f'./RBEP2D.out'

def run_func(i, params, system, mag):
    description = 'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)
    #print(" ".join([executable, "%.8e %.8e" % tuple(params), system.sim_name, description]))
    os.system(' '.join([executable, "%.8e %.8e" % tuple(params), system.sim_name, description]))


def run_sim(system):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    # keep the naming convention consistent between iterations
    mag = floor(log(system.num_samples, 10)) + 1
    # check the software name and version
    print("*** Running external software... ***\n")

    # loop over and pass parameter samples to the executable in parallel
    Parallel(n_jobs=20)(delayed(run_func)(i, params, system, mag) for i, params in enumerate(system.param_data))
    os.system('for f in *sim.t*; do mv -- "$f" "${f%.txt*}.txt" 2> /dev/null; done')
    os.system('for f in *param.t*; do mv -- "$f" "${f%.txt*}.txt" 2> /dev/null; done')

    for i, params in enumerate(system.param_data):
        get_difference(system.sim_name+'_'+'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)+'_sim.txt', '1d_baseline.txt')



def write_dict_to_file(data, file_name):
    """
    write a python dictionary data into a text file
    """
    with open(file_name, 'w') as f:
        keys = data.keys()
        f.write('# ' + ' '.join(keys) + '\n')
        num = len(data[list(keys)[0]])
        for i in range(num):
            f.write(' '.join([str(data[key][i]) for key in keys]) + '\n')



def get_difference(file_name: str, file_name2: str):
    delimiters = ['\t', ' ', ',']
    data = np.absolute(np.subtract(np.genfromtxt(file_name, ndmin=2),np.genfromtxt(file_name2, ndmin=2)))

    with open(file_name, 'r') as f_open:
        first_line = f_open.read().splitlines()[0]
        for d in delimiters:
            keys = first_line.split(d)
            # remove # in the header line
            if '#' in keys:
                keys.remove('#')
            # remove empty strings from the list
            keys = list(filter(None, keys))
            if len(keys) == data.shape[1]:
                break

    # store data in a dictionary
    keys_and_data = {}
    for key in keys:
        if '#' in key:
            key_no_hash = key.split(' ')[-1]
        else:
            key_no_hash = key
        keys_and_data[key_no_hash] = data[:, keys.index(key)]

    #write_dict_to_file(keys_and_data,file_name)
    with open(file_name, 'w') as f:
        keys = keys_and_data.keys()
        f.write('# ' + ' '.join(keys) + '\n')
        num = len(keys_and_data[list(keys)[0]])
        for i in range(num):
            f.write(' '.join([str(keys_and_data[key][i]) for key in keys]) + '\n')
    return


def RBEP_calibration():
    calibration = BayesianCalibration.from_dict(
        {
            "num_iter": 10,
            "system": {
                "system_type": IODynamicSystem,
                "param_min": [-0.5, -4.5],
                "param_max": [0.5, -3],
                "param_names": ['a', 'b'],
                "num_samples": 100,
                "obs_data_file": PATH + '/linear_obs.dat',
                "obs_names": ['f','g','h','i','j','k','l','m','n','o','p','q','s','t','t','v','w','x','y','z'],
                "ctrl_name": 'u',
                "sim_name": 'linear',
                "sim_data_dir": PATH + '/sim_data/',
                "sim_data_file_ext": '.txt',
                "sigma_tol": 0.2,
                "callback": run_sim,


            },
            "calibration": {
                "inference": {
                    "ess_target": 0.3,
                    "scale_cov_with_max": False,
                },
                "sampling": {
                    "max_num_components": 1,
                    "covariance_type": "full",
                    "slice_sampling": False,
                }
            },
            "save_fig": -1,
        }
    )
    calibration.run()
    most_prob_params = calibration.get_most_prob_params()
    print(f'Most probable parameter values: {most_prob_params}')
    print(f'The ensemble prediction of parameter values: {calibration.system.estimated_params[-1]}')
    # write the most probable parameter values to a file
    with open('results.txt', 'a') as f:
        f.write(' '.join([f'{param}' for param in most_prob_params] +\
                         [f'{param}' for param in calibration.system.estimated_params[-1]]) + '\n')


os.system('./RBEP2D.out 0.15 -3.55 linear obs a')
os.system('mv linear_obs_sim.txt linear_obs.dat 2> /dev/null')
get_difference('linear_obs.dat', 'linear_obs_baseline.dat')

for i in range(100):
    RBEP_calibration()
