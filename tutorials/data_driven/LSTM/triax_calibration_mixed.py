"""
 This tutorial shows how to perform iterative Bayesian calibration for a DEM simulation of triaxial compression
 using GrainLearning. The simulations can be performed using Yade on a desktop computer.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration, IODynamicSystem, IterativeBayesianFilter, SMC, GaussianMixtureModel
import grainlearning.rnn.train as train_rnn
import grainlearning.rnn.predict as predict_rnn
from grainlearning.rnn import preprocessor
import numpy as np
from grainlearning.tools import write_to_table


PATH = '/home/hcheng/GrainLearning/grainLearning/tutorials/physics_based/DEM_triaxial_compression'
executable = 'yade-batch'
yade_script = f'{PATH}/triax_YADE_DEM_model.py'


# Define the configuration dictionary for the ML surrogate
my_config = {
    'input_data': None,
    'param_data': None,
    'output_data': None,
    'train_frac': 0.7,
    'val_frac': 0.2,
    'window_size': 20,
    'window_step': 1,
    'patience': 25,
    'epochs': 100,
    'learning_rate': 1e-4,
    'lstm_units': 128,
    'dense_units': 128,
    'batch_size': 64,
    'standardize_outputs': True,
    'save_weights_only': True
}
    
def run_sim_original(calib):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    print("*** Running external software YADE ... ***\n")

    # create a directory to store simulation data
    sim_data_dir = calib.system.sim_data_dir.rstrip('/')
    calib.system.sim_data_sub_dir = f'{sim_data_dir}/iter{calib.system.curr_iter}'

    if calib.curr_iter > 0:
        param_data = calib.system.param_data[calib.ids_origin]
    else:
        param_data = calib.system.param_data

    param_data_file = write_to_table(
        calib.system.sim_name,
        param_data,
        calib.system.param_names,
        calib.system.curr_iter,
        threads=calib.threads,
    )

    os.system(' '.join([executable, param_data_file, yade_script]))

    calib.system.move_data_to_sim_dir()

    calib.system.get_sim_data_files(param_data.shape[0])

    if calib.curr_iter > 0:
        calib.system.load_sim_data(param_data.shape[0], calib.ids_origin)
    else:
        calib.system.load_sim_data(param_data.shape[0])

    return calib.system.sim_data


def run_sim_surrogate(params_origin, output_origin, params_surrogate):
    """Train the ML surrogate and evaluate model output with the ML surrogate.

    :param params_origin: The parameter data used by the original model.
    :param output_origin: The output data produced by the original model.
    :param params_surrogate: The parameter data to be used by the ML surrogate.
    """
    # expend the parameter and output data
    my_config['param_data'] = np.vstack([my_config['param_data'], params_origin])
    my_config['output_data'] = np.vstack([my_config['output_data'], output_origin])

    preprocessor_lstm = preprocessor.PreprocessorLSTM.from_dict(my_config)
    _ = train_rnn.train_without_wandb(preprocessor_lstm, config=my_config)
    model, train_stats, config = predict_rnn.get_pretrained_model(my_config['output_dir'])

    # run the surrogate for the second half of the samples
    data_inputs = preprocessor_lstm.prepare_input_data(params_surrogate)
    # make predictions with the trained model
    output_surrogate = predict_rnn.predict_batch(model, data_inputs, train_stats, config,
                                                 batch_size=params_surrogate.shape[0])
    # converting the predictions to GL format (temporal dimension at the end)
    output_surrogate = np.moveaxis(output_surrogate, 1, -1)

    return output_surrogate


# 3. Define the callback function using the ML surrogate
def run_sim_mixed(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    # if first iteration, run the original function
    if calib.curr_iter == 0:
        calib.system.set_sim_data(run_sim_original(calib))
        my_config['input_data'] = calib.system.ctrl_data
        my_config['param_data'] = calib.system.param_data
        my_config['output_data'] = calib.system.sim_data
    else:
        # split samples into two subsets to be used with the original function and the ML surrogate
        np.random.seed()
        ids = np.random.permutation(len(calib.system.param_data))
        split_index = int(len(ids) * 0.5)
        ids_origin, ids_surrogate = ids[:split_index], ids[split_index:]
        calib.ids_origin, calib.ids_surrogate = ids_origin, ids_surrogate

        # run the original function for the first half of the samples
        param_data_origin = calib.system.param_data[ids_origin]
        sim_data_origin = run_sim_original(calib)

        # run the surrogate for the second half of the samples
        my_config['output_dir'] = f'{calib.system.sim_data_dir}/iter{calib.curr_iter}/surrogate'
        param_data_surrogate = calib.system.param_data[ids_surrogate]
        sim_data_surrogate = run_sim_surrogate(param_data_origin, sim_data_origin, param_data_surrogate)

        # put the two subsets of simulation data together according to the original order
        sim_data = np.zeros([calib.system.num_samples, calib.system.num_obs, calib.system.num_steps])
        sim_data[ids_surrogate] = sim_data_surrogate
        sim_data[ids_origin] = sim_data_origin

        # set `sim_data` to system
        calib.system.set_sim_data(sim_data)


param_names = ['kr', 'eta', 'mu']
num_samples = int(5 * len(param_names) * log(len(param_names)))
obs_data = np.loadtxt(PATH + '/triax_DEM_test_run_sim.txt').T
ctrl_data = obs_data[1]
obs_data = np.vstack([obs_data[0], obs_data[-1]])
calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": 0,
        "num_iter": 5,
        "error_tol": 0.1,
        "callback": run_sim_mixed,
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
            "Bayes_filter": {
                "scale_cov_with_max": True,
                "init_inference_step": my_config['window_size'] + 1,
            },
            "sampling": {
                "max_num_components": 5,
                "covariance_type": "tied",                
                "slice_sampling": True,
            },
        },
        "save_fig": 0,
        "threads": 1,
    }
)

calibration.run()

import matplotlib.pylab as plt
from grainlearning.tools import plot_posterior, plot_pdf

plot_posterior('test', param_names, calibration.system.param_data[calibration.ids_surrogate],
               calibration.inference.Bayes_filter.posteriors[:, calibration.ids_surrogate])
plot_posterior('test', param_names, calibration.system.param_data[calibration.ids_origin],
               calibration.inference.Bayes_filter.posteriors[:, calibration.ids_origin])
plt.show()

plot_pdf('test', param_names, [calibration.system.param_data[calibration.ids_surrogate],
                               calibration.system.param_data[calibration.ids_origin]])
plt.show()
