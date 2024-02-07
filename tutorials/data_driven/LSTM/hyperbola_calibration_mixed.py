import grainlearning.rnn.train as train_rnn
import grainlearning.rnn.predict as predict_rnn
from grainlearning.rnn import preprocessor
import numpy as np
from grainlearning import BayesianCalibration
from matplotlib import pyplot as plt

x_obs = np.arange(100)
# hyperbola in a form similar to the Duncan-Chang material model, q = \eps / (a * 100 + b * \eps)
y_obs = x_obs / (0.2 * 100 + 5.0 * x_obs)


def nonlinear(x, params):
    return x / (params[0] * 100 + params[1] * x)


# Define the configuration dictionary for the ML surrogate
my_config = {
    'input_data': None,
    'param_data': None,
    'output_data': None,
    'train_frac': 0.7,
    'val_frac': 0.2,
    'window_size': 10,
    'window_step': 1,
    'patience': 25,
    'epochs': 20,
    'learning_rate': 1e-4,
    'lstm_units': 128,
    'dense_units': 128,
    'batch_size': 64,
    'standardize_outputs': True,
    'save_weights_only': True
}


def run_sim_original(x, params):
    """Run different realizations of the original model.

    :param x: the input sequence
    :param params: the parameters
    """
    data = []
    for params in params:
        # Run the model
        y = nonlinear(x, params)
        data.append(np.array(y, ndmin=2))
    return np.array(data)


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
    model, train_stats, config = predict_rnn.get_pretrained_model('outputs')

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
        sim_data = run_sim_original(calib.system.ctrl_data, calib.system.param_data)
        calib.system.set_sim_data(sim_data)
        my_config['input_data'] = calib.system.ctrl_data
        my_config['param_data'] = calib.system.param_data
        my_config['output_data'] = calib.system.sim_data
    else:
        # split samples into two subsets to be used with the original function and the ML surrogate
        np.random.seed()
        ids = np.random.permutation(len(calib.system.param_data))
        split_index = int(len(ids) * 0.5)
        ids_origin, ids_surrogate = ids[:split_index], ids[split_index:]

        # run the original function for the first half of the samples
        param_data_origin = calib.system.param_data[ids_origin]
        sim_data_origin = run_sim_original(calib.system.ctrl_data, param_data_origin)

        # run the surrogate for the second half of the samples
        param_data_surrogate = calib.system.param_data[ids_surrogate]
        sim_data_surrogate = run_sim_surrogate(param_data_origin, sim_data_origin, param_data_surrogate)

        # put the two subsets of simulation data together according to the original order
        sim_data = np.zeros([calib.system.num_samples, calib.system.num_obs, calib.system.num_steps])
        sim_data[ids_surrogate] = sim_data_surrogate
        sim_data[ids_origin] = sim_data_origin

        # set `sim_data` to system
        calib.system.set_sim_data(sim_data)


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 5,
        "callback": run_sim_mixed,
        "system": {
            "param_min": [0.1, 0.1],
            "param_max": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "sim_name": 'hyperbola',
            "sigma_tol": 0.01,
        },
        "calibration": {
            "inference": {
                "ess_target": 0.3,
                "scale_cov_with_max": True,
            },
            "sampling": {
                "max_num_components": 1,
                "n_init": 1,
                "random_state": 0,
                "slice_sampling": True,
            },
            "initial_sampling": "halton",
        },
        "save_fig": 0,
    }
)

calibration.run()
