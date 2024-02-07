import grainlearning.rnn.train as train_rnn
import grainlearning.rnn.predict as predict_rnn
from grainlearning.rnn import preprocessor
import numpy as np
from grainlearning import BayesianCalibration
from matplotlib import pyplot as plt

x_obs = np.arange(100)
# hyperbola in a form similar to the Duncan-Chang material model, q = \eps / (a * 100 + b * \eps)
y_obs = x_obs / (0.2 * 100 + 5.0 * x_obs)


def run_sim(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    data = []
    for params in calib.system.param_data:
        # Run the model
        y_sim = nonlinear(calib.system.ctrl_data, params)
        data.append(np.array(y_sim, ndmin=2))
    calib.system.set_sim_data(data)


def nonlinear(x, params):
    return x / (params[0] * 100 + params[1] * x)


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 1,
        "callback": run_sim,
        "system": {
            "param_min": [0.1, 1],
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
        "save_fig": -1,
    }
)

# 1. Run one iteration of the calibration to collect the simulation data for training the ML surrogate
calibration.run()

# 2.1. Create my dictionary of configuration
my_config = {
    'input_data': calibration.system.ctrl_data,
    'param_data': calibration.system.param_data,
    'output_data': calibration.system.sim_data,
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

# 2.2. Create an object Preprocessor to pre-process my data
preprocessorLSTM = preprocessor.PreprocessorLSTM.from_dict(my_config)

# 2.3. Run the training using bare tensorflow
history_simple = train_rnn.train_without_wandb(preprocessorLSTM, config=my_config)

# 2.4. Get the trained model
model, train_stats, config = predict_rnn.get_pretrained_model('outputs')


# 3. Define the callback function using the ML surrogate
def run_sim_mixed(calib):
    # split samples into two subsets to be used with the original function and the ML surrogate
    np.random.seed()
    ids = np.random.permutation(len(calib.system.param_data))
    split_index = int(len(ids) * 0.5)
    ids_origin, ids_surrogate = ids[:split_index], ids[split_index:]

    # run the original function for the first half of the samples
    param_data_origin = calib.system.param_data[ids_origin]
    sim_data_origin = []
    for params in param_data_origin:
        # Run the model
        y_sim = nonlinear(calib.system.ctrl_data, params)
        sim_data_origin.append(np.array(y_sim, ndmin=2))
    sim_data_origin = np.array(sim_data_origin)

    # retrain the ML surrogate
    my_config['input_data'] = calib.system.ctrl_data
    my_config['param_data'] = np.vstack([my_config['param_data'], param_data_origin])
    my_config['output_data'] = np.vstack([my_config['output_data'], sim_data_origin])

    preprocessorLSTM = preprocessor.PreprocessorLSTM.from_dict(my_config)
    history_simple = train_rnn.train_without_wandb(preprocessorLSTM, config=my_config)
    model, train_stats, config = predict_rnn.get_pretrained_model('outputs')

    # run the surrogate for the second half of the samples
    param_data_surrogate = calib.system.param_data[ids_surrogate]
    data_inputs = preprocessorLSTM.prepare_input_data(param_data_surrogate)
    # make predictions with the trained model
    sim_data_surrogate = predict_rnn.predict_batch(model, data_inputs, train_stats, config,
                                                   batch_size=param_data_surrogate.shape[0])
    # converting the predictions to GL format (temporal dimension at the end)
    sim_data_surrogate = np.moveaxis(sim_data_surrogate, 1, -1)

    # put the two subsets of simulation data together according to the original order
    sim_data = np.zeros([calib.system.num_samples, calib.system.num_obs, calib.system.num_steps])
    sim_data[ids_surrogate] = sim_data_surrogate
    sim_data[ids_origin] = sim_data_origin

    # set `sim_data` to system
    calib.system.set_sim_data(sim_data)


# set the callback function to the one that runs the ML surrogate
calibration.callback = run_sim_mixed

# continue the calibration with the surrogate
calibration.num_iter = 10
calibration.save_fig = 0
calibration.run()
