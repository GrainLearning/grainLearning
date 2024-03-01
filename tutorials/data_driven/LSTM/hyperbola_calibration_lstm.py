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
            "num_samples": 100,
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

calibration.run()

# 1. Create my dictionary of configuration
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

# 2. Create an object Preprocessor to pre-process my data
preprocessor = preprocessor.PreprocessorLSTM.from_dict(my_config)

# 3. Run the training using bare tensorflow
history_simple = train_rnn.train_without_wandb(preprocessor, config=my_config)

plt.plot(history_simple.history['loss'], label='training loss')
plt.plot(history_simple.history['val_loss'], label='validation loss')
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

# 4. Make predictions with the trained model
model, train_stats, config = predict_rnn.get_pretrained_model('outputs')

data_inputs = preprocessor.prepare_input_data(calibration.system.param_data)

predictions = predict_rnn.predict_batch(model, data_inputs, train_stats, config,
                                        batch_size=calibration.system.num_samples)
# converting the predictions to GL format (temporal dimension at the end)
predictions = np.moveaxis(predictions, 1, -1)

# compute the mean square error per sample
error = np.mean((predictions - calibration.system.sim_data) ** 2, axis=-1)

# plot the error
plt.plot(error)
plt.xlabel("sample")
plt.ylabel("MSE")
plt.show()


# define the callback function using the ML surrogate
def run_sim_surrogate(calib):
    data_inputs = preprocessor.prepare_input_data(calib.system.param_data)
    # make predictions with the trained model
    sim_data = predict_rnn.predict_batch(model, data_inputs, train_stats, config, batch_size=calib.system.num_samples)
    # converting the predictions to GL format (temporal dimension at the end)
    sim_data = np.moveaxis(sim_data, 1, -1)
    # update sim_data in system
    calib.system.set_sim_data(sim_data)


# set the callback function to the one that runs the ML surrogate
calibration.callback = run_sim_surrogate

# continue the calibration with the surrogate
calibration.num_iter = 10
calibration.save_fig = 0
calibration.run()
