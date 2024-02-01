import grainlearning.rnn.train as train_rnn
import grainlearning.rnn.predict as predict_rnn
from grainlearning.rnn import preprocessor
import numpy as np
from grainlearning import BayesianCalibration
from matplotlib import pyplot as plt
from grainlearning.rnn.evaluate_model import plot_predictions
import tensorflow as tf

x_obs = np.arange(120)
y_obs = 0.2 * x_obs + 5.0


def run_sim(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    data = []
    for params in calib.system.param_data:
        # Run the model
        y_sim = linear(calib.system.ctrl_data, params)
        data.append(np.array(y_sim, ndmin=2))
    calib.system.set_sim_data(data)


def linear(x, params):
    return params[0] * x + params[1]


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 1,
        "callback": run_sim,
        "system": {
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 100,
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "sim_name": 'linear',
            "sigma_tol": 0.01,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
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

data_inputs = ({'inputs': preprocessor.input_data, 'params': calibration.system.param_data}, preprocessor.input_data)
data_inputs = tf.data.Dataset.from_tensor_slices(data_inputs)

predictions = predict_rnn.predict_macroscopics(model, data_inputs, train_stats, config,
                                               batch_size=calibration.system.num_samples)
# converting the predictions to GL format (temporal dimension at the end)
predictions = np.moveaxis(predictions, 1, -1)

# compute the mean square error per sample
error = np.mean((predictions - calibration.system.sim_data[:, :, config['window_size']:]) ** 2, axis=-1)

# plot the error
plt.plot(error)
plt.xlabel("sample")
plt.ylabel("MSE")
