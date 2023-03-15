import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot as plt

from grainlearning.rnn import predict


PRESSURES = ['0.2e6', '0.5e6', '1.0e6']
EXPERIMENT_TYPES = ['drained', 'undrained']
P_INDEX = -2 # Pressure and experiment were added at the end of contact_params.
E_INDEX = -1 # These are the indexes to retrieve them


def plot_predictions(model: tf.keras.Model, data: tf.data.Dataset, train_stats: dict, config: dict, batch_size: int = 256):
    """
    Take the first sample in the test set for each combination of pressure
    and experiment type, and plot for it the true and predicted macroscopic
    features.

    :param model: Model to perform predictions with.
    :param data: Tensorflow dataset to predict on.
    :param train_stats: Dictionary containing training set statistics.
    :param config: Dictionary containing the configuration with which the model was trained.

    :return figure:
    """
    # configuration of matplotlib
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['font.family'] = 'sans-serif'

    predictions = predict.predict_macroscopics(model, data, train_stats, config,
                                       batch_size=batch_size, single_batch=True)
    # extract tensors from dataset
    test_inputs, labels = next(iter(data.batch(batch_size)))

    window_size = config['window_size']
    raw_sequence_length = train_stats['sequence_length']
    sequence_length = raw_sequence_length - window_size

    labels = labels[:, -sequence_length:]

    steps = np.array(list(range(sequence_length)))
    num_predicted = predictions.shape[1]
    steps_predicted = list(range(sequence_length - num_predicted, sequence_length))
    steps_predicted = steps

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    ids = {'e': 0, 'f_0': 3, 'a_c': 4, 'a_n': 5, 'a_t': 6, 'p': 1, 'q': 2}
    label_combination_inv = "\\frac{q}{p} - \\frac{2}{5} (a_c + a_n + \\frac{3}{2} a_t)"

    add_e0 = config['add_e0'] if 'add_e0' in config else False
    add_pressure = config['add_pressure'] if 'add_pressure' in config else True
    add_experiment_type = config['add_experiment_type'] if 'add_experiment_type' in config else True

    if add_pressure and add_experiment_type:
        representative_idxs = _find_representatives(test_inputs, add_e0)
    else:
        representative_idxs = _find_random_samples(test_inputs, 5)

    def _plot_sequence(i, j, y_key, i_s=0, x_key='steps', color='blue'):
        if x_key == 'steps':
            x = steps
            x_p = steps_predicted
        else:
            x = labels[i_s, :, ids[x_key]]
            x_p = predictions[i_s, :, ids[x_key]]
        y = labels[i_s, :, ids[y_key]]
        y_p = predictions[i_s, :, ids[y_key]]
        fill_ax(ax[i, j], x, y, x_p, y_p, x_label=x_key, y_label=y_key, color=color)

    for i_s, color in zip(representative_idxs,
            ['blue', 'green', 'purple', 'darkgreen', 'navy', 'yellowgreen']):

        if add_pressure: p_label = str(float(test_inputs['contact_parameters'][i_s][P_INDEX]))
        else: config['pressure']
        if add_experiment_type: e_label = 'drained' if test_inputs['contact_parameters'][i_s][E_INDEX]==1 else 'undrained'
        else: e_label = config['experiment_type']

        _plot_sequence(0, 0, 'e', i_s=i_s, color=color)
        _plot_sequence(0, 1, 'f_0', i_s=i_s, color=color)
        fill_ax(ax[0, 2],
                steps, _extract_q_over_p(labels, ids, i_s=i_s),
                steps_predicted, _extract_q_over_p(predictions, ids, i_s=i_s),
                y_label='q/p', x_label='steps', color=color)
        _plot_sequence(1, 0, 'a_c', i_s=i_s, color=color)
        _plot_sequence(1, 1, 'a_n', i_s=i_s, color=color)
        _plot_sequence(1, 2, 'a_t', i_s=i_s, color=color)
        _plot_sequence(2, 1, 'p', i_s=i_s, color=color)
        _plot_sequence(2, 2, 'q', i_s=i_s, color=color)
        fill_ax(ax[2, 0],
                steps, _extract_combination_inv(labels, ids, i_s=i_s),
                steps_predicted, _extract_combination_inv(predictions, ids, i_s=i_s),
                y_label=label_combination_inv, x_label='steps', color=color,
                add_legend=True, p_label=p_label, e_label=e_label)

    return fig


def fill_ax(ax, x_labels, y_labels, x_preds, y_preds,
            title: str = '', x_label: str = '', y_label: str = '', color: str = 'blue',
            ylim = None, add_legend = False, p_label = "", e_label = ""):
    """
    Configures the plot: data, title and axis label
    """
    ax.plot(x_labels, y_labels, label=f"truth P={p_label}MPa {e_label}", color=color)
    ax.plot(x_preds, y_preds, label="prediction", linestyle='dashed', color=color)
    if title: ax.set_title(title)
    if x_label: ax.set_xlabel(rf'${x_label}$')
    if y_label: ax.set_ylabel(rf'${y_label}$')
    if ylim: ax.set_ylim(ylim)
    if add_legend: ax.legend()


# Auxiliary functions to create plots
def _extract_combination_inv(data, ids, i_s=0):
    """
    Calculate residual (should be zero) of:
    q/p = 2/5 (a_c + a_n + 3/2 a_t)
    """
    q = data[i_s, :, ids['q']]
    p = data[i_s, :, ids['p']]
    a_c = data[i_s, :, ids['a_c']]
    a_n = data[i_s, :, ids['a_n']]
    a_t = data[i_s, :, ids['a_t']]
    comb = q / p - 2 / 5 * (a_c + a_n + 3 / 2 * a_t)
    return comb

def _extract_q_over_p(data, ids, i_s=0):
    q = data[i_s, :, ids['q']]
    p = data[i_s, :, ids['p']]
    return q / p

def _find_representatives(input_data, add_e0):
    """
    Return a list of indices indicating samples each combination of pressure and experiment type.
    """
    global P_INDEX, E_INDEX

    representatives = []
    contact_params = input_data['contact_parameters']
    if add_e0:
        P_INDEX -= 1
        E_INDEX -= 1

    for pressure in PRESSURES:
        for experiment_type in EXPERIMENT_TYPES:
            p = float(pressure[:3]) # better /1e6
            e = 1 if experiment_type == 'drained' else 0
            i = 0
            sample = contact_params[i]
            while not (sample[P_INDEX] == p and sample[E_INDEX] == e) and i < len(contact_params)-1:
                sample = contact_params[i + 1]
                i += 1
            representatives.append(i)
    return representatives

def _find_random_samples(input_data, num_samples):
    return random.sample(range(0, len(input_data['contact_parameters'])), num_samples)
