import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
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

    predictions = predict.predict_batch(model, data, train_stats, config, batch_size=batch_size)
    # extract tensors from dataset
    test_inputs, labels = next(iter(data.batch(batch_size)))

    window_size = config['window_size']
    raw_sequence_length = train_stats['sequence_length']
    sequence_length = raw_sequence_length - window_size
    labels = labels[:, -sequence_length:]

    steps = np.array(list(range(sequence_length)))

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    ids = {'e': 0, 'f_0': 3, 'a_c': 4, 'a_n': 5, 'a_t': 6, 'p': 1, 'q': 2}
    label_combination_inv = "\\frac{q}{p} - \\frac{2}{5} (a_c + a_n + \\frac{3}{2} a_t)"

    add_pressure, add_experiment_type, add_e0 = _checks_extra_contact_params(config)

    if add_pressure and add_experiment_type:
        representative_idxs = _find_representatives(test_inputs, add_e0, add_pressure, add_experiment_type)
    else:
        representative_idxs = _find_random_samples(test_inputs, 5)

    def _plot_sequence(i, j, y_key, i_s=0, x_key='steps', color='blue'):
        if x_key == 'steps':
            x = steps
            x_p = steps
        else:
            x = labels[i_s, :, ids[x_key]]
            x_p = predictions[i_s, :, ids[x_key]]
        y = labels[i_s, :, ids[y_key]]
        y_p = predictions[i_s, :, ids[y_key]]
        fill_ax(ax[i, j], x, y, x_p, y_p, x_label=x_key, y_label=y_key, color=color)

    for i_s, color in zip(representative_idxs,
            ['blue', 'green', 'purple', 'darkgreen', 'navy', 'yellowgreen']):

        p_label, e_label = _get_p_e_labels(config, test_inputs['params'][i_s])

        _plot_sequence(0, 0, 'e', i_s=i_s, color=color)
        _plot_sequence(0, 1, 'f_0', i_s=i_s, color=color)
        fill_ax(ax[0, 2],
                steps, _extract_q_over_p(labels, ids, i_s=i_s),
                steps, _extract_q_over_p(predictions, ids, i_s=i_s),
                y_label='q/p', x_label='steps', color=color)
        _plot_sequence(1, 0, 'a_c', i_s=i_s, color=color)
        _plot_sequence(1, 1, 'a_n', i_s=i_s, color=color)
        _plot_sequence(1, 2, 'a_t', i_s=i_s, color=color)
        _plot_sequence(2, 1, 'p', i_s=i_s, color=color)
        _plot_sequence(2, 2, 'q', i_s=i_s, color=color)
        fill_ax(ax[2, 0],
                steps, _extract_combination_inv(labels, ids, i_s=i_s),
                steps, _extract_combination_inv(predictions, ids, i_s=i_s),
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

def _find_representatives(input_data, add_e0: bool, add_pressure: bool, add_experiment_type: bool):
    """
    Return a list of indices indicating samples each combination of pressure and experiment type.
    """
    if add_pressure and add_experiment_type:
        global P_INDEX, E_INDEX

        representatives = []
        contact_params = input_data['params']
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
    else:
        return _find_random_samples(input_data, 5)


def _find_random_samples(input_data, num_samples):
    return np.random.choice(len(input_data['params']), num_samples, replace=False)

def _checks_extra_contact_params(config: dict):
    """
    Gives the values of add_e0, add_pressure and add_experiment_type, checking if they were indicated in config.
    Otherwise, gives the default values used in preprocessing.
    """
    add_pressure = config['add_pressure'] if 'add_pressure' in config else True
    add_experiment_type = config['add_experiment_type'] if 'add_experiment_type' in config else True
    add_e0 = config['add_e0'] if 'add_e0' in config else False

    return add_pressure, add_experiment_type, add_e0


def _get_p_e_labels(config: dict, contact_params):
    """
    Gives teh label to use for pressure and experiment type for a given dataset.
    """
    add_pressure, add_experiment_type, _ = _checks_extra_contact_params(config)
    if add_pressure: p_label = str(float(contact_params[P_INDEX]))
    else: p_label = config['pressure']
    if add_experiment_type: e_label = 'drained' if contact_params[E_INDEX]==1 else 'undrained'
    else: e_label = config['experiment_type']
    return p_label, e_label


def plot_metric_distribution(data, predictions, config):
    """
    Plot the histogram of the scores.
    """
    test_inputs, labels = next(iter(data.batch(len(data))))
    scores = [mean_absolute_error(labels[i, config['window_size']:, :].numpy(), predictions.numpy()[i, :, :])
              for i in range(labels.numpy().shape[0])]
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Mean Absolute Error')
    ax.set_ylabel('Frequency')
    ax.hist(scores)
    return fig
