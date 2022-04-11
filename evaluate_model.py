import os
import h5py
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

from preprocessing import prepare_datasets
from windows import predict_over_windows

PRESSURES = ['0.2e6', '0.5e6', '1.0e6']
EXPERIMENT_TYPES = ['drained', 'undrained']
P_INDEX = -2
E_INDEX = -1

PRESSURE = 'All'
EXPERIMENT_TYPE = 'All'
MODEL_NAME = 'simple_rnn'
PLOT_DIR = 'plots/'
DATA_DIR = 'data/sequences.hdf5'

use_windows = True

def plot_losses(losses):
    epoch_list = list(range(len(losses['train'])))
    fig, ax = plt.subplots()

    for split in ['train', 'val']:
        ax.plot(epoch_list, losses[split], label=split + 'loss')
    ax.set_yscale('log')
    fig.legend()

    fig.savefig(PLOT_DIR + f'loss_{PRESSURE}_{EXPERIMENT_TYPE}.png')

def plot_predictions(split_data, model, train_stats):
    test_inputs = split_data['test'][0][:32]
    labels = split_data['test'][1][:32]

    window_size = train_stats['window_size']
    raw_sequence_length = train_stats['sequence_length']
    sequence_length = raw_sequence_length - window_size

    labels = labels[:, -sequence_length:]
    if use_windows:
        predictions = predict_over_windows(
                test_inputs, model, window_size, raw_sequence_length,
                predict_initial=False)
    else:
        predictions = model.predict(test_inputs)

    steps = np.array(list(range(sequence_length)))
    num_predicted = predictions.shape[1]
    steps_predicted = list(range(sequence_length - num_predicted, sequence_length))
    steps_predicted = steps

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    # e, f_0, a_c, a_n, a_t
    ids = {'e': 0, 'f_0': 3, 'a_c': 4, 'a_n': 5, 'a_t': 6, 'p': 1, 'q': 2}
    representative_idxs = _find_representatives(split_data)

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

    def extract_combination_inv(data, i_s=0):
        """Combination of parameters that is supposed to be zero."""
        q = data[i_s, :, ids['q']]
        p = data[i_s, :, ids['p']]
        a_c = data[i_s, :, ids['a_c']]
        a_n = data[i_s, :, ids['a_n']]
        a_t = data[i_s, :, ids['a_t']]
        comb = q / p - 2 / 5 * (a_c + a_n + 3 / 2 * a_t)
        return comb

    def extract_p_over_q(data, i_s=0):
        q = data[i_s, :, ids['q']]
        p = data[i_s, :, ids['p']]
        return p / q

    ylim = [-3, 3]
    for i_s, color in zip(representative_idxs,
            ['blue', 'green', 'purple', 'darkgreen', 'navy', 'yellowgreen']):
        _plot_sequence(0, 0, 'e', i_s=i_s, color=color)
        _plot_sequence(0, 1, 'f_0', i_s=i_s, color=color)
        fill_ax(ax[0, 2],
                steps, extract_p_over_q(labels, i_s=i_s),
                steps_predicted, extract_p_over_q(predictions, i_s=i_s),
                y_label='p/q', x_label='steps', color=color,
                ylim=ylim)
        _plot_sequence(1, 0, 'a_c', i_s=i_s, color=color)
        _plot_sequence(1, 1, 'a_n', i_s=i_s, color=color)
        _plot_sequence(1, 2, 'a_t', i_s=i_s, color=color)
        _plot_sequence(2, 1, 'p', i_s=i_s, color=color)
        _plot_sequence(2, 2, 'q', i_s=i_s, color=color)

        fill_ax(ax[2, 0],
                steps, extract_combination_inv(labels, i_s=i_s),
                steps_predicted, extract_combination_inv(predictions, i_s=i_s),
                y_label='vanishing combination', x_label='steps', color=color,
                ylim=ylim)

    fig.suptitle(f'pressure {PRESSURE}, type {EXPERIMENT_TYPE}')
    fig.savefig(PLOT_DIR + f'predictions_{MODEL_NAME}_{PRESSURE}_{EXPERIMENT_TYPE}.png')

def fill_ax(ax, x_labels, y_labels, x_preds, y_preds,
        title='', x_label='', y_label='', color='blue', ylim=None):
    ax.plot(x_labels, y_labels, label='truth', color=color)
    ax.plot(x_preds, y_preds, label='predictions', linestyle='dashed', color=color)
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if ylim:
        ax.set_ylim(ylim)

def _plot_features(ax, steps, steps_predicted, labels, test_predictions,
        ax_i, ax_j, i_f_x, i_f_y):
    ax[ax_i, ax_j].plot(steps, label='truth')

def _find_representatives(data):
    """
    returns a list of indices indicating samples each combination of pressure
    and experiment type.
    """
    representatives = []
    for pressure in PRESSURES:
        for experiment_type in EXPERIMENT_TYPES:
            p = float(pressure[:3])
            e = 1 if experiment_type == 'drained' else 0
            i = 0
            sample = data['test'][0][i:i+1]
            while not (sample[0, 0, P_INDEX] == p and sample[0, 0, E_INDEX] == e):
                i += 1
                sample = data['test'][0][i:i+1]
            representatives.append(i)

    return representatives


def main():
    datafile = h5py.File(DATA_DIR, 'r')
    output_labels = datafile.attrs['outputs']

    model_directory = f'trained_models/{MODEL_NAME}_{PRESSURE}_{EXPERIMENT_TYPE}_3'
    model = keras.models.load_model(model_directory)
    train_stats = np.load(model_directory + '/train_stats.npy', allow_pickle=True).item()
    losses = np.load(model_directory + '/losses.npy', allow_pickle=True).item()

    split_data, _ = prepare_datasets(
            raw_data=DATA_DIR,
            pressure=PRESSURE,
            experiment_type=EXPERIMENT_TYPE,
            pad_length=train_stats['window_size']
            )

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    plot_losses(losses)
    plot_predictions(split_data, model, train_stats)

if __name__ == '__main__':
    main()

