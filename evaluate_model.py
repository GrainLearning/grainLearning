import os
import h5py
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

from preprocessing import prepare_datasets
from windows import predict_over_windows


PRESSURE = 'All'
EXPERIMENT_TYPE = 'All'
MODEL_NAME = 'simple_rnn'
PLOT_DIR = 'plots/'
DATA_DIR = 'data/rnn_data.hdf5'

def plot_losses(losses):
    epoch_list = list(range(len(losses['train'])))
    fig, ax = plt.subplots()

    for split in ['train', 'val']:
        ax.plot(epoch_list, losses[split], label=split + 'loss')
    fig.legend()

    fig.savefig(PLOT_DIR + f'loss_{PRESSURE}_{EXPERIMENT_TYPE}.png')

def plot_predictions(split_data, model, train_stats):
    test_inputs = split_data['test'][0][:32]
    labels = split_data['test'][1][:32]

    window_size = train_stats['window_size']
    sequence_length = train_stats['sequence_length']
    predictions = predict_over_windows(
            test_inputs, model, window_size, sequence_length)

    steps = np.array(list(range(sequence_length)))
    steps_predicted = np.array(list(range(window_size, sequence_length)))

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    # e, f_0, a_c, a_n, a_t
    ids = {'e': 0, 'f_0': 3, 'a_c': 4, 'a_n': 5, 'a_t': 6, 'p': 1, 'q': 2}

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

    for i_s, color in enumerate(['blue', 'green']):
        _plot_sequence(0, 0, 'e', i_s=i_s, color=color)
        _plot_sequence(0, 1, 'f_0', i_s=i_s, color=color)
        _plot_sequence(0, 2, 'p', x_key='q', i_s=i_s, color=color)
        _plot_sequence(1, 0, 'a_c', i_s=i_s, color=color)
        _plot_sequence(1, 1, 'a_n', i_s=i_s, color=color)
        _plot_sequence(1, 2, 'a_t', i_s=i_s, color=color)

        def extract_combination(data, i_s=0):
            q = data[i_s, :, ids['q']]
            p = data[i_s, :, ids['p']]
            a_c = data[i_s, :, ids['a_c']]
            a_n = data[i_s, :, ids['a_n']]
            a_t = data[i_s, :, ids['a_t']]
            comb = q / p - 2 / 5 * (a_c + a_n + 3 / 2 * a_t)
            return comb

        fill_ax(ax[2, 0], steps, extract_combination(labels, i_s=i_s), steps_predicted,
                extract_combination(predictions, i_s=i_s), y_label='vanishing combination', color=color)

    fig.suptitle(f'pressure {PRESSURE}, type {EXPERIMENT_TYPE}')
    fig.savefig(PLOT_DIR + f'predictions_{MODEL_NAME}_{PRESSURE}_{EXPERIMENT_TYPE}.png')

def fill_ax(ax, x_labels, y_labels, x_preds, y_preds,
        title='', x_label='', y_label='', color='blue'):
    ax.plot(x_labels, y_labels, label='truth', color=color)
    ax.plot(x_preds, y_preds, label='predictions', linestyle='dashed', color=color)
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def _plot_features(ax, steps, steps_predicted, labels, test_predictions,
        ax_i, ax_j, i_f_x, i_f_y):
    ax[ax_i, ax_j].plot(steps, label='truth')


def main():
    datafile = h5py.File(DATA_DIR, 'r')
    output_labels = datafile.attrs['outputs']

    model_directory = f'trained_models/{MODEL_NAME}_{PRESSURE}_{EXPERIMENT_TYPE}'
    model = keras.models.load_model(model_directory)
    train_stats = np.load(model_directory + '/train_stats.npy', allow_pickle=True).item()
    losses = np.load(model_directory + '/losses.npy', allow_pickle=True).item()

    split_data, _ = prepare_datasets(
            pressure=PRESSURE,
            experiment_type=EXPERIMENT_TYPE,
            )

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    plot_losses(losses)
    plot_predictions(split_data, model, train_stats)

if __name__ == '__main__':
    main()

