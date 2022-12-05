import os, pathlib
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

from preprocessing import prepare_datasets
from predict import predict_macroscopics

PRESSURES = ['0.2e6', '0.5e6', '1.0e6']
EXPERIMENT_TYPES = ['drained', 'undrained']
P_INDEX = 5
E_INDEX = 6


def plot_predictions(model, data, train_stats, config):
    """
    Take the first sample in the test set for each combination of pressure
    and experiment type, and plot for it the true and predicted macroscopic
    features.

    Args:
        model: Model to perform predictions with.
        data: Tensorflow dataset to predict on.
        train_stats: Dictionary containing training set statistics.
        config: Dictionary containing the configuration with which the model was trained.

    Returns:
        figure
    """
    predictions = predict_macroscopics(model, data, train_stats, config,
            batch_size=256, single_batch=True)
    # extract tensors from dataset
    predictions = next(iter(predictions))
    test_inputs, labels = next(iter(data.batch(256)))

    window_size = train_stats['window_size']
    raw_sequence_length = train_stats['sequence_length']
    sequence_length = raw_sequence_length - window_size

    labels = labels[:, -sequence_length:]

    steps = np.array(list(range(sequence_length)))
    num_predicted = predictions.shape[1]
    steps_predicted = list(range(sequence_length - num_predicted, sequence_length))
    steps_predicted = steps

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    ids = {'e': 0, 'f_0': 3, 'a_c': 4, 'a_n': 5, 'a_t': 6, 'p': 1, 'q': 2}
    representative_idxs = _find_representatives(test_inputs)

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
        """Combine parameters in way that is supposed to be zero."""
        q = data[i_s, :, ids['q']]
        p = data[i_s, :, ids['p']]
        a_c = data[i_s, :, ids['a_c']]
        a_n = data[i_s, :, ids['a_n']]
        a_t = data[i_s, :, ids['a_t']]
        comb = q / p - 2 / 5 * (a_c + a_n + 3 / 2 * a_t)
        return comb

    def extract_q_over_p(data, i_s=0):
        q = data[i_s, :, ids['q']]
        p = data[i_s, :, ids['p']]
        return q / p

    ylim = [-3, 3]
    for i_s, color in zip(representative_idxs,
            ['blue', 'green', 'purple', 'darkgreen', 'navy', 'yellowgreen']):
        _plot_sequence(0, 0, 'e', i_s=i_s, color=color)
        _plot_sequence(0, 1, 'f_0', i_s=i_s, color=color)
        fill_ax(ax[0, 2],
                steps, extract_q_over_p(labels, i_s=i_s),
                steps_predicted, extract_q_over_p(predictions, i_s=i_s),
                y_label='q/p', x_label='steps', color=color,
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

    return fig

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

def _find_representatives(input_data):
    """Return a list of indices indicating samples each combination of pressure and experiment type."""
    representatives = []
    contact_params = input_data['contact_parameters']
    for pressure in PRESSURES:
        for experiment_type in EXPERIMENT_TYPES:
            p = float(pressure[:3])
            e = 1 if experiment_type == 'drained' else 0
            i = 0
            sample = contact_params[i]
            while not (sample[P_INDEX] == p and sample[E_INDEX] == e):
                i += 1
                sample = contact_params[i + 1]
            representatives.append(i)

    return representatives


def main():
    data_dir = pathlib.Path('data/sequences.hdf5')
    plot_dir = pathlib.Path('plots/')

    pressure = 'All'
    experiment_type = 'All'
    model_name = 'simple_rnn'
    saved_model_name = f'{model_name}_{pressure}_{experiment_type}_conditional'
    model_directory = pathlib.Path('trained_models/' + saved_model_name)

    model = keras.models.load_model(model_directory)
    train_stats = np.load(model_directory / 'train_stats.npy', allow_pickle=True).item()
    losses = np.load(model_directory / 'losses.npy', allow_pickle=True).item()

    split_data, _ = prepare_datasets(
            raw_data=data_dir,
            pressure=pressure,
            experiment_type=experiment_type,
            pad_length=train_stats['window_size'],
            use_windows=False,
            add_e0=True,  # was used in the model that is tested with
            )

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    config = {'use_windows': True, 'window_size': train_stats['window_size'],
            'standardize_outputs': True}
    plot_predictions(model, split_data['test'], train_stats, config)


if __name__ == '__main__':
    main()
