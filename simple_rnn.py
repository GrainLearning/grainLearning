"""
Implement the simplest possible model.
Model itself based on Ma's.
Data just picking a single pressure and type.
Contact parameters handled by just concatenating to the sequence data.
No additional hand-engineered features (like Ma's chi).
Do rescale input data.
"""
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os

datafile = h5py.File('data/rnn_data.hdf5', 'r')
pressure = '0.2e6'
experiment_type = 'drained'

data = datafile[pressure][experiment_type]
inputs = data['inputs'][:]
outputs = data['outputs'][:]
contact_params = data['contact_params'][:]
num_samples, sequence_length, num_labels = outputs.shape

# concatenate contact parameters to inputs
contact_params_sequence = np.expand_dims(contact_params, axis=1)
contact_params_sequence = np.repeat(contact_params_sequence, sequence_length, 1)
total_inputs = np.concatenate([inputs, contact_params_sequence], axis=2)

num_features = total_inputs.shape[2]

def make_splits(inputs, outputs, fracs):
    """
    Split data into train, val, test based on samples (contact params),
    not within a sequence.
    """
    n_tot = inputs.shape[0]
    n_train = int(fracs['train'] * n_tot)
    n_val = int(fracs['val'] * n_tot)
    split_data = {
            'train': (inputs[:n_train], outputs[:n_train]),
            'val': (inputs[n_train:n_train + n_val], outputs[n_train:n_train + n_val]),
            'test': (inputs[n_train + n_val:], outputs[n_train + n_val:])
            }
    return split_data

split_fracs = {'train': 0.7, 'val': 0.15, 'test': 0.15}
split_data = make_splits(total_inputs, outputs, split_fracs)

def standardize_splits(split_data):
    """
    Standardize all 3 splits, using the mean and std of the training data.
    Do this for the input features and the output labels.
    """
    X_mean = split_data['train'][0].mean(axis=(0, 1)),
    y_mean = split_data['train'][1].mean(axis=(0, 1)),
    train_stats = {
        'X_mean': X_mean,
        'X_std': (split_data['train'][0] - X_mean).std(axis=(0, 1)),
        'y_mean': y_mean,
        'y_std': (split_data['train'][1] - y_mean).std(axis=(0, 1)),
        }

    standardized_splits = dict()
    for split in ['train', 'val', 'test']:
        standardized_splits[split] = standardize(split_data[split], train_stats)


    return train_stats, standardized_splits

def standardize(data, stats):
    X, y = data
    return ((X - stats['X_mean']) / stats['X_std'],
                (y - stats['y_mean']) / stats['y_std'])

train_stats, split_data = standardize_splits(split_data)

def sliding_windows(data, window_size: int, window_step: int):
    """
    Take a dataset of sequences of shape N, S, L and output another dataset
    of shorter sequences of size `window_size`, taken at intervals `window_step`
    so of shape M, window_size, L, with M >> N.
    Also shuffle the data.
    """
    inputs, outputs = data
    num_samples, sequence_length, num_labels = outputs.shape
    Xs, ys = [], []
    start, end = 0, window_size
    while end < sequence_length:
        Xs.append(inputs[:, start:end])
        ys.append(outputs[:, end + 1])
        start += window_step
        end += window_step

    Xs = np.array(Xs)
    ys = np.array(ys)
    # now we have the first dimension for samples and the second for windows,
    # we want to merge those to treat them as independent samples
    num_indep_samples = Xs.shape[0] * Xs.shape[1]
    Xs = np.reshape(Xs, (num_indep_samples,) + Xs.shape[2:])
    ys = np.reshape(ys, (num_indep_samples,) + ys.shape[2:])

    return shuffle(Xs, ys)

def shuffle(Xs, ys):
    inds = np.random.permutation(len(Xs))
    return Xs[inds], ys[inds]

window_size, window_step = 20, 5
windows = {split: sliding_windows(split_data[split], window_size, window_step)
            for split in ['train', 'val', 'test']}

# this is close to Ma's model (apart from the LSTM modification) 
model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.LSTM(50),
        layers.Dense(20),
        layers.Dense(num_labels),
        ])
optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

epochs = 100  # Ma: 2_000
batch_size = 256
history = model.fit(
        *windows['train'],
        epochs=epochs,
        validation_data=windows['val'],
        batch_size=batch_size,
        )

model_directory = f'trained_models/simple_rnn_{pressure}_{experiment_type}'
model.save(model_directory)
np.save(model_directory + '/train_stats.npy', train_stats)

loss = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
        }
fig, ax = plt.subplots()
for split in ['train', 'val']:
    ax.plot(list(range(epochs)), loss[split], label=split + 'loss')
fig.legend()

if not os.path.exists('plots'):
    os.makedirs('plots')
fig.savefig('plots/' + f'loss_{pressure}_{experiment_type}.png')

def predict(inputs, model, window_size):
    """
    Take a batch of sequences, iterate over windows making predictions.
    """
    predictions = []

    start, end = 0, window_size
    while end < sequence_length:
        predictions.append(model(inputs[:, start:end]))
        start += 1
        end += 1

    predictions = np.array(predictions)
    predictions = np.transpose(predictions, (1, 0, 2))

    return predictions

test_inputs = split_data['test'][0][:32]
test_outputs = split_data['test'][1][:32]
test_predictions = predict(test_inputs, model, window_size)

steps = np.array(list(range(sequence_length)))
steps_predicted = np.array(list(range(window_size, sequence_length)))

fig, ax = plt.subplots(3, 3)
for feature_idx in range(test_outputs.shape[2]):
    i = feature_idx % 3
    j = feature_idx // 3
    ax[i, j].plot(steps, test_outputs[0, :, feature_idx], label='truth')
    ax[i, j].plot(steps_predicted, test_predictions[0, :, feature_idx], label='predictions')
    ax[i, j].set_title(datafile.attrs['outputs'][feature_idx])
fig.legend()
fig.savefig('plots/' + f'test_predictions.png')

import IPython ; IPython.embed() ; exit(1)
