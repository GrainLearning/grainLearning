"""
Implement the simplest possible model.
Model itself based on Ma's.
Data just picking a single pressure and type.
Contact parameters handled by just concatenating to the sequence data.
No additional hand-engineered features (like Ma's chi).
No rescaling. (do need this for sensible results)
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

epochs = 10  # Ma: 2_000
batch_size = 256
history = model.fit(
        *windows['train'],
        epochs=epochs,
        validation_data=windows['val'],
        batch_size=batch_size,
        )

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

