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
import os

from preprocessing import prepare_datasets
from models import baseline_model
from windows import sliding_windows, predict_over_windows

pressure = 'All'  # '0.2e6'
experiment_type = 'All'  # 'drained'

split_data, train_stats = prepare_datasets(
        pressure=pressure,
        experiment_type=experiment_type,
        )

window_size, window_step = 20, 5
windows = {split: sliding_windows(
             split_data[split], window_size, window_step)
            for split in ['train', 'val', 'test']}

_, sequence_length, num_features = split_data['train'][0].shape
num_labels = windows['train'][1].shape[-1]

# this is close to Ma's model (apart from the LSTM modification) 
model =  baseline_model(num_features, num_labels)

optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

epochs = 3  # Ma: 2_000
batch_size = 256

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
        *windows['train'],
        epochs=epochs,
        validation_data=windows['val'],
        batch_size=batch_size,
        callbacks=[early_stopping],
        )

model_directory = f'trained_models/simple_rnn_{pressure}_{experiment_type}_tmp'
model.save(model_directory)
np.save(model_directory + '/train_stats.npy', train_stats)

loss = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
        }
fig, ax = plt.subplots()
epoch_list = list(range(len(loss['train'])))

for split in ['train', 'val']:
    ax.plot(epoch_list, loss[split], label=split + 'loss')
fig.legend()

if not os.path.exists('plots'):
    os.makedirs('plots')
fig.savefig('plots/' + f'loss_{pressure}_{experiment_type}.png')

test_inputs = split_data['test'][0][:32]
test_outputs = split_data['test'][1][:32]
test_predictions = predict_over_windows(
        test_inputs, model, window_size, sequence_length)

steps = np.array(list(range(sequence_length)))
steps_predicted = np.array(list(range(window_size, sequence_length)))

# just to get the plot labels
import h5py
datafile = h5py.File('data/rnn_data.hdf5', 'r')

fig, ax = plt.subplots(3, 3)
for feature_idx in range(test_outputs.shape[2]):
    i = feature_idx % 3
    j = feature_idx // 3
    ax[i, j].plot(steps, test_outputs[0, :, feature_idx], label='truth')
    ax[i, j].plot(steps_predicted, test_predictions[0, :, feature_idx], label='predictions')
    ax[i, j].set_title(datafile.attrs['outputs'][feature_idx])
fig.suptitle(f'pressure {pressure}, type {experiment_type}')
fig.savefig('plots/' + f'test_predictions_{pressure}_{experiment_type}.png')

