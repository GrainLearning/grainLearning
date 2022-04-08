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
from models import baseline_model, baseline_model_seq
from windows import sliding_windows

pressure = 'All'  # '0.2e6'
experiment_type = 'All'  # 'drained'
use_windows = True

split_data, train_stats = prepare_datasets(
        raw_data='data/sequences.hdf5',
        pressure=pressure,
        experiment_type=experiment_type,
        )
_, sequence_length, num_features = split_data['train'][0].shape
num_labels = split_data['train'][1].shape[-1]

if not use_windows:
    final_data = split_data
else:
    window_size, window_step = 20, 1
    windows = {split: sliding_windows(
                 split_data[split], window_size, window_step)
                for split in ['train', 'val', 'test']}

    train_stats['window_size'] = window_size
    train_stats['window_step'] = window_step
    train_stats['sequence_length'] = sequence_length

    final_data = windows


model =  baseline_model(num_features, num_labels)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

epochs = 2_000 # Ma: 2_000
batch_size = 256

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True)

history = model.fit(
        *final_data['train'],
        epochs=epochs,
        validation_data=final_data['val'],
        batch_size=batch_size,
        callbacks=[early_stopping],
        )

model_directory = f'trained_models/simple_rnn_{pressure}_{experiment_type}_2'
model.save(model_directory)
np.save(model_directory + '/train_stats.npy', train_stats)

losses = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
        }
np.save(model_directory + '/losses.npy', losses)

