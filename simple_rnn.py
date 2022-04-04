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
from windows import sliding_windows

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

train_stats['window_size'] = window_size
train_stats['window_step'] = window_step
train_stats['sequence_length'] = sequence_length

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

model_directory = f'trained_models/simple_rnn_{pressure}_{experiment_type}'
model.save(model_directory)
np.save(model_directory + '/train_stats.npy', train_stats)

losses = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
        }
np.save(model_directory + '/losses.npy', losses)

