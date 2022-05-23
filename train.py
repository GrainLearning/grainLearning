"""
Script to train a model to predict macroscopic features of a DEM simulation.
Tracked using weights and biases.
"""
from tensorflow import keras
import numpy as np
import wandb

from preprocessing import prepare_datasets
from models import rnn_model
from windows import windowize_datasets
from evaluate_model import plot_predictions

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        split_data, train_stats = prepare_datasets(**config)

        final_data = windowize_datasets(split_data, train_stats, **config)

        model = rnn_model(
                train_stats['num_load_features'],
                train_stats['num_contact_params'],
                train_stats['num_labels'], **config)

        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=config.patience, restore_best_weights=True)

        wandb_callback = wandb.keras.WandbCallback(
                monitor='val_root_mean_squared_error',
                save_model=True,
                save_weights_only=True,
            ),

        for split in ['train', 'val', 'test']:
            final_data[split] = final_data[split].batch(config.batch_size)

        history = model.fit(
                final_data['train'],
                epochs=config.epochs,
                validation_data=final_data['val'],
                callbacks=[early_stopping, wandb_callback],
                )

        val_prediction_samples = plot_predictions(split_data, model, train_stats)
        wandb.log({"predictions": val_prediction_samples})


def main(config):
    train(config)


if __name__ == '__main__':
    defaults = {
        'raw_data': 'data/sequences.hdf5',
        'pressure': 'All',
        'experiment_type': 'All',
        'model': 'conditional',
        'use_windows': True,
        'window_size': 10,
        'window_step': 1,
        'patience': 50,
        'epochs': 3,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'standardize_outputs': True,
        'add_e0': False,
    }

    main(config=defaults)

