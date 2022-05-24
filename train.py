"""
Script to train a model to predict macroscopic features of a DEM simulation.
Tracked using weights and biases.
"""
from tensorflow import keras
import numpy as np
import wandb

from preprocessing import prepare_datasets
from models import rnn_model
from evaluate_model import plot_predictions

def train(config=None):
    """
    Train a model and report to weights and biases.

    A sweep can be created from the command line using a configuration file,
    for example `example_sweep.yaml`, as:
        `wandb sweep example_sweep.yaml`
    And run with the line shown subsequently in the terminal.

    The config is loaded from the yaml file.
    """
    with wandb.init(config=config):
        config = wandb.config

        # preprocess data
        split_data, train_stats = prepare_datasets(**config)

        # set up the model
        model = rnn_model(train_stats, **config)
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae'],
            )

        # create batches
        for split in ['train', 'val']:  # do not batch test set
            split_data[split] = split_data[split].batch(config.batch_size)

        # set up training
        early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.patience,
                restore_best_weights=True,
            )
        wandb_callback = wandb.keras.WandbCallback(
                monitor='val_loss',
                save_model=True,
                save_weights_only=True,
                validation_data=split_data['val'],
            )
        callbacks = [wandb_callback, early_stopping]

        # train
        history = model.fit(
                split_data['train'],
                epochs=config.epochs,
                validation_data=split_data['val'],
                callbacks=callbacks,
            )

        # do some predictions on validation data and save plots to wandb.
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
        'epochs': 1,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'standardize_outputs': True,
        'add_e0': False,
    }

    main(config=defaults)

