"""
Script to train a model to predict macroscopic features of a DEM simulation.

Tracked using weights and biases.
"""
import wandb, os, shutil
import numpy as np
import tensorflow as tf
from pathlib import Path

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
        np.save(os.path.join(wandb.run.dir, 'train_stats.npy'), train_stats)

        # set up the model
        model = rnn_model(train_stats, **config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae'],
            )

        # create batches
        for split in ['train', 'val']:  # do not batch test set
            split_data[split] = split_data[split].batch(config.batch_size)

        # set up training
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.patience,
                restore_best_weights=True,
            )
        wandb_callback = wandb.keras.WandbCallback(
                monitor='val_loss',
                save_model=(not config.save_weights_only),
                save_weights_only=config.save_weights_only,
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
        # do some predictions on test data and save plots to wandb.
        #val_prediction_samples = plot_predictions(model, split_data['test'], train_stats, config)
        #wandb.log({"predictions": val_prediction_samples})

def train_without_wandb(config=None):
    """
    Train a model locally: no report to wandb.
    Saves either the model or its weight to folder outputs.

    :params config: dictionary containing taining hyperparameters and some model parameters
    """
    path_save_data = Path('outputs')
    if os.path.exists(path_save_data):
        delete_outputs = input(f"The contents of {path_save_data} will be permanently deleted, do you want to proceed? [y/n]: ")
        if delete_outputs == "y": shutil.rmtree(path_save_data)
        else:
            print("Cancelling training")
            return
    os.mkdir(path_save_data)

    # preprocess data
    split_data, train_stats = prepare_datasets(**config)
    np.save(path_save_data/'train_stats.npy', train_stats)
    np.save(path_save_data/'config.npy', config)

    # set up the model
    model = rnn_model(train_stats, **config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
        )

    # create batches
    for split in ['train', 'val']:  # do not batch test set
        split_data[split] = split_data[split].batch(config["batch_size"])

    # set up training
    if config["save_weights_only"] : path_save_data = path_save_data/"weights.h5"
    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config["patience"],
            restore_best_weights=True,
        )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                path_save_data,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=config["save_weights_only"]
            )
    callbacks = [early_stopping, checkpoint]

    # train
    history = model.fit(
            split_data['train'],
            epochs=config["epochs"],
            validation_data=split_data['val'],
            callbacks=callbacks,
        )


if __name__ == '__main__':
    defaults = {
        'raw_data': 'data/sequences.hdf5',
        'pressure': 'All',
        'experiment_type': 'All',
        'model': 'conditional',
        'conditional': True,
        'use_windows': True,
        'window_size': 10,
        'window_step': 1,
        'patience': 50,
        'epochs': 2,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'standardize_outputs': True,
        'add_e0': False,
        'save_weights_only':True,
    }

    #train(config=defaults)
    train_without_wandb(config=defaults)

