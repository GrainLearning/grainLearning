"""
Script to train a model to predict macroscopic features of a DEM simulation.
"""
import os, shutil

import numpy as np
from pathlib import Path
import tensorflow as tf
import wandb

from .models import rnn_model
from .preprocessing import prepare_datasets

def train(config=None):
    """
    Train a model and report to weights and biases.

    :param config: dictionary containing model and training configurations.

    If called in the framewrok of a sweep:
    A sweep can be created from the command line using a configuration file,
    for example `example_sweep.yaml`, as: ``wandb sweep example_sweep.yaml``
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
                save_model=True,
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

def train_without_wandb(config=None):
    """
    Train a model locally: no report to wandb.
    Saves either the model or its weight to folder outputs.

    :param config: dictionary containing taining hyperparameters and some model parameters
    """
    path_save_data = Path('outputs')
    if os.path.exists(path_save_data):
        delete_outputs = input(f"The contents of {path_save_data} will be permanently deleted,\
                                 do you want to proceed? [y/n]: ")
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

def get_default_dict():
    """Returns a dictionary with default values for the configuration of
    RNN model and training procedure. Possible fields are:

    * RNN model
        - 'raw_data': Path to hdf5 file generated using parse_data_YADE.py
        - 'pressure' and 'experiment_type': Name of the subfield of dataset to consider. It can also be 'All'.
        - 'conditional':
         - True: Create a conditional RNN. The contact parameters vector doesn't have
           to be copied multiple times into the strain sequence.
         - False: Concatenate copies of contact_params to each one of the inputs steps.
        - 'use_windows': At the moment the model only works if windows are considered.
        - 'window_size': int, number of steps composing a window.

    * Training procedure
        - 'patience': patience of `tf.keras.callbacks.EarlyStopping`.
        - 'epochs': Maximum number of epochs.
        - 'learning_rate': double, learning_rate of `tf.keras.optimizers.Adam`.
        - 'batch_size': Size of the data batches per training step.
        - 'standardize_outputs': If True transform the data labels to have zero mean and unit variance.
          Also, in train_stats the mean and variance of each label will be stored,
          so that can be used to transform predicitons.
          (This is very usful if the labels are not between [0,1])
        - 'add_e0': Whether to add the initial void ratio (output) as a contact parameter.
        - 'save_weights_only':
         * True: Only the weights will be saved.
         * False: The whole model will be saved **(Recommended)**.

    """
    defaults = {
        'raw_data': 'data/sequences.hdf5',
        'pressure': 'All',
        'experiment_type': 'All',
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
        'save_weights_only': False,
    }
    return defaults
