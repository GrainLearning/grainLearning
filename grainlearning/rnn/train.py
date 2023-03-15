"""
Train a model to predict macroscopic features of a DEM simulation.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb

from grainlearning.rnn.models import rnn_model
from grainlearning.rnn.preprocessing import prepare_datasets
from grainlearning.rnn.windows import windowize_single_dataset


def train(config=None):
    """
    Train a model and report to weights and biases.

    If called in the framework of a sweep:
    A sweep can be created from the command line using a configuration file,
    for example `example_sweep.yaml`, as: ``wandb sweep example_sweep.yaml``
    And run with the line shown subsequently in the terminal.
    The config is loaded from the yaml file.

    :param config: dictionary containing model and training configurations.

    :return: Same as ``tf.keras.Model.fit()``: A History object.
      Its History.history attribute is a record of training loss values and
      metrics values at successive epochs, as well as
      validation loss values and validation metrics values.
    """
    with wandb.init(config=config):
        config = wandb.config
        config = _check_config(config)
        config_optimizer = _get_optimizer_config(config)

        # preprocess data
        split_data, train_stats = prepare_datasets(**config)
        np.save(os.path.join(wandb.run.dir, 'train_stats.npy'), train_stats)

        # set up the model
        model = rnn_model(train_stats, **config)
        optimizer = tf.keras.optimizers.Adam(**config_optimizer)
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

        # Evaluate in test dataset and log to wandb the metrics
        test_data = windowize_single_dataset(split_data['test'], **config)
        test_loss, test_mae = model.evaluate(test_data.batch(config.batch_size))
        print(f"test loss = {test_loss}, test mae = {test_mae}")
        wandb.log({'test_loss': test_loss, 'test_mae': test_mae})

        return history

def train_without_wandb(config=None):
    """
    Train a model locally: no report to wandb.
    Saves either the model or its weight to folder outputs.

    :param config: dictionary containing taining hyperparameters and some model parameters

    :return: Same as ``tf.keras.Model.fit()``: A History object.
      Its History.history attribute is a record of training loss values and
      metrics values at successive epochs, as well as
      validation loss values and validation metrics values.
    """
    config = _check_config(config)
    config_optimizer = _get_optimizer_config(config)
    path_save_data = Path('outputs')
    if os.path.exists(path_save_data):
        delete_outputs = input(f"The contents of {path_save_data} will be permanently deleted,\
                                 do you want to proceed? [y/n]: ")
        if delete_outputs == "y": shutil.rmtree(path_save_data)
        else:
            raise SystemExit("Cancelling training")

    os.mkdir(path_save_data)

    # preprocess data
    split_data, train_stats = prepare_datasets(**config)
    np.save(path_save_data/'train_stats.npy', train_stats)
    np.save(path_save_data/'config.npy', config)

    # set up the model
    model = rnn_model(train_stats, **config)
    optimizer = tf.keras.optimizers.Adam(**config_optimizer)
    model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
        )

    # create batches
    for split in ['train', 'val']:  # do not batch test set
        split_data[split] = split_data[split].batch(config['batch_size'])

    # set up training
    if config['save_weights_only'] : path_save_data = path_save_data/"weights.h5"
    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
        )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                path_save_data,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=config['save_weights_only']
            )

    # train
    history = model.fit(
            split_data['train'],
            epochs=config['epochs'],
            validation_data=split_data['val'],
            callbacks=[early_stopping, checkpoint],
        )

    # Evaluate in test dataset and print the metrics
    test_data = windowize_single_dataset(split_data['test'], **config)
    test_loss, test_mae = model.evaluate(test_data.batch(config['batch_size']))
    print(f"test loss = {test_loss}, test mae = {test_mae}")

    return history


def get_default_dict():
    """
    Returns a dictionary with default values for the configuration of data preparation,
    RNN model and training procedure. Possible fields are:

    * Data preparation

      * `'raw_data'`: Path to hdf5 file generated using parse_data_YADE.py
      * `'pressure'` and `'experiment_type'`: Name of the subfield of dataset to consider. It can also be 'All'.
      * `'standardize_outputs'`: If True transform the data labels to have zero mean and unit variance.
        Also, in train_stats the mean and variance of each label will be stored,
        so that can be used to transform predicitons.
        (This is very usful if the labels are not between [0,1])
      * `'add_e0'`: Whether to add the initial void ratio (output) as a contact parameter.
      * `'add_pressure'`: Wether to add the pressure to contact_parameters.
      * `'add_experiment_type'`: Wether to add the experiment type to contact_parameters.
      * `'train_frac'`: Fraction of the data used for training, between [0,1].
      * `'val_frac'`: Fraction of the data used for validation, between [0,1].
        The fraction of the data used for test is then ``1 - train_frac - val_frac``.

    * RNN model

      * `'window_size'`: int, number of steps composing a window.
      * `'window_step'`: int, number of steps between consecutive windows (default = 1).
      * `'pad_length'`: int, equals to ``window_size``. Length of the sequence that with be pad at the start.
      * `'lstm_units'`: int, number of neurons or units in LSTM layer.
      * `'dense_units'`: int, number of neurons or units of dense layer.

    * Training procedure

      * `'patience'`: patience of `tf.keras.callbacks.EarlyStopping`.
      * `'epochs'`: Maximum number of epochs.
      * `'learning_rate'`: double, learning_rate of `tf.keras.optimizers.Adam`.
      * `'batch_size'`: Size of the data batches per training step.
      * `'save_weights_only'`:

        * True: Only the weights will be saved.
        * False: The whole model will be saved **(Recommended)**.


    :return: Dictionary containing default values of the arguments that the user can set.
    """
    return {
        'raw_data': 'data/sequences.hdf5',
        'pressure': 'All',
        'experiment_type': 'All',
        'add_e0': False,
        'add_pressure': True,
        'add_experiment_type': True,
        'train_frac': 0.7,
        'val_frac': 0.15,
        'window_size': 10,
        'window_step': 1,
        'pad_length': 0,
        'lstm_units': 200,
        'dense_units': 200,
        'patience': 5,
        'epochs': 100,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'standardize_outputs': True,
        'save_weights_only': False
    }


def _check_config(config):
    """
    Checks that values requiring an input from the user would be specified in config.

    :param config: Dictionary containing the values of different arguments.

    :return: Updated config dictionary.
    """
    # Necessary keys
    if 'raw_data' not in config.keys(): raise ValueError("raw_data has not been defined in config")
    # Note: I systematically use config.keys() instead of in config, because config can be a dict from wandb
    # This object behaves differently than python dict (might be jsut the version), but this solves it.

    # Warning that defaults would be used if not defined.
    # Adding the default to config because is required in other functions.
    keys_to_check = ['window_size', 'save_weights_only', 'batch_size', 'epochs', 'learning_rate', 'patience']
    defaults = get_default_dict()
    for key in keys_to_check:
        config = _warning_config_field(key, config, defaults[key], add_default_to_config=True)

    # Warning that defaults would be used if not defined
    keys_to_check = ['pressure', 'experiment_type', 'standardize_outputs', 'add_e0',
                     'pad_length', 'train_frac', 'val_frac', 'add_pressure', 'add_experiment_type']
    for key in keys_to_check:
        _warning_config_field(key, config, defaults[key])

    # Warning for an unexpected key value
    config_optimizer = _get_optimizer_config(config)
    for key in config.keys():
        if key not in defaults and key not in config_optimizer:
            warnings.warn(f"Unexpected key in config: {key}. Allowed keys are {defaults.keys()}.")

    return config


def _warning_config_field(key, config, default, add_default_to_config=False):
    """
    Raises a warning if key is not included in config dictionary.
    Also informs the default value that will be used.
    If add_default_to_config=True, then it adds the key and its default value to config.
    """
    # customized warning to print -only- the warning message
    def _custom_format_warning(msg, *_):
        return str(msg) + '\n' # ignore everything except the message

    warnings.formatwarning = _custom_format_warning

    if key not in config:
        if add_default_to_config: config[key] = default
        warnings.warn(f"No {key} specified in config, using default {default}.")

    return config


def _get_optimizer_config(config):
    """
    Returns a dictionary with the keys and values of the intersection
    between config and possible parameters of the optimizer.
    :param config: Dictionary containing the values of different arguments.
    """
    config_optimizer = {}
    keys_optimizer = tf.keras.optimizers.Adam.__init__.__code__.co_varnames
    for key in config.keys():
        if key in keys_optimizer and key not in ('self', 'kwargs', 'name'):
            config_optimizer[key] = config[key]

    return config_optimizer
