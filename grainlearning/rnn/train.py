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
from grainlearning.rnn.preprocessor import Preprocessor
from grainlearning.rnn.windows import windowize_single_dataset


def train(preprocessor: Preprocessor, config=None, model: tf.keras.Model = None):
    """
    Train a model and report to weights and biases.

    If called in the framework of a sweep:
    A sweep can be created from the command line using a configuration file,
    for example `example_sweep.yaml`, as: ``wandb sweep example_sweep.yaml``
    And run with the line shown subsequently in the terminal.
    The config is loaded from the yaml file.

    :param preprocessor: Preprocessor object to load and prepare the data.
    :param config: dictionary containing model and training configurations.
    :param model: Keras model if ``None`` is passed then an ``rnn_model`` will be created (default).

    :return: Same as ``tf.keras.Model.fit()``: A History object.
      Its History.history attribute is a record of training loss values and
      metrics values at successive epochs, as well as
      validation loss values and validation metrics values.
    """
    with wandb.init(config=config):
        config = wandb.config
        config = _check_config(config, preprocessor)
        config_optimizer = _get_optimizer_config(config)
        preprocessor.run_dir = wandb.run.dir

        # preprocess data
        split_data, train_stats = preprocessor.prepare_datasets()
        np.save(os.path.join(wandb.run.dir, 'train_stats.npy'), train_stats)

        # set up the model
        if model is None:
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


def train_without_wandb(preprocessor: Preprocessor, config=None, model: tf.keras.Model = None):
    """
    Train a model locally: no report to wandb.
    Saves either the model or its weight to folder outputs.

    :param preprocessor: Preprocessor object to load and prepare the data.
    :param config: dictionary containing taining hyperparameters and some model parameters.
    :param model: Keras model if ``None`` is passed then an ``rnn_model`` will be created (default).

    :return: Same as ``tf.keras.Model.fit()``: A History object.
      Its History.history attribute is a record of training loss values and
      metrics values at successive epochs, as well as
      validation loss values and validation metrics values.
    """
    config = _check_config(config, preprocessor)
    config_optimizer = _get_optimizer_config(config)
    path_save_data = Path('outputs')
    if os.path.exists(path_save_data):
        delete_outputs = input(f"The contents of {path_save_data} will be permanently deleted,\
                                 do you want to proceed? [y/n]: ")
        if delete_outputs == "y":
            shutil.rmtree(path_save_data)
        else:
            raise SystemExit("Cancelling training")

    os.mkdir(path_save_data)

    # preprocess data
    split_data, train_stats = preprocessor.prepare_datasets()
    np.save(path_save_data / 'train_stats.npy', train_stats)
    np.save(path_save_data / 'config.npy', config)

    # set up the model
    if model is None:
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
    if config['save_weights_only']: path_save_data = path_save_data / "model.weights.h5"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        restore_best_weights=True,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(path_save_data),
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


def get_default_config():
    """
    Returns a dictionary with default values for the configuration of RNN model
    and training procedure. Possible fields are:

    * RNN model

      * ``'window_size'``: int, number of steps composing a window.
      * ``'window_step'``: int, number of steps between consecutive windows (default = 1).
      * ``'pad_length'``: int, equals to ``window_size``. Length of the sequence that with be pad at the start.
      * ``'lstm_units'``: int, number of neurons or units in LSTM layer.
      * ``'dense_units'``: int, number of neurons or units of dense layer.

    * Training procedure

      * ``'patience'``: patience of `tf.keras.callbacks.EarlyStopping`.
      * ``'epochs'``: Maximum number of epochs.
      * ``'learning_rate'``: double, learning_rate of `tf.keras.optimizers.Adam`.
      * ``'batch_size'``: Size of the data batches per training step.
      * ``'save_weights_only'``: Boolean

        * True: Only the weights will be saved (**Recommended** fro compatibility across platforms).
        * False: The whole model will be saved.


    :return: Dictionary containing default values of the arguments that the user can set.
    """
    return {
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


def _check_config(config: dict, preprocessor: Preprocessor):
    """
    Checks that values requiring an input from the user would be specified in config.

    :param config: Dictionary containing the values of different arguments.

    :return: Updated config dictionary.
    """
    # Note: I systematically use config.keys() instead of in config, because config can be a dict from wandb
    # This object behaves differently than python dict (might be jsut the version), but this solves it.

    # Warning that defaults would be used if not defined.
    # Adding the default to config because is required in other functions.
    keys_to_check = ['window_size', 'save_weights_only', 'batch_size', 'epochs', 'learning_rate', 'patience']
    defaults = get_default_config()
    for key in keys_to_check:
        config = _warning_config_field(key, config, defaults[key], add_default_to_config=True)

    # Warning that defaults would be used if not defined
    keys_to_check = ['pad_length']
    for key in keys_to_check:
        _warning_config_field(key, config, defaults[key])

    # Warning for an unexpected key value
    config_optimizer = _get_optimizer_config(config)
    for key in config.keys():
        if key not in defaults and key not in config_optimizer and key not in preprocessor.get_default_config():
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
        return str(msg) + '\n'  # ignore everything except the message

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


class HyperTuning:
    """
    Class to run hyperparameter tuning with wandb.
    """

    def __init__(self, sweep_config, search_space, other_config=None, entity_name='', project_name='my_sweep'):
        """
        :param sweep_config: Dictionary containing the configuration of the sweep.

            For example:

            .. highlight:: python
            .. code-block:: python

            sweep_config = {
                'method': 'random',
                'name': my_sweep,
                'metric': {'goal': 'minimize', 'name': 'mae'},
                'early_terminate': {
                    'type': 'hyperband',
                    's': 2,
                    'eta': 3,
                    'max_iter': 27
                }
            }

        :param search_space: Dictionary containing the search space of the sweep.

            For example:

            .. highlight:: python
            .. code-block:: python

            search_space = {
                'learning_rate': {
                    'distribution': 'q_log_uniform_values',
                    'q': 1e-4,
                    'min': 1e-4,
                    'max': 1e-2
                },
                'lstm_units': {
                    'distribution': 'q_log_uniform_values',
                    'q': 1,
                    'min': 32,
                    'max': 256
                },
            }

        :param other_config: Dictionary containing other relevant configuration parameters.
        :param project_name: Name of the project in wandb.
        """
        # Set the configuration
        self.sweep_config = sweep_config
        self.search_space = search_space
        self.other_config = other_config
        self.project_name = project_name
        self.entity_name = entity_name
        self.sweep_id = None

    def get_sweep_id(self):
        """
        Returns the sweep_id of a sweep created with the configuration specified in sweep_config and search_space.
        """
        # add default parameters from my_config into sweep_config, use 'values' as the key
        self.sweep_config['parameters'] = {}
        for key, value in self.other_config.items():
            # if key is not listed in the following list which should be excluded
            if key not in ['input_data', 'param_data', 'output_data']:
                self.sweep_config['parameters'].update({key: {'values': [value]}})
        # update sweep_config with the parameters to be searched and their distributions
        self.sweep_config['parameters'].update(self.search_space)
        # create the sweep
        sweep_id = wandb.sweep(self.sweep_config, entity=self.entity_name, project=self.project_name)
        self.sweep_id = sweep_id

    def run_sweep(self, training_func, count=100):
        """
        Function to run hyperparameter tuning with `training_func`
        """
        # Log in to wandb
        wandb.login()
        # Configure the sweep
        self.get_sweep_id()
        # Run an agent
        wandb.agent(self.sweep_id, function=training_func, count=count)
        # Close the wandb session
        wandb.finish()
        # write the data of the HyperTuning object to a npy file
        np.save(f'wandb/{self.entity_name}_{self.project_name}_{self.sweep_id}.npy', self.__dict__)
