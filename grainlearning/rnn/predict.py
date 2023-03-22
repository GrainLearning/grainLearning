"""
Module containing functions to load a trained RNN model and make a prediction.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import wandb
import yaml

from grainlearning.rnn.models import rnn_model
from grainlearning.rnn.preprocessing import prepare_datasets
from grainlearning.rnn.windows import predict_over_windows


NAME_MODEL_H5 = 'model-best.h5'
NAME_TRAIN_STATS = 'train_stats.npy'


def get_best_run_from_sweep(entity_project_sweep_id: str):
    """
    Load the best performing model found with a weights and biases sweep.
    Also load the data splits it was trained on.

    :param entity_project_sweep_id: string of form <user>/<project>/<sweep_id>

    :return:
        - model: The trained model with the lowest validation loss.
        - data: The train, val, test splits as used during training.
        - stats: Some statistics on the training set and configuration.
        - config: The configuration dictionary used to train the model.
    """
    sweep = wandb.Api().sweep(entity_project_sweep_id)
    best_run = sweep.best_run()
    best_model = wandb.restore(
            NAME_MODEL_H5,  # this saves the model locally under this name
            run_path=entity_project_sweep_id + best_run.id,
            replace=True,
        )
    config = best_run.config
    if os.path.exists(Path(best_model.dir)/NAME_TRAIN_STATS):
        train_stats = np.load(Path(best_model.dir)/NAME_TRAIN_STATS, allow_pickle=True).item()
        data, _ = prepare_datasets(**config)
    else:
        data, train_stats = prepare_datasets(**config)

    model = rnn_model(train_stats, **config)
    model.load_weights(best_model.name)
    return model, data, train_stats, config


def get_pretrained_model(path_to_model: str):
    """
    Loads configuration, training statistics and model of a pretrained model.

    Reads train_stats, and creates dataset.

    :param path_to_model: str or pathlib.Path to the folder where is stored.

    :return:
        - model: keras model ready to use.
        - train_stats: Array containing the values used to standardize the data (if config.standardize_outputs = True),
          and lenghts of sequences, load_features, contact_params, labels, window_size and window_step.
        - config: dictionary with the model configuration
    """
    path_to_model = Path(path_to_model)

    # Load config
    config = load_config(path_to_model)

    # Load train_stats
    if os.path.exists(path_to_model / NAME_TRAIN_STATS):
        train_stats = np.load(path_to_model / NAME_TRAIN_STATS, allow_pickle=True).item()
    else: raise FileNotFoundError('train_stats.npy was not found')

    # Load model
    model = load_model(path_to_model, train_stats, config)

    return model, train_stats, config


def load_config(path_to_model: Path):
    """
    Searches for the configuration (of the model training) file in 'path_to_model'.
    Read config.yaml into a python dictionary equivalent to config.
    config.yaml contains information about hyperparameters and model parameters, is generated in every run of wandb.
    Raises FileNotFoundError if there are not files matching possible formats.

    :param path_to_model: str or pathlib.Path to the folder where is stored.

    :return: Dictionary config.
    """
    path_to_model = Path(path_to_model)

    # Possible yaml names
    config_yaml = ['config.yaml', 'config.yml']
    if os.path.exists(path_to_model / config_yaml[0]) or os.path.exists(path_to_model / config_yaml[1]): # Model has been trained using wandb

        if os.path.exists(path_to_model / config_yaml[0]): yaml_file = open(path_to_model / config_yaml[0], 'r', encoding="utf-8")
        else: yaml_file = open(path_to_model / config_yaml[1], 'r', encoding="utf-8")

        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()

        del config['wandb_version']; del config['_wandb']
        for key in config.keys():
            del config[key]['desc']
            config[key] = config[key]['value']

    elif os.path.exists(path_to_model / 'config.npy'): # Model has been trained without wandb and config saved as .npy
        config = np.load(path_to_model / 'config.npy', allow_pickle=True).item()
    elif os.path.exists(path_to_model / 'config.h5'): # Model has been trained without wandb and config saved as .h5
        config = h5py.File(path_to_model / 'config.h5', 'r')
    else: raise FileNotFoundError('config was not found we tried formats (.yaml, .yml, .npy, .h5)')

    return config


def load_model(path_to_model: Path, train_stats: dict, config: dict):
    """
    Searches for the file containing the saved model in 'path_to_model'.
    Raises FileNotFoundError if there are not files matching possible formats.

    :param path_to_model: str or pathlib.Path to the folder where is stored.
    :param train_stats: Dictionary containing different dimensions of the data used to trained and standardization values.
      Its contents are saved in 'path_to_model/train_stats.npy'.
    :param config: Dictionary containing the configuration of the training for such model.

    :return: Keras model.
    """
    path_to_model = Path(path_to_model)
    if os.path.exists(path_to_model / NAME_MODEL_H5): # Model has been trained using wandb
        try:
            model = tf.keras.models.load_model(path_to_model / NAME_MODEL_H5) # whole model was saved
        except ValueError:
            model = rnn_model(train_stats, **config)
            model.load_weights(path_to_model / NAME_MODEL_H5) # only weights were saved

    elif os.path.exists(path_to_model / 'saved_model.pb'): # Model has been saved directly using tf.keras
        model = tf.keras.models.load_model(path_to_model)
    elif os.path.exists(path_to_model / 'weights.h5'): # Model's weights have been saved directly using tf.keras
        model = rnn_model(train_stats, **config)
        model.load_weights(path_to_model / 'weights.h5')
    else: raise FileNotFoundError("Could not find a model to load")

    return model


def predict_macroscopics(
        model: tf.keras.Model,
        data: tf.data.Dataset,
        train_stats: dict,
        config: dict,
        batch_size: int = 256
        ):
    """
    Use the given model to predict the features of the given data.
    If 'standardize_outputs' in config, rescale the predictions to their original units.

    :param model: Keras RNN model
    :param data: Tensorflow dataset containing inputs: 'load_sequence' and 'contact_parameters', and outputs.
    :param train_stats: Dictionary containing statistics of the training set.
    :param config: Dictionary containing the configuration with which the model was trained.
    :param batch_size: Size of batches to use.

    :return: predictions: tf.Tensor containing the predictions in original units.
    """
    data = data.batch(batch_size)

    # Check that input sizes of data correspond to those of the pre-trained model
    inputs, _ = next(iter(data))
    if inputs['load_sequence'].shape[2] != train_stats['num_load_features']:
        raise ValueError(f"Number of elements in load_sequence of data does not match the model load_sequence shape. \
            Got {inputs['load_sequence'].shape[2]}, expected {train_stats['num_load_features']}.")

    if inputs['contact_parameters'].shape[1] != train_stats['num_contact_params']:
        raise ValueError(f"Number of elements in contact_parameters of data does not match the model \
            contact_parameters shape. \
            Got {inputs['contact_parameters'].shape[2]}, expected {train_stats['num_contact_params']}.")

    if inputs['load_sequence'].shape[1] != train_stats['sequence_length']:
        raise ValueError(f"Sequence length of the train_stats {train_stats['sequence_length']} does not match \
            that of the data {inputs['load_sequence'].shape[1]}. If the train_stats are does of the model, \
            check pad_length. Can be that the trained model is not compatible.")

    predictions = predict_over_windows(data, model, config['window_size'], train_stats['sequence_length'])

    if config['standardize_outputs']:
        mean = tf.cast(train_stats['mean'], tf.float32)
        std = tf.cast(train_stats['std'], tf.float32)
        predictions = predictions.map(lambda y: std * y + mean)

    return next(iter(predictions))
