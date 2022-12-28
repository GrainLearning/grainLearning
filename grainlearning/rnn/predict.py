"""
Module containing functions to load a trained RNN model and make a prediction.
"""
import yaml, os

import numpy as np
from pathlib import Path
import tensorflow as tf
import wandb

from .models import rnn_model
from .preprocessing import prepare_datasets
from .windows import predict_over_windows

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
            'model-best.h5',  # this saves the model locally under this name
            run_path=entity_project_sweep_id + best_run.id,
            replace=True,
        )
    config = best_run.config
    if os.path.exists(Path(best_model.dir)/'train_stats.npy'):
        train_stats = np.load(Path(path_to_model)/'train_stats.npy', allow_pickle=True).item()
        data, _ = prepare_datasets(**config)
    else:
        data, train_stats = prepare_datasets(**config)

    model = rnn_model(stats, **config)
    model.load_weights(best_model.name)
    return model, data, train_stats, config


def get_pretrained_model(path_to_model: str):
    """
    Loads configuration, training statistics and model of a pretrained model.

    Reads train_stats, and creates dataset.

    :param path_to_model: str or pathlib.Path to the folder where is stored.

    :returns:
        - model: keras model ready to use.
        - train_stats: Array containing the values used to standardize the data (if config.standardize_outputs = True),
          and lenghts of sequences, load_features, contact_params, labels, window_size and window_step.
        - config: dictionary with the model configuration
    """
    # Read config.yaml into a python dictionary equivalent to config.
    # config.yaml contains information about hyperparameters and model parameters, is generated in every run of wandb.
    path_to_model = Path(path_to_model)
    if os.path.exists(path_to_model / 'config.yaml'): # Model has been trained using wandb
        file = open(path_to_model / 'config.yaml', 'r')
        config = yaml.load(file, Loader=yaml.FullLoader)
        del config['wandb_version']; del config['_wandb']
        for key in config.keys():
            del config[key]['desc']
            config[key] = config[key]['value']
    elif os.path.exists(path_to_model / 'config.npy'): # Model has been trained without wandb and config saved as .npy
        config = np.load(path_to_model / 'config.npy', allow_pickle=True).item()
    elif os.path.exists(path_to_model / 'config.h5'): # Model has been trained without wandb and config saved as .h5
        config = h5py.File(path_to_model / 'config.h5', 'r')
    else: raise FileNotFoundError('config was not found we tried formats (.yaml, .npy, .h5)')

    # Load train_stats
    if os.path.exists(path_to_model / 'train_stats.npy'):
        train_stats = np.load(path_to_model / 'train_stats.npy', allow_pickle=True).item()
    else: raise FileNotFoundError('train_stats.npy was not found')

    # Load model
    if os.path.exists(path_to_model / 'model-best.h5'): # Model has been trained using wandb
        try:
            model = tf.keras.models.load_model(path_to_model / 'model-best.h5') # whole model was saved
        except ValueError:
            model = rnn_model(train_stats, **config)
            model.load_weights(path_to_model / 'model-best.h5') # only weights were saved

    elif os.path.exists(path_to_model / 'save_model.pb'): # Model has been saved directly using tf.keras
        model = tf.keras.model.load_model(path_to_model)

    elif os.path.exists(path_to_model / 'weights.h5'): # Model's weights have been saved directly using tf.keras
        model = rnn_model(train_stats, **config)
        model.load_weights(path_to_model / 'weights.h5')

    else: raise FileNotFoundError("Couldnt find a model to load")

    return model, train_stats, config


def predict_macroscopics(
        model: tf.keras.Model,
        data: tf.data.Dataset,
        train_stats: dict,
        config: dict,
        batch_size: int = 256,
        single_batch: bool = False,
        ):
    """
    Use the given model to predict the features of the given data.
    If standardized, rescale the predictions to their original units.

    :param model: Keras RNN model
    :param data: Tensorflow dataset containing 'load_sequence' and 'contact_parameters' inputs.
    :param train_stats: Dictionary containing statistics of the trainingset.
    :param config: Dictionary containing the configuration with which the model was trained.
    :param batch_size: Size of batches to use.
    :param single_batch: Whether to predict only a single batch (defaults to False).

    :return: predictions: Tensorflow dataset containing the predictions in original units.
    """
    data = data.batch(batch_size)
    if single_batch:
        data = tf.data.Dataset.from_tensor_slices(next(iter(data))).batch(batch_size)

    if config['use_windows']:
        predictions = predict_over_windows(data, model, config['window_size'], train_stats['sequence_length'])
    else:
        raise NotImplementedError()

    if config['standardize_outputs']:
        mean = tf.cast(train_stats['mean'], tf.float32)
        std = tf.cast(train_stats['std'], tf.float32)
        predictions = predictions.map(lambda y: std * y + mean)
    return predictions


if __name__ == '__main__':

    # TODO: Put these possibilitites in docs

    # 1. Chossing the best model from a sweep (Aron)
    #entity_project_sweep_id = 'apjansen/grain_sequence/xyln7qwp/'
    #model, data, train_stats, config = get_best_run_from_sweep(entity_project_sweep_id)

    # 2. Chossing a model saved to trained_models folder
    path_to_trained_model = Path('trained_models/Luisas_model')
    model, train_stats, config = get_pretrained_model(path_to_trained_model)

    # load input data for the model to predict. Here hook your load sequence, contact_params
    data, _ = prepare_datasets(**config)

    predictions = predict_macroscopics(model, data['test'], train_stats, config,
            batch_size=256, single_batch=True)
