import wandb
import tensorflow as tf


from models import rnn_model
from preprocessing import prepare_datasets
from windows import predict_over_windows

def get_best_run_from_sweep(entity_project_sweep_id: str):
    """
    Load the best performing model found with a weights and biases sweep.
    Also load the data splits it was trained on.

    Args:
        entity_project_sweep_id (str): string of form {user}/{project}/{sweep_id}

    Returns:
        model: The trained model with the lowest validation loss.
        data: The train, val, test splits as used during training.
        stats: Some statistics on the training set and configuration.
        config: The configuration dictionary used to train the model.
    """
    sweep = wandb.Api().sweep(entity_project_sweep_id)
    best_run = sweep.best_run()
    best_model = wandb.restore(
            'model-best.h5',  # this saves the model locally under this name
            run_path=entity_project_sweep_id + best_run.id,
            replace=True,
        )
    config = best_run.config
    data, stats = prepare_datasets(**config)
    model = rnn_model(stats, **config)
    model.load_weights(best_model.name)
    return model, data, stats, config


def predict_macroscopics(
        model,
        data,
        train_stats: dict,
        config: dict,
        batch_size: int = 256,
        single_batch: bool = False,
        ):
    """
    Use the given model to predict the features of the given data.
    If standardized, rescale the predictions to their original units.

    Args:
        model:
        data: Tensorflow dataset containing 'load_sequence' and 'contact_parameters' inputs.
        train_stats: Dictionary containing statistics of the trainingset.
        config: Dictionary containing the configuration with which the model was trained.
        batch_size (int): Size of batches to use.
        single_batch (bool): Whether to predict only a single batch (defaults to False).

    Returns:
        predictions: Tensorflow dataset containing the predictions in original units.
    """
    data = data.batch(batch_size)
    if single_batch:
        data = tf.data.Dataset.from_tensor_slices(next(iter(data))).batch(batch_size)

    if config['use_windows']:
        predictions = predict_over_windows(data, model, config['window_size'], train_stats['sequence_length'])
    else:
        raise NotImplementedError()

    return predictions


if __name__ == '__main__':
    entity_project_sweep_id = 'apjansen/grain_sequence/xyln7qwp/'
    model, data, stats, config = get_best_run_from_sweep(entity_project_sweep_id)
    predictions = predict_macroscopics(model, data['test'], stats, config,
            batch_size=256, single_batch=False)


