import wandb
import tensorflow as tf


from models import rnn_model
from preprocessing import prepare_datasets
from evaluate_model import plot_predictions


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
    """
    sweep = wandb.Api().sweep(entity_project_sweep_id)
    best_run = sweep.best_run()
    best_model = wandb.restore('model-best.h5',
            run_path=entity_project_sweep_id + best_run.id)
    config = best_run.config
    data, stats = prepare_datasets(**config)
    model = rnn_model(stats, **config)
    model.load_weights(best_model.name)
    return model, data, stats


if __name__ == '__main__':
    entity_project_sweep_id = 'apjansen/grain_sequence/xyln7qwp/'
    model, data, stats = get_best_run_from_sweep(entity_project_sweep_id)
    fig = plot_predictions(data, model, stats)


