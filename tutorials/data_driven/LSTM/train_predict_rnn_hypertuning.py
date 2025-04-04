import grainlearning.rnn.train as train_rnn
from grainlearning.rnn.train import HyperTuning
from grainlearning.rnn import preprocessor
from grainlearning.rnn.predict import get_best_run_from_sweep, predict_batch
from grainlearning.rnn.evaluate_model import plot_metric_distribution
import wandb


def my_training_function():
    """ A function that wraps the training process"""
    # update window_size of my_config from wandb
    with wandb.init():
        my_config['window_size'] = wandb.config['window_size']
    preprocessor_TC = preprocessor.PreprocessorTriaxialCompression(**my_config)
    train_rnn.train(preprocessor_TC, config=my_config)


# 1. Create my dictionary of configuration
my_config = {
    'raw_data': 'triaxial_compression_variable_input.hdf5',
    'pressure': '0.2e6',
    'experiment_type': 'drained',
    'add_experiment_type': False,
    'add_pressure': True,
    'add_e0': True,
    'train_frac': 0.7,
    'val_frac': 0.15,
    'window_size': 20,
    'pad_length': 10,
    'window_step': 1,
    'patience': 25,
    'epochs': 10,
    'learning_rate': 1e-4,
    'lstm_units': 250,
    'dense_units': 250,
    'batch_size': 256,
    'standardize_outputs': True,
    'save_weights_only': True
}

# 2. Define the sweep configuration
sweep_config = {
    'method': 'random',
    # TODO: how does the user know about the metric?
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27
    }
}

search_space = {
    'learning_rate': {
        # a flat distribution between 1e-4 and 1e-2
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
    # 'dense_units': {
    #     'distribution': 'q_log_uniform_values',
    #     'q': 1,
    #     'min': 32,
    #     'max': 256
    # },
    # 'batch_size': {
    #     'distribution': 'q_log_uniform_values',
    #     'q': 1,
    #     'min': 32,
    #     'max': 256
    # },
    'window_size': {
        'distribution': 'q_uniform',
        'q': 2,
        'min': 4,
        'max': 30
    },
}

# 3. Run the sweep
hyper_tuner = HyperTuning(sweep_config, search_space, my_config, entity_name='grainlearning', project_name='my_sweep')
hyper_tuner.run_sweep(my_training_function, count=100)

# 4. Get the best model
# change metric to validation loss
hyper_tuner.sweep_config['metric']['name'] = 'val_loss'
entity_project_sweep_id = f"{hyper_tuner.entity_name}/{hyper_tuner.project_name}/{hyper_tuner.sweep_id}"
model, train_stats, config = get_best_run_from_sweep(entity_project_sweep_id)
config['pressure'] = my_config['pressure']

# 5. Load input data to predict from
preprocessor_TC = preprocessor.PreprocessorTriaxialCompression(**config)
data = preprocessor_TC.prepare_single_dataset()

# 6. Make and plot predictions
predictions = predict_batch(model, data, train_stats, config, batch_size=len(data))
fig = plot_metric_distribution(data, predictions, config)
fig.show()
