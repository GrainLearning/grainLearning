import wandb
import grainlearning.rnn.train as train_rnn
from grainlearning.rnn import preprocessor
import os

os.environ["WANDB_MODE"] = "offline"


def my_training_function():
    """ A function that wraps the training process"""
    # update window_size of my_config from wandb
    with wandb.init():
        my_config['window_size'] = wandb.config['window_size']
    preprocessor_TC = preprocessor.PreprocessorTriaxialCompression(**my_config)
    train_rnn.train(preprocessor_TC)


# 1. Create my dictionary of configuration
my_config = {
    'raw_data': 'triaxial_compression_variable_input.hdf5',
    'pressure': '0.2e6',
    'experiment_type': 'drained',
    'add_experiment_type': False,
    'add_pressure': True,
    'add_e0': True,
    'train_frac': 0.1,
    'val_frac': 0.1,
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


# 2. A function to specify the sweep configuration, returning a sweep_id
def get_sweep_id(method):
    sweep_config = {
        'method': 'random',
        'name': method,
        # TODO: @Luisa: how does user know what metric is defined in the model?
        'metric': {'goal': 'minimize', 'name': 'mae'},
        'parameters': {},
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 27
        }
    }
    # add default parameters from my_config into sweep_config, use 'values' as the key
    for key, value in my_config.items():
        sweep_config['parameters'].update({key: {'values': [value]}})
    # update sweep_config with the parameters to be searched and their distributions
    # check the documentation https://docs.wandb.ai/guides/sweeps/sweep-config-keys#distribution-options-for-random-and-bayesian-search on how to define the search space
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
        'dense_units': {
            'distribution': 'q_log_uniform_values',
            'q': 1,
            'min': 32,
            'max': 256
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 1,
            'min': 32,
            'max': 256
        },
        'window_size': {
            'distribution': 'q_uniform',
            'q': 2,
            'min': 4,
            'max': 30
        },
    }
    sweep_config['parameters'].update(search_space)
    # create the sweep
    sweep_id = wandb.sweep(sweep_config, project='triax_sweep')
    return sweep_id


# 3. Log in to wandb
wandb.login()

# 4. Configure the sweep
sweep_id = get_sweep_id('random')

# 5. Run an agent
wandb.agent(sweep_id, function=my_training_function, count=100)
