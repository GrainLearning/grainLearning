import grainlearning.rnn.train as train_rnn
from grainlearning.rnn import preprocessor

# 1. Create my dictionary of configuration
my_config = {
     'raw_data': 'triaxial_compression_variable_input.hdf5',
     'pressure': '0.2e6',
     'experiment_type': 'drained',
     'add_pressure': True,
     'add_e0': True,
     'train_frac': 0.1,
     'val_frac': 0.1,
     'window_size': 10,
     'window_step': 1,
     'patience': 25,
     'epochs': 10,
     'learning_rate': 0.006200000000000001,
     'lstm_units': 122,
     'dense_units': 224,
     'batch_size': 216,
     'standardize_outputs': True,
     'save_weights_only': True
 }

# 2. Create an object Preprocessor to pre-process my data
preprocessor_TC = preprocessor.PreprocessorTriaxialCompression(**my_config)

# # 3. Run the training using bare tensorflow
# history_simple = train_rnn.train_without_wandb(preprocessor_TC, config=my_config)

# 3. Run the training using wandb
history_wandb = train_rnn.train(preprocessor_TC, config=my_config)
