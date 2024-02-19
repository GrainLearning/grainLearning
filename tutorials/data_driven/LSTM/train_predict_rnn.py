import grainlearning.rnn.train as train_rnn
from grainlearning.rnn import preprocessor
from grainlearning.rnn.predict import predict_batch, get_pretrained_model
from grainlearning.rnn.evaluate_model import plot_metric_distribution

# 1. Create my dictionary of configuration
my_config = {
     'raw_data': 'triaxial_compression_variable_input.hdf5',
     'pressure': '0.2e6',
     'experiment_type': 'drained',
     'add_pressure': True,
     'add_e0': True,
     'train_frac': 0.8,
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
     'save_weights_only': False
 }

# 2. Create an object Preprocessor to pre-process my data
preprocessor_TC = preprocessor.PreprocessorTriaxialCompression(**my_config)

# # 3. Run the training using bare tensorflow
# history_simple = train_rnn.train_without_wandb(preprocessor_TC, config=my_config)

# 3. Run the training using wandb
history_wandb = train_rnn.train(preprocessor_TC, config=my_config)

# 5. Load input data to predict from
model, train_stats, config = get_pretrained_model(preprocessor_TC.run_dir)
data = preprocessor_TC.prepare_single_dataset()

# 6. Make predictions and plot the histogram of errors
predictions = predict_batch(history_wandb.model, data, train_stats, config, batch_size=len(data))
fig = plot_metric_distribution(data, predictions, config)
fig.show()
