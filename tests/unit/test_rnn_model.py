import pytest, h5py, wandb, sys, os, io, shutil
import numpy as np
import tensorflow as tf
from pathlib import Path

from grainlearning.rnn.models import rnn_model
import grainlearning.rnn.preprocessing as preprocessing
import grainlearning.rnn.train as train
import grainlearning.rnn.predict as predict

@pytest.fixture(scope="function") # will tear down the fixture after being used in a test_function
def config_test(hdf5_test_file):
    return {
        'raw_data': hdf5_test_file,
        'pressure': '2000000.6',
        'experiment_type': 'undrained',
        'add_e0': False,
        'add_pressure': False,
        'add_experiment_type': False,
        'train_frac': 0.5,
        'val_frac': 0.25,
        'window_size': 1,
        'window_step': 1,
        'pad_length': 0,
        'lstm_units': 2,
        'dense_units': 2,
        'patience': 2,
        'epochs': 2,
        'learning_rate': 1e-3,
        'batch_size': 2,
        'standardize_outputs': False,
        'save_weights_only': False
    }


# Tests models
def test_model_output_shape():
    """ Test if rnn model can be initialized and outputs the expected shape. """
    # Normally gotten from train_stats after data loading
    input_shapes = {
            'num_contact_params': 6,
            'num_load_features': 4,
            'num_labels': 5,
            'sequence_length': 200,
        }
    window_size = 20
    batch_size = 2
    model = rnn_model(input_shapes, window_size)
    assert len(model.layers) == 7 # 2 inputs, 2 hidden states lstm, lstm, dense, dense_output

    test_input_sequence = np.random.normal(size=(batch_size, window_size, input_shapes['num_load_features']))
    test_contacts = np.random.normal(size=(batch_size, input_shapes['num_contact_params']))

    output = model({'load_sequence': test_input_sequence,
                    'contact_parameters': test_contacts})

    assert output.shape == (batch_size, input_shapes['num_labels'])

# Tests train
@pytest.mark.skip(reason="Developing the tests this one takes long.")
def test_train(hdf5_test_file, config_test, monkeypatch):
    """
    Check that training goes well, no errors should be thrown.
    """

    # Option 1: train using wandb
    os.system("wandb offline") # so that when you run these test the info will not be synced
    history_wandb = train.train(config=config_test)
        # check that files have been generated
    assert os.path.exists(Path("wandb/latest-run/files/model-best.h5"))
    assert os.path.exists(Path("wandb/latest-run/files/train_stats.npy"))
    # if running offline this will not be generated
    #assert os.path.exists(Path("wandb/latest-run/files/config.yaml"))
        # check metrics
    assert history_wandb.history.keys() == {'loss', 'mae', 'val_loss', 'val_mae'}

    # Option 2: train using plain tensorflow
        # monkeypatch for input when asking: do you want to proceed? [y/n]:
    monkeypatch.setattr('sys.stdin', io.StringIO('y'))
    history_simple = train.train_without_wandb(config=config_test)
        # check that files have been generated
    assert os.path.exists(Path("outputs/saved_model.pb")) # because 'save_weights_only': False
    assert os.path.exists(Path("outputs/train_stats.npy"))
    assert os.path.exists(Path("outputs/config.npy"))
        # check metrics
    assert history_simple.history.keys() == {'loss', 'mae', 'val_loss', 'val_mae'}

    # removing generated folders
    shutil.rmtree("wandb")
    shutil.rmtree("outputs")

    # Check that if 'save_weights_only' other sort of files would be saved
    config_test['save_weights_only'] = True # can safely do this because the scope of fixture is function

        # Option 1: train using wandb
    train.train(config=config_test)
    assert os.path.exists(Path("wandb/latest-run/files/model-best.h5"))
    assert os.path.exists(Path("wandb/latest-run/files/train_stats.npy"))

        # Option 2: train using plain tensorflow
    train.train_without_wandb(config=config_test)
    assert os.path.exists(Path("outputs/weights.h5")) # because 'save_weights_only': True
    assert os.path.exists(Path("outputs/train_stats.npy"))
    assert os.path.exists(Path("outputs/config.npy"))

    # removing generated folders
    shutil.rmtree("wandb")
    shutil.rmtree("outputs")


# Tests predict
#@pytest.mark.skip(reason="Under construction.")
def test_get_pretrained_model(hdf5_test_file, config_test):
    """ Try to load some models pretrained on synthetic data.
        Such syntetic data was generated using test_train, thus hdf5_test_file with config_test (2000000.6, undrained).
    """
    path_to_model_test = ["./tests/data/rnn/wandb_entire_model",
                          "./tests/data/rnn/wandb_only_weights",
                          "./tests/data/rnn/plain_entire_model",
                          "./tests/data/rnn/plain_only_weights"
                         ]
    config_test_weights_only = config_test.copy()
    config_test_weights_only['save_weights_only'] = True

    for path_to_model in path_to_model_test:
        model, train_stats, config = predict.get_pretrained_model(path_to_model)

        # test number of layers in model
        assert len(model.layers) == 7 # 2 inputs, 2 hidden states lstm, lstm, dense, dense_output
        # test that the model loaded works
        model.summary() # Will throw an exception if the model was not loaded correctly

        # test that train_stats has expected members and values
        assert train_stats.keys() == {'sequence_length', 'num_load_features',
                                      'num_contact_params', 'num_labels'}

        # test params in config matching original config
        if "only_weights" in path_to_model:
            assert config == config_test_weights_only
        else:
            assert config == config_test

    # test that error is trigger if unexistent file is passed

def test_predict_macroscopics(hdf5_test_file):
    model, train_stats, config = predict.get_pretrained_model("./tests/data/rnn/wandb_only_weights/")
    data, _ = preprocessing.prepare_datasets(**config)
    predictions_1 = predict.predict_macroscopics(model, data['test'], train_stats, config, batch_size=1)
    config['pad_length'] = config['window_size']
    config['train_frac'] = 0.25
    config['val_frac'] = 0.25
    data_padded, train_stats_2 = preprocessing.prepare_datasets(**config)
    predictions_2 = predict.predict_macroscopics(model, data_padded['test'], train_stats_2, config, batch_size=2)

    assert isinstance(predictions_1, tf.Tensor)
    assert isinstance(predictions_2, tf.Tensor)

    # check dimensions, check batch size is correctly applied
    assert predictions_1.shape == (1, train_stats['sequence_length'] - config['window_size'], train_stats['num_labels'])
    assert predictions_1.shape == (1, 3 - 1, 4)
    # in 2 case the sequence predicted should have the same size as the inputs (begin was padded).
    assert predictions_2.shape == (2, train_stats['sequence_length'], train_stats['num_labels'])
    assert predictions_2.shape == (2, 3, 4) # always good in case train_stats or config are broken.

    # model loaded: pad_length=0, config, pad_length=1. If using train_stats of the model -> incompatible.
    data_padded = preprocessing.prepare_single_dataset(**config)
    with pytest.raises(ValueError):
        predictions_3 = predict.predict_macroscopics(model, data_padded, train_stats, config, batch_size=2)

    # check that standardize outputs has been correctly applied: cannot comprare.

#@pytest.mark.skip(reason="Under construction.")
def test_predict_over_windows(hdf5_test_file, config_test):
    window_size = 1
    batch_size = 1
    split_data, train_stats = preprocessing.prepare_datasets(raw_data=hdf5_test_file,
                                pressure='1000000', experiment_type='undrained', window_size=window_size)
    model = rnn_model(input_shapes=train_stats, window_size=window_size)
    data = split_data['train'].batch(batch_size)
    predictions = predict.predict_over_windows(data, model, window_size, train_stats['sequence_length'])

    # Test that the output is a tensorflow dataset
    assert isinstance(predictions, tf.data.Dataset)

    # Test that the output predictions have the correct shape
    for pred in predictions.batch(1).map(lambda x, y: x):
        assert pred.shape == (1, sequence_length - window_size, train_stats['num_labels'])

    # Test that the function can handle different batch sizes
    data = data.batch(5)
    predictions = predict.predict_over_windows(data, model, window_size, sequence_length)
    for pred in predictions.batch(1).map(lambda x, y: x):
        assert pred.shape == (5, sequence_length - window_size, train_stats['num_labels'])


