import h5py
import pytest
import tensorflow as tf

from grainlearning.rnn import preprocessing
from grainlearning.rnn import windows
from tests.unit.conftest import create_dataset

@pytest.fixture(scope="session")
def dummy_dataset():
    inputs, outputs, contact_params = create_dataset(num_samples = 100, sequence_length = 10,
                              num_load_features = 4, num_labels = 3, num_contact_params = 5)
    return ({'load_sequence': inputs, 'contact_parameters': contact_params}, outputs)


def test_make_splits(dummy_dataset):
    num_samples = 100
    sequence_length = 10
    num_load_features = 4
    num_labels = 3
    num_contact_params = 5

    train_frac_list = [0.5, 0.7, 0.1]
    val_frac_list = [0.25, 0.15, 0.5]

    for train_frac, val_frac in zip(train_frac_list, val_frac_list):
        split_data = preprocessing._make_splits(dummy_dataset, train_frac, val_frac, seed=42)
        assert split_data.keys() == {'train', 'val', 'test'}

        # Check that each split contains the expected number of samples
        n_train = int(train_frac * num_samples)
        n_val = int(val_frac * num_samples)
        n_test = num_samples - n_train - n_val
        assert n_train + n_val + n_test == num_samples
        assert num_samples == split_data['train'][1].shape[0] + split_data['val'][1].shape[0] + split_data['test'][1].shape[0]

        assert split_data['train'][1].shape[0] == n_train
        assert split_data['val'][1].shape[0] == n_val
        assert split_data['test'][1].shape[0] == n_test

        # Check that the other dimensions of the arrays are still correct
        for key, n_samples in zip(split_data.keys(), [n_train, n_val, n_test]):
            assert split_data[key][0]['load_sequence'].shape == (n_samples, sequence_length, num_load_features)
            assert split_data[key][0]['contact_parameters'].shape == (n_samples, num_contact_params)
            assert split_data[key][1].shape == (n_samples, sequence_length, num_labels)

    with pytest.raises(ValueError): # check that it is not possible to have a 0 length validation dataset
        _ = preprocessing._make_splits(dummy_dataset, 1.0, 0.0, seed=42)


def test_get_dimensions(dummy_dataset):
    split_data = preprocessing._make_splits(dummy_dataset, train_frac=0.7, val_frac=0.15, seed=42)
    split_data = {key: tf.data.Dataset.from_tensor_slices(val) for key, val in split_data.items()}

    for data_i in split_data.values():
        dimensions = preprocessing.get_dimensions(data_i)

        assert dimensions.keys() == {'sequence_length', 'num_load_features', 'num_contact_params', 'num_labels'}

        # from dimensions of dummy_dataset
        assert dimensions['sequence_length'] == 10
        assert dimensions['num_load_features'] == 4
        assert dimensions['num_contact_params'] == 5
        assert dimensions['num_labels'] == 3

def test_merge_datasets(hdf5_test_file):
    with h5py.File(hdf5_test_file, 'r') as datafile:
        # Case all pressures, all experiments
        inputs, outputs, contact_parameters = preprocessing._merge_datasets(datafile,
                                              pressure='All', experiment_type='All')
        # check total (sum) num of samples, and that pressure and experiment type have been added to contact_params
        # Values to be modified if hdf5_test_file changes
        assert inputs.shape == (2 + 10 + 4, 3, 2) # num_samples, sequence_length, load_features
        assert outputs.shape == (2 + 10 + 4, 3, 4) # num_samples, sequence_length, num_labels
        assert contact_parameters.shape == (2 + 10 + 4, 5 + 2) # num_samples, num_contact_params + 2 (pressure, experiment_type)

        # Case specific pressure and experiment
        inputs, outputs, contact_parameters = preprocessing._merge_datasets(datafile,
                                              pressure='0.2e5', experiment_type='drained')
        assert inputs.shape == (2, 3, 2) # num_samples, sequence_length, load_features
        assert outputs.shape == (2, 3, 4) # num_samples, sequence_length, num_labels
        assert contact_parameters.shape == (2, 5 + 2) # num_samples, num_contact_params + 2 (pressure, experiment_type)


def test_prepare_datasets(hdf5_test_file):
    split_data, train_stats = preprocessing.prepare_datasets(hdf5_test_file,
            pressure='1000000', experiment_type='undrained',
            train_frac=0.5, val_frac=0.2,
            pad_length=0, window_size=1, window_step=1,
            standardize_outputs=True,
            add_e0=False, add_pressure=False, add_experiment_type=False)

    # Test split_data
    assert isinstance(split_data, dict)
    assert split_data.keys() == {'train', 'val', 'test'}
    for val in split_data.values():
        assert isinstance(val, tf.data.Dataset)

    # Test train_stats
    assert isinstance(train_stats, dict)
    assert train_stats.keys() == {'mean', 'std', 'sequence_length', 'num_load_features',
                                  'num_contact_params', 'num_labels'}
    assert train_stats['sequence_length'] > 0
    assert train_stats['num_load_features'] > 0
    assert train_stats['num_contact_params'] > 0
    assert train_stats['num_labels'] > 0

    # Test length of contact parameters when adding e0, pressure, experiment type
    # 1. Only add e_0
    split_data_1, train_stats_1 = preprocessing.prepare_datasets(hdf5_test_file,
         pressure='1000000', experiment_type='undrained',
         add_e0=True, add_pressure=False, add_experiment_type=False, standardize_outputs=False)

    # 2. Only add pressure
    split_data_2, train_stats_2 = preprocessing.prepare_datasets(hdf5_test_file,
         pressure='1000000', experiment_type='undrained',
         add_e0=False, add_pressure=True, add_experiment_type=False, standardize_outputs=False, pad_length=1)

    # 3. Only add experiment_type
    split_data_3, train_stats_3 = preprocessing.prepare_datasets(hdf5_test_file,
         pressure='1000000', experiment_type='undrained',
         add_e0=False, add_pressure=False, add_experiment_type=True, standardize_outputs=False, pad_length=3)

    # Comparison against train_stats: No additional contact parameters.
    assert train_stats_1['num_contact_params'] == train_stats['num_contact_params'] + 1
    assert train_stats_2['num_contact_params'] == train_stats['num_contact_params'] + 1
    assert train_stats_3['num_contact_params'] == train_stats['num_contact_params'] + 1

    # Test that standardize_outputs is applied correctly when standardize_outputs=False
    assert 'mean' not in train_stats_1.keys()
    assert 'std' not in train_stats_1.keys()

    # Test that pad_length was applied correctly
    # expected_num_indep_samples = train_frac * num_samples * int(((sequence_length - window_size)/window_step) + 1 + pad_lenght)
    assert len(split_data_1['train']) == 7 * int(((3 - 1)/1) + 1 ) # no pad_lenght
    assert len(split_data_2['train']) == 7 * int(((3 - 1)/1) + 1 + 1) # last + 1 is pad_lenght
    assert len(split_data_3['train']) == 7 * int(((3 - 1)/1) + 1 + 3) # last + 3 is pad_lenght

    # Test that error is raised when unexistent hdf5 file is passed
    with pytest.raises(FileNotFoundError):
        _ = preprocessing.prepare_datasets('unexistent_file.hdf5')

    # We could test here if an error is raised if a group doesn't have the right format: inputs, outpus, contact_params
    # but I'm supposing that people use a file from data_parsing to create their dataset, and thus this is not necessary.

# windows
def test_windowize_single_dataset(dummy_dataset):
    # dimensions with which dummy_dataset was generated.
    sequence_length = 10
    num_load_features = 4
    num_labels = 3
    num_contact_params = 5

    split_data = preprocessing._make_splits(dummy_dataset, train_frac=0.7, val_frac=0.15, seed=42)
    split_data = {key: tf.data.Dataset.from_tensor_slices(val) for key, val in split_data.items()}

    # sanity checks of the splits dimensions
    assert len(split_data['train']) == 70
    assert len(split_data['val']) == 15
    assert len(split_data['test']) == 15

    window_size_list = [2, 3, 3]
    window_step_list = [2, 2, 1]

    for window_size, window_step in zip(window_size_list, window_step_list):
        dataset = windows.windowize_single_dataset(split_data['train'], window_size=window_size, window_step=window_step)
        inputs, outputs = next(iter(dataset.batch(1000))) # batch bigger than num independent samples (windows)
        expected_num_indep_samples = 70 * int(((sequence_length - window_size)/window_step) + 1)

        assert len(dataset) == expected_num_indep_samples
        assert len(inputs) == 2 # load_sequence and contact_params
        assert inputs['load_sequence'].shape == (expected_num_indep_samples, window_size, num_load_features) # num_samples, window_size, num_load_features
        assert inputs['contact_parameters'].shape == (expected_num_indep_samples, num_contact_params) # num_samples, window_size, num_contact_params
        assert outputs.shape == (expected_num_indep_samples, num_labels) # num_samples, num_labels

    with pytest.raises(ValueError): # expect "window_size bigger than sequence_length."
        _ = windows.windowize_single_dataset(split_data['train'], window_size=sequence_length + 1)
