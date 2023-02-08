"""
Fixtures defined here can be used in tests located at different files.
"""
import h5py, pytest
import numpy as np


@pytest.fixture(scope="session")
def hdf5_test_file():
    """
    Creates a hdf5 file with the format of data that the module is able to handle.
    """
    target = "./tests/data/rnn/test_dataset_rnn.hdf5"
    with h5py.File(target, "w") as file: # This will rewrite the file
        # create groups pressure/experiment_type
        file.require_group("2000000.6/undrained")
        file.require_group("1000000/undrained")
        file.require_group("0.2e5/drained")

        # group 0.2e5
        inputs, outputs, contact_params = create_dataset(num_samples=2, sequence_length=3,
                                            num_load_features=2, num_labels=4, num_contact_params=5)
        file['0.2e5/drained/inputs'] = inputs
        file['0.2e5/drained/outputs'] = outputs
        file['0.2e5/drained/contact_params'] = contact_params

        # group 1000000
        inputs, outputs, contact_params = create_dataset(num_samples=10, sequence_length=3,
                                            num_load_features=2, num_labels=4, num_contact_params=5)
        file['1000000/undrained/inputs'] = inputs
        file['1000000/undrained/outputs'] = outputs
        file['1000000/undrained/contact_params'] = contact_params

        # group 2000000.6
        inputs, outputs, contact_params = create_dataset(num_samples=4, sequence_length=3,
                                            num_load_features=2, num_labels=4, num_contact_params=5)
        file['2000000.6/undrained/inputs'] = inputs
        file['2000000.6/undrained/outputs'] = outputs
        file['2000000.6/undrained/contact_params'] = contact_params

    return target


def create_dataset(num_samples: int, sequence_length: int, num_load_features: int,
                   num_labels: int, num_contact_params: int):
    """
    Creates a dataset (numpy) with random numbers between 0-1, given the dimensions.
    We set the seed so that the creation of the dataset is deterministic.
    """
    np.random.seed(42)
    inputs = np.random.rand(num_samples, sequence_length, num_load_features)
    outputs = np.random.rand(num_samples, sequence_length, num_labels)
    contact_params = np.random.rand(num_samples, num_contact_params)

    return inputs, outputs, contact_params
