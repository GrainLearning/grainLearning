import numpy as np
import tensorflow as tf
import h5py

from windows import windowize_train_val

# full lists of pressures and experiment types
PRESSURES = ['0.2e6', '0.5e6', '1.0e6']
EXPERIMENT_TYPES = ['drained', 'undrained']

def prepare_datasets(
        raw_data: str,
        pressure: str = '0.2e6',
        experiment_type: str = 'drained',
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        pad_length: int = 0,
        use_windows: bool = True,
        window_size: int = -1,
        window_step: int = -1,
        standardize_outputs: bool = True,
        add_e0: bool = False,
        seed: int = 42,
        **kwargs,
        ):
    """
    Convert raw data into preprocessed split datasets.

    :param raw_data: Path to hdf5 file containing the data.
    :param pressure: Transverse pressure as a string in format '0.xe6'.
    :param experiment_type: Either 'drained', 'undrained' or 'All'.
    :param train_frac: Fraction of data used in the training set.
    :param val_frac: Fraction of the data used in the validation set.
    :param pad_length: Amount by which to pad the sequences from the start.
    :param use_windows: Whether to split up the time series into windows.
    :param window_size: Number of timesteps to include in a window.
    :param window_step: Offset between subsequent windows.
    :param standardize_outputs: Whether to transform the training set labels
            to have zero mean and unit variance.
    :param add_e0: Whether to add the initial void ratio as a contact parameter.
    :param seed: Random seed used to split the datasets.

    :return: (split_data, train_stats)
        split_data: Dictionary with keys 'train', 'val', 'test', and values the
        corresponding tensorflow Datasets.
        train_stats: Dictionary containing the shape of the data, and
        'mean' and 'std' of the training set, in case `standardize_outputs` is True.
    """
    datafile = h5py.File(raw_data, 'r')

    inputs, outputs, contacts = _merge_datasets(datafile, pressure, experiment_type)
    if add_e0:
        contacts = _add_e0_to_contacts(contacts, inputs)

    if pad_length:
        inputs = _pad_initial(inputs, pad_length)
        outputs = _pad_initial(outputs, pad_length)

    dataset = ({'load_sequence': inputs, 'contact_parameters': contacts}, outputs)
    split_data = _make_splits(dataset, train_frac, val_frac, seed)

    if standardize_outputs:
        split_data, train_stats = _standardize_outputs(split_data)
    else:
        train_stats = {}

    split_data = {key: tf.data.Dataset.from_tensor_slices(val) for key, val in split_data.items()}

    train_stats.update(get_dimensions(split_data['train']))

    if use_windows:
        split_data = windowize_train_val(split_data, train_stats, window_size, window_step)

    return split_data, train_stats

def _merge_datasets(datafile: h5py._hl.files.File, pressure: str, experiment_type: str):
    """
    Merge the datasets with different pressures and experiment types.
    If `pressure` or `experiment_type` is 'All'.
    Otherwise just return the inputs, outputs and contact_params for the given pressure and experimen_type.

    :param datafile: h5py file containing the dataset.
    :param pressure: Experiment confining pressure, 'All' will take all pressures available.
    :param experiment_type: 'drained', 'undrained' or 'All'

    :return: input, output and contact_params arrays merged for the given pressures and expriment_types.
    """
    if pressure == 'All': pressures = PRESSURES
    else: pressures = [pressure]

    if experiment_type == 'All': experiment_types = EXPERIMENT_TYPES
    else: experiment_types = [experiment_type]

    input_sequences,output_sequences,contact_params = ([] for _ in range(3))
    for pres in pressures:
        for exp_type in experiment_types:
            data = datafile[pres][exp_type]
            input_sequences.append(data['inputs'][:])
            output_sequences.append(data['outputs'][:])

            cps = data['contact_params'][:]
            cps = _augment_contact_params(
                    cps, pres, exp_type,
                    pressure == 'All',
                    experiment_type == 'All')
            contact_params.append(cps)

    input_sequences = np.concatenate(input_sequences, axis=0)
    output_sequences = np.concatenate(output_sequences, axis=0)
    contact_params = np.concatenate(contact_params, axis=0)

    return input_sequences, output_sequences, contact_params

def _add_e0_to_contacts(contacts: np.array, inputs: np.array):
    """
    Add the initial void ratio e_0 as an extra contact parameter at the end.
    :param contacts: List of contact parameters
    :param inputs: List of input parameters
    :return: Modified contacts list with e_0 added at the end.
    """
    e0s = inputs[:, 0, 0]  # first element in series, 0th feature == e_0
    e0s = np.expand_dims(e0s, axis=1)
    contacts = np.concatenate([contacts, e0s], axis=1)
    return contacts

def _augment_contact_params(
        contact_params: np.array, pressure: str, experiment_type: str,
        add_pressure: bool, add_type: bool):
    """
    Add the pressure and the experiment type as contact parameters.
    Pressure is divided by 10**6, i.e. '0.3e6' becomes 0.3.
    Experiment type is converted to 1 for drained and 0 for undrained.

    :param contact_params: Array containing contact parameters for all the
            samples with the given pressure and experiment type
    :param pressure: The corresponding pressure.
    :param experiment_type: The corresponding experiment type, 'drained' or 'undrained'.
    :param add_pressure: Whether to add pressure to contact parameters.
    :param add_type: Whether to add experiment type to contact parameters.

    :return: Numpy array containing augmented contact parameters.
    """
    new_info = []
    pres_num = float(pressure) / 10**6
    if add_pressure: new_info.append(pres_num)
    if add_type: new_info.append(experiment_type == 'drained')

    num_samples = contact_params.shape[0]
    new_info = np.expand_dims(new_info, 0)
    new_info = np.repeat(new_info, num_samples, axis=0)

    return np.concatenate([contact_params, new_info], axis=1)

def _make_splits(dataset: tuple, train_frac: float, val_frac: float, seed: int):
    """
    Split data into training, validation, and test sets.
    The split is done on a sample by sample basis, so sequences are not broken up.
    It is done randomly across all pressures and experiment types present.

    :param dataset: Full dataset to split on, in form of a tuple (inputs, outputs),
            where inputs is a dictionary and its keys and the outputs are numpy arrays.
    :param train_frac: Fraction of data used for training set.
    :param val_frac: Fraction of data used for validation set.
    :param seed: Random seed used to make the split.
    :return: Dictionary containing 'train', 'val', and 'test' datasets.
    """
    n_tot = dataset[1].shape[0]
    n_train = int(train_frac * n_tot)
    n_val = int(val_frac * n_tot)

    np.random.seed(seed=seed)
    inds = np.random.permutation(np.arange(n_tot))
    i_train, i_val, i_test = inds[:n_train], inds[n_train:n_train + n_val], inds[-n_val:]

    def get_split(dataset, inds):
        X = {key: tf.gather(val, inds) for key, val in dataset[0].items()}
        y = tf.gather(dataset[1], inds)
        return X, y

    split_data = {
            'train': get_split(dataset, i_train),
            'val': get_split(dataset, i_val),
            'test': get_split(dataset, i_test),
            }
    return split_data

def _standardize_outputs(split_data):
    """
    Standardize outputs of split_data using the mean and std of the training data
    taken over both the samples and the timesteps.
    """
    train_outputs = split_data['train'][1]
    mean = np.mean(train_outputs, axis=(0, 1))
    std = np.std(train_outputs, axis=(0, 1))
    train_stats = {'mean': mean, 'std': std}

    def _standardize(X, y):
        return X, (y - mean) / std

    standardized_splits = split_data
    for split in ['train', 'val']:
        standardized_splits[split] = _standardize(*split_data[split])

    return standardized_splits, train_stats

def _pad_initial(array: np.array, pad_length: int, axis=1):
    """
    Add `pad_length` copies of the initial state in the sequence to the start.
    This is used to be able to predict also the first timestep from a window
    of the same size.

    :param array: Array that is going to be modified
    :param pad_lenght: number of copies of the initial state to be added at the beggining of `array`.

    :return: Modified array
    """
    starts = array[:, :1, :]
    padding = tf.repeat(starts, pad_length, axis=axis)
    padded_array = tf.concat([padding, array], axis=axis)
    return padded_array

def get_dimensions(data: tf.Dataset):
    """
    Extract dimensions of sample from a tensorflow dataset.

    :param data: The dataset to extract from.

    :return: Dictionary containing:
            sequence_length, num_load_features, num_contact_params, num_labels
    """
    train_sample = next(iter(data))  # just to extract a single batch
    sequence_length, num_load_features = train_sample[0]['load_sequence'].shape
    num_contact_params = train_sample[0]['contact_parameters'].shape[0]
    num_labels = train_sample[1].shape[-1]
    return {'sequence_length': sequence_length,
            'num_load_features': num_load_features,
            'num_contact_params': num_contact_params,
            'num_labels': num_labels,
            }