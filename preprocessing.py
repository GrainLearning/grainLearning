import numpy as np
import h5py

PRESSURES = [0.2, 0.5, 1.0]
EXPERIMENT_TYPES = ['drained', 'undrained']

def prepare_datasets(
        raw_data: str,
        standardize: bool = True,
        concatenate_constants: bool = True,
        pressure: str = '0.2e6',
        experiment_type: str = 'drained',
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        pad_length: int = 0,
        seed: int = 42,
        ):

    datafile = h5py.File(raw_data, 'r')

    inputs, outputs, contacts = _merge_datasets(datafile, pressure, experiment_type)

    if concatenate_constants:
        inputs = _concatenate_constants(inputs, contacts)
        data_used = (inputs, outputs)
    else:
        data_used = (inputs, outputs, contacts)

    split_data = _make_splits(data_used, train_frac, val_frac, seed)

    if standardize:
        split_data, train_stats = _standardize_outputs(split_data)

    if pad_length:
        split_data = _pad_initial(split_data, pad_length)
    return split_data, train_stats

def _merge_datasets(datafile, pressure, experiment_type):
    """
    Merge the datasets with different pressures and experiment types,
    if `pressure` or `experiment_type` is 'All'.
    Otherwise just return the inputs, outputs and contact_params.
    """
    if pressure == 'All':
        pressures = PRESSURES
    else:
        # NOTE: relies on pressure being of form x.ye6
        pressures = [float(pressure[:3])]
    if experiment_type == 'All':
        experiment_types = EXPERIMENT_TYPES
    else:
        experiment_types = [experiment_type]

    input_sequences = []
    output_sequences = []
    contact_params = []
    for pres in pressures:
        for exp_type in experiment_types:
            data = datafile[f'{pres}e6'][exp_type]
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

def _augment_contact_params(
        contact_params, pressure: float, experiment_type: str,
        add_pressure: bool, add_type: bool):
    new_info = []
    if add_pressure: new_info.append(pressure)
    if add_type: new_info.append(experiment_type == 'drained')

    new_info = np.expand_dims(new_info, 0)
    new_info = np.repeat(new_info, contact_params.shape[0], axis=0)

    return np.concatenate([contact_params, new_info], axis=1)

def _concatenate_constants(inputs, contacts):
    contacts_sequence = np.expand_dims(contacts, axis=1)
    sequence_length = inputs.shape[1]
    contacts_sequence = np.repeat(contacts_sequence, sequence_length, 1)
    total_inputs = np.concatenate([inputs, contacts_sequence], axis=2)
    return total_inputs

def _make_splits(data, train_frac, val_frac, seed):
    """
    Split data into train, val, test sets,  based on samples,
    not within a sequence.
    data is a tuple of arrays.
    """
    n_tot = data[0].shape[0]
    n_train = int(train_frac * n_tot)
    n_val = int(val_frac * n_tot)

    np.random.seed(seed)
    inds = np.random.permutation(n_tot)
    split_data = {
            'train': tuple(d[inds[:n_train]] for d in data),
            'val': tuple(d[inds[n_train:n_train + n_val]] for d in data),
            'test': tuple(d[inds[n_train + n_val:]] for d in data),
            }
    return split_data

def _standardize_outputs(split_data):
    """
    Standardize outputs, using the mean and std of the training data.
    """
    # NOTE: relies on outputs being at index 1!
    y_mean = split_data['train'][1].mean(axis=(0, 1)),
    train_stats = {
        'y_mean': y_mean,
        'y_std': (split_data['train'][1] - y_mean).std(axis=(0, 1)),
        }

    standardized_splits = dict()
    for split in ['train', 'val', 'test']:
        standardized_splits[split] = _standardize(split_data[split], train_stats)

    return standardized_splits, train_stats

def _standardize(data, stats):
    data = list(data)
    data[1] = (data[1] - stats['y_mean']) / stats['y_std']
    data = tuple(data)
    return data

def _pad_initial(split_data, pad_length):
    """
    Add `pad_length` copies of the initial step in the sequence.
    NOTE: needs fixing if contact parameters included separately.
    """
    for split in ['train', 'val', 'test']:
        X, y = split_data[split]
        X_padded = _pad_array(X, pad_length)
        y_padded = _pad_array(y, pad_length)
        split_data[split] = X_padded, y_padded

    return split_data

def _pad_array(array, pad_length, axis=1):
    starts = array[:, :1, :]
    padding = np.repeat(starts, pad_length, axis=axis)
    padded_array = np.concatenate([padding, array], axis=axis)
    return padded_array




