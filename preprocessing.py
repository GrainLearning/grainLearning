import numpy as np
import h5py
import tensorflow as tf

PRESSURES = [0.2, 0.5, 1.0]
EXPERIMENT_TYPES = ['drained', 'undrained']

def prepare_datasets(
        raw_data: str,
        pressure: str = '0.2e6',
        experiment_type: str = 'drained',
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        pad_length: int = 0,
        standardize_outputs: bool = True,
        seed: int = 42,
        ):

    datafile = h5py.File(raw_data, 'r')

    inputs, outputs, contacts = _merge_datasets(datafile, pressure, experiment_type)
    dataset = tf.data.Dataset.from_tensor_slices(({'load_sequence': inputs, 'contact_parameters': contacts}, outputs))

    if pad_length:
        dataset = _pad_initial(dataset, pad_length)

    split_data = _make_splits(dataset, train_frac, val_frac, seed)

    if standardize_outputs:
        split_data, train_stats = _standardize_outputs(split_data)
    else:
        train_stats = dict()

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

def _make_splits(dataset, train_frac, val_frac, seed):
    """
    Split data into train, val, test sets,  based on samples,
    not within a sequence.
    data is a tuple of arrays.
    """
    n_tot = len(dataset)
    n_train = int(train_frac * n_tot)
    n_val = int(val_frac * n_tot)

    dataset = dataset.shuffle(n_tot, seed=seed)
    train = dataset.take(n_train)
    remaining = dataset.skip(n_train)
    val = remaining.take(n_val)
    test = remaining.skip(n_val)

    split_data = {
            'train': train,
            'val': val,
            'test': test,
            }
    return split_data

def _standardize_outputs(split_data):
    """
    Standardize outputs, using the mean and std of the training data.
    """
    def get_means(inputs, outputs):
        return tf.reduce_mean(outputs, axis=(0,1))
    def get_stds(inputs, outputs):
        return tf.math.reduce_std(outputs, axis=(0,1))
    means = split_data['train'].map(get_means)
    mean = means.reduce(np.float64(0.), lambda x, y: x + y) / len(split_data['train'])
    stds = split_data['train'].map(get_stds)
    std = stds.reduce(np.float64(0.), lambda x, y: x + y) / len(split_data['train'])

    train_stats = {'mean': mean, 'std': std}

    def _standardize(inputs, outputs):
        return inputs, (outputs - mean) / std

    standardized_splits = dict()
    for split in ['train', 'val', 'test']:
        standardized_splits[split] = split_data[split].map(_standardize)

    return standardized_splits, train_stats

def _pad_initial(dataset, pad_length):
    """
    Add `pad_length` copies of the initial step in the sequence.
    """
    def pad_sequence(inputs, outputs):
        inputs['load_sequence'] = _pad_array(inputs['load_sequence'], pad_length)
        outputs = _pad_array(outputs, pad_length)
        return inputs, outputs

    dataset = dataset.map(pad_sequence)
    return dataset

def _pad_array(array, pad_length, axis=0):
    """Applied to samples without a batch dimension!"""
    starts = array[:1, :]
    padding = tf.repeat(starts, pad_length, axis=axis)
    padded_array = tf.concat([padding, array], axis=axis)
    return padded_array

