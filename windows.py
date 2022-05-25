import numpy as np
import tensorflow as tf


def windowize_train_val(split_data, train_stats, window_size, window_step, **kwargs):
    """
    Convert sequences into windows of given length. Leave test set untouched.

    Args:
        split_data (dict): Dictionary with keys 'train', 'val', 'test' pointing to
            tensorflow datasets.
        train_stats (dict): Dictionary storing statistics of the training data.
        window_size (int): Number of timesteps to include in a window.
        window_step (int): Offset between subsequent windows.

    Returns:
        windows: Dictionary of dataset splits, where the training split has been
            modified into windows

    Modifies:
        train_stats: Adds window_size and window_step to this dictionary.
    """
    windows = split_data
    windows['train'] = _windowize_single_dataset(split_data['train'], window_size, window_step)
    windows['val'] = _windowize_single_dataset(split_data['val'], window_size, window_step)

    train_stats['window_size'] = window_size
    train_stats['window_step'] = window_step

    return windows


def _windowize_single_dataset(
        data,
        window_size: int,
        window_step: int,
        seed: int = 42):
    """
    Take a dataset of sequences of shape N, S, L and output another dataset
    of shorter sequences of size `window_size`, taken at intervals `window_step`
    so of shape M, window_size, L, with M >> N.
    Also shuffle the data.
    """
    load_sequences, contact_parameters, outputs = extract_tensors(data)
    num_samples, sequence_length, num_labels = outputs.shape
    start, end = 0, window_size

    # For brevity denote load_sequence, contacts, outputs as X, c, y
    Xs, cs, ys = [], [], []
    for end in range(window_size, sequence_length + 1):
        input_window = load_sequences[:, end - window_size:end]
        final_output = outputs[:, end - 1]
        Xs.append(input_window)
        ys.append(final_output)
        cs.append(contact_parameters)

    Xs = np.array(Xs)
    cs = np.array(cs)
    ys = np.array(ys)

    # now we have the first dimension for samples and the second for windows,
    # we want to merge those to treat each window as an independent sample
    num_indep_samples = Xs.shape[0] * Xs.shape[1]
    Xs = np.reshape(Xs, (num_indep_samples,) + Xs.shape[2:])
    cs = np.reshape(cs, (num_indep_samples,) + cs.shape[2:])
    ys = np.reshape(ys, (num_indep_samples,) + ys.shape[2:])

    # finally shuffle the windows
    Xs, cs, ys =  _shuffle(Xs, cs, ys, seed)
    # and convert back into a tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(({'load_sequence': Xs, 'contact_parameters': cs}, ys))
    return dataset

def _shuffle(Xs, cs, ys, seed):
    np.random.seed(seed)
    inds = np.random.permutation(len(Xs))
    return Xs[inds], cs[inds], ys[inds]

def predict_over_windows(
        inputs,
        model,
        window_size: int,
        sequence_length: int,
        #batch_size: int,
        ):
    """
    Take a batch of full sequences, iterate over windows making predictions.

    It splits up the sequence into windows of given length, each offset by one timestep,
    uses the model to make predictions on all of those windows,
    and concatenates the result into a whole sequence again.
    Note the length of the output sequence will be shorter by the window_size than
    the input sequence.

    Args:
        inputs (dict): Dictionary of inputs 'load_sequence' and 'contact_parameters'.
        model: The model to predict with.
        window_size (int): Number of timesteps in a single window.
        sequence_length (int): Number of timesteps in a full sequence.
        batch_size (int): Batch size to do predictions on.

    Returns:
        Tensor of predicted sequences.
    """
    predictions = []

    for end in range(window_size, sequence_length):
        prediction = model([inputs['load_sequence'][:, end - window_size:end], inputs['contact_parameters']])
        predictions.append(prediction)
    # concatenate predictions of windows back into a sequence
    predictions = tf.stack(predictions, axis=1)

    return predictions

def extract_tensors(data):
    """
    Given a tensorflow Dataset extract all tensors.

    Args:
        data: Tensorflow dataset.

    Returns:
        Tuple of numpy arrays inputs, contacts, outputs.
    """
    inputs, contacts, outputs = [], [], []
    for _inputs, _outputs in iter(data):
        inputs.append(_inputs['load_sequence'])
        contacts.append(_inputs['contact_parameters'])
        outputs.append(_outputs)

    return np.array(inputs), np.array(contacts), np.array(outputs)

