import numpy as np
import tensorflow as tf


def windowize_train_val_test(split_data: dict, window_size: int, window_step: int, **_):
    """
    Convert sequences into windows of given length. Leave test set untouched.

    :param split_data: Dictionary with keys `'train'`, `'val'`, `'test'` pointing to
            tensorflow datasets.
    :param window_size: Number of timesteps to include in a window.
    :param window_step: Offset between subsequent windows.

    :return: windows: Dictionary of dataset splits, where the training and validation splits have been
            modified into windows.
    """
    windows = split_data
    windows['train'] = windowize_single_dataset(split_data['train'], window_size, window_step)
    windows['val'] = windowize_single_dataset(split_data['val'], window_size, window_step)
    return windows


def windowize_single_dataset(
        data: tf.data.Dataset,
        window_size: int = 1,
        window_step: int = 1,
        seed: int = 42,
        **_):
    """
    Take a dataset ((inputs, params),outputs) of sequences of shape N, S, L and output
    another dataset of shorter sequences of size ``window_size``, taken at intervals ``window_step``.
    The resulting dataset will have ``M=((sequence_length - window_size)/window_step) + 1)`` samples,
    corresponding to independent windows in the given dataset.
    For a given window taken from inputs, the output is the element of outputs sequence at the
    last position of the window.

    .. note::
      For a clear picture of what goes on here check `Sliding windows` section in the documentation.

    Data is shuffled.

    :param data: dataset of sequences of shape N, S, L.
    :param window_size: Size of the window.
    :param window_step: Offset between subsequent windows.
    :param seed: Random seed.

    :return:
      * ``inputs`` of shape: (M, window_size, ``num_load_features``), with M >> N.
      * ``outputs`` of shape: (M, L_outputs)
    """
    inputs, params, outputs = extract_tensors(data)
    _, sequence_length, _ = outputs.shape

    if window_size >= sequence_length:
        raise ValueError(f"window_size {window_size} >= sequence_length {sequence_length}.")

    # For brevity denote inputs, params, outputs as x, c, y
    xs, cs, ys = [], [], []
    for end in range(window_size, sequence_length + 1, window_step):
        xs.append(inputs[:, end - window_size:end]) # input window
        ys.append(outputs[:, end - 1]) # final output
        cs.append(params)

    xs = np.array(xs)
    cs = np.array(cs)
    ys = np.array(ys)

    # now we have the first dimension for samples and the second for windows,
    # we want to merge those to treat each window as an independent sample
    num_indep_samples = xs.shape[0] * xs.shape[1]
    xs = np.reshape(xs, (num_indep_samples,) + xs.shape[2:])
    cs = np.reshape(cs, (num_indep_samples,) + cs.shape[2:])
    ys = np.reshape(ys, (num_indep_samples,) + ys.shape[2:])

    # finally shuffle the windows
    xs, cs, ys =  _shuffle(xs, cs, ys, seed)
    # and convert back into a tensorflow dataset
    return tf.data.Dataset.from_tensor_slices(({'inputs': xs, 'params': cs}, ys))


def _shuffle(xs, cs, ys, seed):
    np.random.seed(seed)
    inds = np.random.permutation(len(xs))
    return xs[inds], cs[inds], ys[inds]


def predict_over_windows(
        inputs: dict,
        model: tf.keras.Model,
        window_size: int,
        sequence_length: int,
        ):
    """
    Take a batch of full sequences, iterate over windows making predictions.
    It splits up the sequence into windows of given length, each offset by one timestep,
    uses the model to make predictions on all of those windows,
    and concatenates the result into a whole sequence again.
    Note the length of the output sequence will be shorter by the window_size than
    the input sequence.

    :param inputs: dict containing inputs: `'load_sequence'` and `'params'`, both being tensorflow.Tensor.
    :param model: The model to predict with.
    :param window_size: Number of timesteps in a single window.
    :param sequence_length: Number of timesteps in a full sequence.

    :return: tensorflow.Tensor of predicted sequences.
    """
    predictions = [
        model([inputs['inputs'][:, end - window_size:end], inputs['params']])
        for end in range(window_size, sequence_length)
        ]
    predictions = tf.stack(predictions, axis=1)
    return predictions


def extract_tensors(data: tf.data.Dataset):
    """
    Given a tensorflow Dataset extract all tensors.

    :param data: Tensorflow dataset.

    :return: 3 numpy arrays: inputs, contacts, outputs.
    """
    inputs, contacts, outputs = [], [], []
    for _inputs, _outputs in iter(data):
        inputs.append(_inputs['inputs'])
        contacts.append(_inputs['params'])
        outputs.append(_outputs)

    return np.array(inputs), np.array(contacts), np.array(outputs)
