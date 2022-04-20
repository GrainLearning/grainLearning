import numpy as np
import tensorflow as tf

def sliding_windows(
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
    num_samples = len(data)
    sample = next(iter(data))
    sequence_length, num_labels = sample[1].shape
    Xs, cs, ys = [], [], []
    start, end = 0, window_size

    inputs, contacts, outputs = extract_tensors(data)

    while end <= sequence_length:
        Xs.append(inputs[:, start:end])
        cs.append(contacts)
        ys.append(outputs[:, end - 1])
        start += window_step
        end += window_step

    Xs = np.array(Xs)
    cs = np.array(cs)
    ys = np.array(ys)
    # now we have the first dimension for samples and the second for windows,
    # we want to merge those to treat them as independent samples
    num_indep_samples = Xs.shape[0] * Xs.shape[1]
    Xs = np.reshape(Xs, (num_indep_samples,) + Xs.shape[2:])
    cs = np.reshape(cs, (num_indep_samples,) + cs.shape[2:])
    ys = np.reshape(ys, (num_indep_samples,) + ys.shape[2:])

    Xs, cs, ys =  _shuffle(Xs, cs, ys, seed)
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
        ):
    """
    Take a batch of sequences, iterate over windows making predictions.
    """
    predictions = []

    start, end = 0, window_size
    while end < sequence_length:
        prediction = model([inputs['load_sequence'][:, start:end], inputs['contact_parameters']])
        predictions.append(prediction)
        start += 1
        end += 1

    predictions = np.array(predictions)
    predictions = np.transpose(predictions, (1, 0, 2))

    return predictions

def extract_tensors(data):
    inputs, contacts, outputs = [], [], []
    for _inputs, _outputs in iter(data):
        inputs.append(_inputs['load_sequence'])
        contacts.append(_inputs['contact_parameters'])
        outputs.append(_outputs)

    return np.array(inputs), np.array(contacts), np.array(outputs)

def windowize(split_data, train_stats, sequence_length,
        use_windows, window_size, window_step, **kwargs):
    if not use_windows:
        return split_data

    windows = {split: sliding_windows( split_data[split], window_size, window_step)
                for split in ['train', 'val', 'test']}

    train_stats['window_size'] = window_size
    train_stats['window_step'] = window_step
    train_stats['sequence_length'] = sequence_length

    return windows

