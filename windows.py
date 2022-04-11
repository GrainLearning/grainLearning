import numpy as np

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
    inputs, outputs = data
    num_samples, sequence_length, num_labels = outputs.shape
    Xs, ys = [], []
    start, end = 0, window_size
    while end + 1 < sequence_length:
        Xs.append(inputs[:, start:end])
        ys.append(outputs[:, end + 1])
        start += window_step
        end += window_step

    Xs = np.array(Xs)
    ys = np.array(ys)
    # now we have the first dimension for samples and the second for windows,
    # we want to merge those to treat them as independent samples
    num_indep_samples = Xs.shape[0] * Xs.shape[1]
    Xs = np.reshape(Xs, (num_indep_samples,) + Xs.shape[2:])
    ys = np.reshape(ys, (num_indep_samples,) + ys.shape[2:])

    return _shuffle(Xs, ys, seed)

def _shuffle(Xs, ys, seed):
    np.random.seed(seed)
    inds = np.random.permutation(len(Xs))
    return Xs[inds], ys[inds]

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
        predictions.append(model(inputs[:, start:end]))
        start += 1
        end += 1

    predictions = np.array(predictions)
    predictions = np.transpose(predictions, (1, 0, 2))

    return predictions

