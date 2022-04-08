from tensorflow.keras import layers, Sequential
import tensorflow as tf


def baseline_model(num_features: int, num_labels: int,
        lstm_units: int = 50, dense_units: int = 20,
        seed: int = 42):
    """
    Baseline model based on Ma et al., meant to work on data where contact
    parameters are simply concatenated on to the sequence data.
    Default parameters also taken from Ma et al.
    """

    tf.random.set_seed(seed)

    model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.LSTM(lstm_units),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_labels),
        ])

    return model


def baseline_model_seq(num_features: int, num_labels: int,
        lstm_units: int = 50, dense_units: int = 20,
        seed: int = 42):
    """
    Baseline model based on Ma et al., meant to work on data where contact
    parameters are simply concatenated on to the sequence data.
    Default parameters also taken from Ma et al.
    """

    tf.random.set_seed(seed)

    model = Sequential([
        layers.Input(shape=(None, num_features)),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_labels),
        ])

    return model
