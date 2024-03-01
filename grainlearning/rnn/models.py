"""
Module containing a function that creates a RNN model.
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


def rnn_model(
        input_shapes: dict,
        window_size: int = 20,
        lstm_units: int = 50,
        dense_units: int = 20,
        seed: int = 42,
        **_,
        ):
    """
    Neural network with an LSTM layer.

    Takes in an input sequence and the parameters and produces an output sequence.
    The parameters are used to initialize the hidden state of the LSTM.

    :param input_shapes: Dictionary containing `'num_input_features'`, `'num_params'`,
        `'num_labels'`. It can contain other keys but these are the ones used here.
    :param window_size: Length of time window.
    :param lstm_units: Number of units of the hidden state of the LSTM.
    :param dense_units: Number of units used in the dense layer after the LSTM.
    :param seed: The random seed used to initialize the weights.

    :return: A Keras model.
    """
    # make initialization of weights reproducible
    tf.random.set_seed(seed)

    sequence_length = window_size
    load_sequence = layers.Input(
            shape=(sequence_length, input_shapes['num_input_features']), name='inputs')
    params = layers.Input(shape=(input_shapes['num_params'],), name='params')

    # compute hidden state of LSTM based on contact parameters
    state_h = layers.Dense(lstm_units, activation='tanh', name='state_h')(params)
    state_c = layers.Dense(lstm_units, activation='tanh', name='state_c')(params)
    initial_state = [state_h, state_c]

    X = load_sequence
    X = layers.LSTM(lstm_units, return_sequences=False)(X,
            initial_state=initial_state)

    X = layers.Dense(dense_units, activation='relu')(X)
    outputs = layers.Dense(input_shapes['num_labels'])(X)

    model = Model(inputs=[load_sequence, params], outputs=outputs)

    return model


def rnn_model_for_triax(
        input_shapes: dict,
        window_size: int = 20,
        lstm_units: int = 50,
        dense_units: int = 20,
        seed: int = 42,
        **_,
        ):
    """
    A wrapper of neural network model with an LSTM layer for triaxial loading conditions.
    :param input_shapes: Dictionary containing `'num_load_features'`, `'num_contact_params'`, and `'num_labels'`.
    :param window_size: Length of time window.
    :param lstm_units: Number of units of the hidden state of the LSTM.
    :param dense_units: Number of units used in the dense layer after the LSTM.
    :param seed: The random seed used to initialize the weights.

    :return: A Keras model.
    """
    # change the name of the keys to match the original model
    input_shapes['num_input_features'] = input_shapes.pop('num_load_features')
    input_shapes['num_params'] = input_shapes.pop('num_contact_params')
    return rnn_model(input_shapes, window_size, lstm_units, dense_units, seed)
