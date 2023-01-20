"""
Module containing a function that creates a RNN model.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K


def rnn_model(
        input_shapes: dict,
        window_size: int = 20,
        lstm_units: int = 50,
        dense_units: int = 20,
        use_windows: bool = True,
        conditional: bool = False,
        seed: int = 42,
        **kwargs,
        ):
    """
    Neural network with an LSTM layer.

    Takes in a load sequence and contact parameters, and outputs the macroscopic responses.
    Default settings are based on Ma et al. and simply concatenate the contact parameters
    to the load sequence.
    This can be changed into a conditional RNN by setting ``conditional=True``.
    In that case, contact parameters are used to intialize the hidden state of the LSTM.

    :param input_shapes: Dictionary containing `'num_load_features'`, `'num_contact_params'`,
        `'num_labels'`.
    :param window_size: Length of time window.
    :param lstm_units: Number of units of the hidden state of the LSTM.
    :param dense_units: Number of units used in the dense layer after the LSTM.
    :param use_windows: Whether to use time windows (True, default)
        or process the entire sequence at once (False).
    :param conditional: Whether to use a conditional RNN (True),
        or to concatenate the contact parameters to the sequence (False, default).
    :param seed: The random seed used to initialize the weights.

    :return: A Keras model.
    """
    # make initialization of weights reproducible
    tf.random.set_seed(seed)

    sequence_length = window_size if use_windows else None  # None means variable
    load_sequence = layers.Input(
            shape=(sequence_length, input_shapes['num_load_features']), name='load_sequence')
    contact_params = layers.Input(shape=(input_shapes['num_contact_params'],), name='contact_parameters')

    if conditional:
        # compute hidden state of LSTM based on contact parameters
        state_h = layers.Dense(lstm_units, activation='tanh', name='state_h')(contact_params)
        state_c = layers.Dense(lstm_units, activation='tanh', name='state_c')(contact_params)
        initial_state = [state_h, state_c]

        X = load_sequence
    else:
        initial_state = None  # the default, just zeroes

        # concatenate contact parameters along load sequence
        num_repeats = window_size if use_windows else K.shape(load_sequence)[1]
        contact_params_repeated = _DynamicRepeatVector(contact_params, num_repeats)(contact_params)

        X = layers.Concatenate()([load_sequence, contact_params_repeated])

    X = layers.LSTM(lstm_units, return_sequences=(not use_windows))(X,
            initial_state=initial_state)

    X = layers.Dense(dense_units, activation='relu')(X)
    outputs = layers.Dense(input_shapes['num_labels'])(X)

    model = Model(inputs=[load_sequence, contact_params], outputs=outputs)

    return model


def _DynamicRepeatVector(contact_params, num_repeats: int):
    """
    To deal with repetitions of variable sequence lenghts.
    Adapted from https://github.com/keras-team/keras/issues/7949#issuecomment-383550274
    NOTE: Can't get this to work when not using windows ...
    """
    num_features = K.shape(contact_params)[1]  # contact_params.shape[1]

    def repeat_vector(contact_params):
        return layers.RepeatVector(num_repeats)(contact_params)
    return layers.Lambda(repeat_vector, output_shape=(None, num_repeats, num_features))

