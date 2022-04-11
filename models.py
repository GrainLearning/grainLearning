from tensorflow.keras import layers, Model
import tensorflow as tf


def baseline_model(
        num_load_features: int,
        num_contact_params: int,
        num_labels: int,
        window_size: int,
        lstm_units: int = 50,
        dense_units: int = 20,
        seed: int = 42,
        ):
    """
    Baseline model based on Ma et al., meant to work on data where contact
    parameters are simply concatenated on to the sequence data.
    Default parameters also taken from Ma et al.
    """

    tf.random.set_seed(seed)

    load_sequence = layers.Input(shape=(window_size, num_load_features), name='load_sequence')
    contact_params = layers.Input(shape=(num_contact_params,), name='contact_parameters')
    contact_params_repeated = layers.RepeatVector(window_size)(contact_params)
    X = layers.Concatenate()([load_sequence, contact_params_repeated])

    X = layers.LSTM(lstm_units)(X)
    X = layers.Dense(dense_units, activation='relu')(X)
    outputs = layers.Dense(num_labels)(X)

    model = Model(inputs=[load_sequence, contact_params], outputs=outputs)

    return model

def baseline_model_seq(num_load_features: int, num_labels: int,
        lstm_units: int = 50, dense_units: int = 20,
        seed: int = 42):
    """
    Baseline model based on Ma et al., meant to work on data where contact
    parameters are simply concatenated on to the sequence data.
    Default parameters also taken from Ma et al.
    """

    tf.random.set_seed(seed)

    model = Sequential([
        layers.Input(shape=(None, num_load_features)),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_labels),
        ])

    return model

def conditional(num_load_features: int, num_params: int, num_labels: int,
        lstm_units: int = 50, dense_units: int = 20, seed: int = 42,
        ):

    tf.random.set_seed(seed)

    input_sequences = layers.Input(shape=(200, num_load_features))
    input_params = layers.Input(shape=(num_params))

    hidden_state = layers.Dense(2 * lstm_units, activation='relu')(input_params)
    hidden_state = layers.Dense(2 * lstm_units, activation='relu')(hidden_state)
    state_h, state_c = layers.Lambda(split)(hidden_state)

    # line below complaining None type is not subscriptable..
    outputs = layers.LSTM(lstm_units)(input_sequences, initial_state=[state_h, state_c])
    outputs = layers.Dense(dense_units, activation='relu')
    outputs = layers.Dense(num_labels)

    model = keras.Model(inputs=[input_sequences, input_params], outputs=outputs)

    return model

def split(tensor):
    tensor = tf.reshape(tensor, (-1, 2, tensor.shape[1] // 2))
    a, b = tf.split(tensor, 2, axis=1)
    a = tf.squeeze(a)
    b = tf.squeeze(b)
    return [a, b]

def main():
    num_load_features = 3
    num_params = 6
    num_labels = 10
    model = conditional(num_load_features, num_params, num_labels)


if __name__ == '__main__':
    main()

