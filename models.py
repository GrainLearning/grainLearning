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
        **kwargs,
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

def baseline_model_seq(
        num_load_features: int,
        num_labels: int,
        lstm_units: int = 50,
        dense_units: int = 20,
        seed: int = 42,
        **kwargs,
        ):
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

def conditional(
        num_load_features: int,
        num_contact_params: int,
        num_labels: int,
        window_size: int,
        lstm_units: int = 50,
        dense_units: int = 20,
        seed: int = 42,
        **kwargs,
        ):

    tf.random.set_seed(seed)

    load_sequence = layers.Input(shape=(window_size, num_load_features), name='load_sequence')
    contact_params = layers.Input(shape=(num_contact_params,), name='contact_parameters')

    # hidden_state = layers.Dense(dense_units, activation='relu')(contact_params)
    state_h = layers.Dense(lstm_units, activation='tanh')(contact_params)
    state_c = layers.Dense(lstm_units, activation='tanh')(contact_params)

    X = layers.LSTM(lstm_units)(load_sequence, initial_state=[state_h, state_c])
    X = layers.Dense(dense_units, activation='relu')(X)
    outputs = layers.Dense(num_labels)(X)

    model = Model(inputs=[load_sequence, contact_params], outputs=outputs)

    return model

def main():
    num_load_features = 3
    num_params = 6
    num_labels = 10
    window_size = 20
    model = conditional(num_load_features, num_params, num_labels, window_size)
    model_concat = baseline_model(num_load_features, num_params, num_labels, window_size)
    model.summary()
    model_concat.summary()
    tst_params = tf.random.normal((32, num_params))
    tst_load = tf.random.normal((32, window_size, num_load_features))
    out = model({'load_sequence': tst_load, 'contact_parameters': tst_params})
    print(out.shape)



if __name__ == '__main__':
    main()

