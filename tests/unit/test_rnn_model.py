import numpy as np
from grainlearning.rnn.models import rnn_model

def test_output_shape():
    """Test if rnn model can be initialized and outputs the expected shape."""
    # Normally gotten from train_stats after data loading
    input_shapes = {
            'num_contact_params': 6,
            'num_load_features': 4,
            'num_labels': 5,
            'sequence_length': 200,
        }
    window_size = 20
    batch_size = 2
    model = rnn_model(input_shapes, window_size, conditional=True)

    test_input_sequence = np.random.normal(size=(batch_size, window_size, input_shapes['num_load_features']))
    test_contacts = np.random.normal(size=(batch_size, input_shapes['num_contact_params']))

    output = model({'load_sequence': test_input_sequence,
                    'contact_parameters': test_contacts})

    assert output.shape == (batch_size, input_shapes['num_labels'])
