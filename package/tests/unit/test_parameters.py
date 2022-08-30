import numpy as np
from grainlearning import Parameters


def test_init_params():
    """Test the Parameters class to see if it initializes from __init__ and .from_dict"""
    param_cls = Parameters(
        names=["E", "Eta", "Psi"], mins=[1e6, 0.05, 7.0], maxs=[4e7, 1.0, 13.0]
    )

    assert isinstance(param_cls, Parameters)

    param_dct = Parameters.from_dict(
        {
            "names": ["E", "Eta", "Psi"],
            "mins": [1e6, 0.05, 7.0],
            "maxs": [4e7, 1.0, 13.0],
        }
    )

    raw_param_dct = param_dct.__dict__
    raw_param_cls = param_cls.__dict__

    np.testing.assert_equal(raw_param_dct, raw_param_cls)

def test_generate_halton():
    """Test the Parameters class if the generated halton sequence is between mins and maxs"""
    param_cls = Parameters(names=["E", "pois"], mins=[1e6, 0.19], maxs=[1e7, 0.5])

    param_cls.generate_halton(num_samples=10)

    print(param_cls.data)
    assert all(param_cls.data[:, 1] >= 0.19 - 0.0000001), "parameter pios min out of range"
    assert all(param_cls.data[:,1] <= 0.5 + 0.0000001), "parameter pios max out of range"
    assert all(param_cls.data[:,0] >= 1e6 - 0.0000001), "parameter pios min out of range"
    assert all(param_cls.data[:,0] <= 1e7 + 0.0000001), "parameter pios max out of range"

#%%
