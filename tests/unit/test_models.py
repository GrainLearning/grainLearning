import numpy as np
from grainlearning import Model


def test_init():
    """Test if models are initialized correctly"""
    model_cls = Model(
        param_mins=[1, 2],
        param_maxs=[3, 4],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=10,
    )

    assert isinstance(model_cls, Model)

    config = {
        "param_mins": [1, 2],
        "param_maxs": [3, 4],
        "obs_data":[[12, 3, 4, 4], [12, 4, 5, 4]],
        "ctrl_data": [1, 2, 3, 4],
        "num_samples": 10,
    }
    model_dct = Model.from_dict(config)

    np.testing.assert_equal(model_cls.__dict__, model_dct.__dict__)
    np.testing.assert_array_almost_equal(
        model_cls._inv_normalized_sigma,
        [
            [
                1.41421356,
                0.0,
            ],
            [0.0, 0.70710678],
        ],
        0.00001,
    )

def test_run_model():
    """Test if the run model callback function works as expected"""
    def run_model(model):

        model.sim_data = [
            [[12, 3, 4, 4]],
            [[11, 24, 4, 3]],
        ]  # must be of shape (num_samples,num_obs,num_steps), thus addinga dummy dimension

    config = {
        "param_mins": [1, 2],
        "param_maxs": [3, 4],
        "obs_data": [2, 4, 6, 7],
        "ctrl_data": [1, 2, 3, 4],
        "inv_obs_weigh": [0.5, 0.25],
        "num_samples": 2,
        "callback": run_model,
    }
    model_cls = Model.from_dict(config)

    model_cls.run()

    np.testing.assert_almost_equal(
        model_cls.sim_data,
        [
            [[12, 3, 4, 4]],
            [[11, 24, 4, 3]],
        ],
    )


def test_generate_halton():
    """Test the Parameters class if the generated halton sequence is between mins and maxs"""
    model_cls = Model.from_dict(
        {
            "param_mins": [1, 2],
            "param_maxs": [3, 4],
            "obs_data": [2, 4, 6, 7],
            "ctrl_data": [1, 2, 3, 4],
            "num_samples": 2,
        }
    )

    print(model_cls.param_data)
    
    assert all(
        model_cls.param_data[:, 1] >= 2 - 0.0000001
    ), "parameter 1 min out of range"
    assert all(
        model_cls.param_data[:, 1] <= 4 + 0.0000001
    ), "parameter 1 max out of range"
    assert all(
        model_cls.param_data[:, 0] >= 1 - 0.0000001
    ), "parameter 2 min out of range"
    assert all(
        model_cls.param_data[:, 0] <= 3 + 0.0000001
    ), "parameter 3 max out of range"