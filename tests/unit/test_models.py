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
