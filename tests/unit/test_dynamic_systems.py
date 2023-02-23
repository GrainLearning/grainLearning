"""Test the dynamic systems module."""
import numpy as np
from grainlearning import DynamicSystem


def test_init():
    """Test if models are initialized correctly"""
    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=10,
    )

    assert isinstance(system_cls, DynamicSystem)

    config = {
        "param_min": [1, 2],
        "param_max": [3, 4],
        "obs_data": [[12, 3, 4, 4], [12, 4, 5, 4]],
        "ctrl_data": [1, 2, 3, 4],
        "num_samples": 10,
        "callback": None,
    }
    model_dct = DynamicSystem.from_dict(config)

    np.testing.assert_equal(system_cls.__dict__, model_dct.__dict__)
    np.testing.assert_array_almost_equal(
        system_cls.get_inv_normalized_sigma(),
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
        "param_min": [1, 2],
        "param_max": [3, 4],
        "obs_data": [2, 4, 6, 7],
        "ctrl_data": [1, 2, 3, 4],
        "inv_obs_weigh": [0.5, 0.25],
        "num_samples": 2,
        "callback": run_model,
    }
    system_cls = DynamicSystem.from_dict(config)

    system_cls.run()

    np.testing.assert_almost_equal(
        system_cls.sim_data,
        [
            [[12, 3, 4, 4]],
            [[11, 24, 4, 3]],
        ],
    )
