
import numpy as np
from grainlearning import Observations


def test_init_obs():
    """Test to see if Observation class initiates correctly from __init__ and from_dict"""

    obs_sp = Observations(
        data=[0.1, 0.5, 0.9],
        ctrl=  [0.01, 0.02, 0.03],
        ctrl_name="axail_strain",
        names=["stress"],
    )
    
    assert obs_sp.data.shape == (1,3)
    
    obs_cls = Observations(
        data=[[0.1, 0.5, 0.9], [0.007, 0.011, 0.034]],
        ctrl=  [0.01, 0.02, 0.03],
        ctrl_name="axail_strain",
        names=["stress", "volumetric_strain"],
    )

    assert isinstance(obs_cls, Observations)

    obs_dct = Observations.from_dict(
        {
            "data": [[0.1, 0.5, 0.9], [0.007, 0.011, 0.034]],
            "ctrl": [0.01, 0.02, 0.03],
            "ctrl_name": "axail_strain",
            "names": ["stress", "volumetric_strain"],
        }
    )

    # np.assert_equal(obs_dct.data, obs_cls.data)

    raw_obs_dct = obs_dct.__dict__
    raw_obs_cls = obs_cls.__dict__
    print(raw_obs_dct,raw_obs_cls)
    np.testing.assert_equal(raw_obs_dct, raw_obs_cls)

# %%
