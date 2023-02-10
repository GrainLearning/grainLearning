import numpy as np

from grainlearning import GaussianMixtureModel, DynamicSystem, generate_params_qmc


def test_init():
    """Test initialization of the Gaussian Mixture Model"""
    gmm_cls = GaussianMixtureModel(max_num_components=5)

    assert isinstance(gmm_cls, GaussianMixtureModel)

    gmm_dct = GaussianMixtureModel.from_dict({"max_num_components": 5})

    np.testing.assert_equal(gmm_dct.__dict__, gmm_cls.__dict__)

    assert gmm_cls.prior_weight == 1.0 / 5


def test_expand_proposal_to_normalized_params():
    """Test if parameters are expanded given certain weights"""
    proposal = np.array([0.25, 0.25, 0.25, 0.25])

    system_cls = DynamicSystem(
        param_min=[1e6, 0.19],
        param_max=[1e7, 0.5],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=4,
    )

    gmm_cls = GaussianMixtureModel(max_num_components=5, expand_weight=2)

    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    gmm_cls.expand_weighted_parameters(proposal, system_cls)
    # expanded_parms
    np.testing.assert_almost_equal(np.amax(gmm_cls.expanded_normalized_params, axis=0), np.ones(2))

    np.testing.assert_array_almost_equal(
        gmm_cls.expanded_normalized_params,
        np.array(
            [
                [0.12903226, 0.4789916],
                [0.12903226, 0.4789916],
                [0.70967742, 0.7394958],
                [0.70967742, 0.7394958],
                [0.41935484, 1.0],
                [0.41935484, 1.0],
                [1.0, 0.56582633],
                [1.0, 0.56582633],
            ]
        ),
        0.0001,
    )


def test_regenerate_params():
    """Test if resampling is within bounds"""
    proposal = np.array([0.2, 0.3, 0.3, 0.2])

    system_cls = DynamicSystem(
        param_min=[1e6, 0.19],
        param_max=[1e7, 0.5],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=4,
    )

    gmm_cls = GaussianMixtureModel(max_num_components=2, expand_weight=2, seed=100, cov_type="full")

    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    gmm_cls.expand_weighted_parameters(proposal, system_cls)

    np.testing.assert_almost_equal(np.amax(gmm_cls.expanded_normalized_params, axis=0), np.ones(2))

    new_params = gmm_cls.regenerate_params(proposal, system_cls)

    np.testing.assert_allclose(
        new_params,
        np.array(
            [[2.50061801e+06, 1.92539376e-01],
             [5.40525882e+06, 3.10537276e-01],
             [3.46458943e+06, 3.76945456e-01],
             [5.66254261e+06, 2.67135646e-01]])
    )


def test_generate_halton():
    """Test the Parameters class if the generated halton sequence is between mins and maxs"""
    system_cls = DynamicSystem.from_dict(
        {
            "param_min": [1, 2],
            "param_max": [3, 4],
            "obs_data": [2, 4, 6, 7],
            "ctrl_data": [1, 2, 3, 4],
            "num_samples": 2,
            "callback": None,
        }
    )

    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    print(system_cls.param_data)

    assert all(
        system_cls.param_data[:, 1] >= 2 - 0.0000001
    ), "parameter 1 min out of range"
    assert all(
        system_cls.param_data[:, 1] <= 4 + 0.0000001
    ), "parameter 1 max out of range"
    assert all(
        system_cls.param_data[:, 0] >= 1 - 0.0000001
    ), "parameter 2 min out of range"
    assert all(
        system_cls.param_data[:, 0] <= 3 + 0.0000001
    ), "parameter 3 max out of range"
