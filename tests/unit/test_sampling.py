"""Test the sampling module."""
import numpy as np
from grainlearning import GaussianMixtureModel, DynamicSystem, generate_params_qmc


def test_init():
    """Test initialization of the Gaussian Mixture Model"""
    #: Create a Gaussian Mixture Model object
    gmm_cls = GaussianMixtureModel(
        max_num_components=5,
        covariance_type="full",
        n_init=1,
        random_state=0,
        slice_sampling=True,
    )

    gmm_dct = GaussianMixtureModel.from_dict(
        {
            "max_num_components": 5,
            "covariance_type": "full",
            "n_init": 1,
            "random_state": 0,
            "slice_sampling": True,
        }
    )

    #: Assert that the object is of the correct type
    assert isinstance(gmm_cls, GaussianMixtureModel)
    assert isinstance(gmm_dct, GaussianMixtureModel)

    # Assert that two ways of instantiation are equal
    np.testing.assert_equal(gmm_dct.__dict__, gmm_cls.__dict__)

    # Check if the weight concentration prior is set correctly by default
    assert gmm_cls.weight_concentration_prior == 1.0 / gmm_cls.max_num_components

    # Check if the covariance type is set correctly
    gmm_cls_new = GaussianMixtureModel(
        max_num_components=5,
        weight_concentration_prior=1.0,
        covariance_type="full",
        n_init=1,
        random_state=0,
        slice_sampling=True,
    )
    assert gmm_cls_new.weight_concentration_prior == 1.0


def test_expand_proposal_to_normalized_params():
    """Test if parameters are expanded given certain weights"""
    #: Make a dummy proposal distribution
    proposal = np.array([0.2, 0.4, 0.3, 0.1])

    #: Initialize a system object
    system_cls = DynamicSystem(
        param_min=[1e6, 0.19],
        param_max=[1e7, 0.5],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=4,
    )

    #: Initialize a Gaussian Mixture Model object
    gmm_cls = GaussianMixtureModel(max_num_components=5, expand_factor=3)

    #: Generate the initial parameter samples
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    #: Expand the ensemble by populating the samples with importance weights
    gmm_cls.expand_and_normalize_weighted_samples(proposal, system_cls)

    #: Check if the parameter values are normalized
    np.testing.assert_almost_equal(np.amax(gmm_cls.expanded_normalized_params, axis=0), np.ones(2))

    #: Check if the parameter values are normalized and expanded correctly
    np.testing.assert_array_almost_equal(
        gmm_cls.expanded_normalized_params,
        np.array(
            [
                [0.12903226, 0.4789916],
                [0.12903226, 0.4789916],
                [0.70967742, 0.7394958],
                [0.70967742, 0.7394958],
                [0.70967742, 0.7394958],
                [0.70967742, 0.7394958],
                [0.41935484, 1.],
                [0.41935484, 1.],
                [0.41935484, 1.],
                [1., 0.56582633]
            ]
        ),
        0.0001,
    )


def test_regenerate_params():
    """Test if parameter samples are regenerated correctly"""
    #: Make a dummy proposal distribution
    proposal = np.array([0.2, 0.3, 0.3, 0.2])

    system_cls = DynamicSystem(
        param_min=[1e6, 0.19],
        param_max=[1e7, 0.5],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=4,
    )

    #: Initialize a Gaussian Mixture Model object
    gmm_cls = GaussianMixtureModel(max_num_components=2, covariance_type="full", random_state=100, expand_factor=2)

    #: Generate the initial parameter samples
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    #: Expand the ensemble by populating the samples with importance weights
    gmm_cls.expand_and_normalize_weighted_samples(proposal, system_cls)

    #: Check if the parameter values are normalized
    np.testing.assert_almost_equal(np.amax(gmm_cls.expanded_normalized_params, axis=0), np.ones(2))

    #: Regenerate the parameters
    new_params = gmm_cls.regenerate_params(proposal, system_cls)

    #: Assert if the new parameters are correct
    np.testing.assert_array_almost_equal(
        new_params,
        np.array(
            [
                [2.50061801e+06, 1.92539376e-01],
                [5.40525882e+06, 3.10537276e-01],
                [3.46458943e+06, 3.76945456e-01],
                [5.66254261e+06, 2.67135646e-01]
            ]
        ),
        0.0001,
    )

    #: Make a new dummy proposal distribution
    proposal = np.array([0.0, 0.5, 0.4, 0.1])

    #: Initialize again a Gaussian Mixture Model object with slice sampling activated
    gmm_cls = GaussianMixtureModel(max_num_components=2, covariance_type="full", expand_factor=10, slice_sampling=True)

    #: Generate the initial parameter samples
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    #: Regenerate the parameter samples
    new_params = gmm_cls.regenerate_params(proposal, system_cls)

    #: Assert if the new parameters are correct
    np.testing.assert_array_almost_equal(
        new_params,
        np.array(
            [
                [5.50000000e+06, 2.93333333e-01],
                [3.25000000e+06, 3.96666667e-01],
                [6.90625000e+06, 2.47407407e-01],
                [4.23437500e+06, 3.35432099e-01]
            ]
        ),
        0.0001,
    )


def test_draw_samples_within_bounds():
    """Test if the samples are drawn within the bounds"""
    #: Make a dummy proposal distribution
    proposal = np.array(np.ones(10) / 10)

    #: Initialize a system object
    system_cls = DynamicSystem(
        param_min=[1e6, 0.19],
        param_max=[1e7, 0.5],
        obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=proposal.shape[0],
    )

    #: Initialize a Gaussian Mixture Model object
    gmm_cls = GaussianMixtureModel(max_num_components=2, random_state=100, expand_factor=2)

    #: Generate the initial parameter samples
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    #: Train the Gaussian Mixture Model
    gmm_cls.train(proposal, system_cls)

    # Adjust the bounds to be smaller than the current range
    system_cls.param_min = [3e6, 0.25]
    system_cls.param_max = [7e6, 0.45]

    # Draw samples within the new bounds
    new_params = gmm_cls.draw_samples_within_bounds(system_cls, system_cls.num_samples)

    # Assert if the new parameters are generated correctly
    np.testing.assert_array_almost_equal(
        new_params,
        np.array(
            [
                [5.28625749e+06, 2.64309176e-01],
                [3.17723928e+06, 2.60832533e-01],
                [4.08334948e+06, 3.34190765e-01]
            ]
        ),
        0.0001,
    )

    #: Initialize a new Gaussian Mixture Model object
    gmm_cls.slice_sampling = True

    # Draw samples within the new bounds
    new_params = gmm_cls.draw_samples_within_bounds(system_cls, system_cls.num_samples)

    # Assert if the new parameters are generated correctly
    np.testing.assert_array_almost_equal(
        new_params,
        np.array(
            [
                [5.00000000e+06, 3.16666667e-01],
                [4.00000000e+06, 3.83333333e-01],
                [6.00000000e+06, 2.72222222e-01],
                [3.50000000e+06, 3.38888889e-01],
                [5.50000000e+06, 4.05555556e-01],
                [4.50000000e+06, 2.94444444e-01],
                [6.50000000e+06, 3.61111111e-01],
                [3.25000000e+06, 4.27777778e-01],
                [5.25000000e+06, 2.57407407e-01]
            ]
        ),
        0.0001,
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
