"""Test the sampling module."""
from os import remove
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
    gmm_cls = GaussianMixtureModel(max_num_components=2, covariance_type="tied", init_params="k-means++",
                                   random_state=100, expand_factor=2)

    #: Generate the initial parameter samples
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    #: Expand the ensemble by populating the samples with importance weights
    gmm_cls.expand_and_normalize_weighted_samples(proposal, system_cls)

    #: Check if the parameter values are normalized
    np.testing.assert_almost_equal(np.amax(gmm_cls.expanded_normalized_params, axis=0), np.ones(2))

    #: Regenerate the parameters
    new_params = gmm_cls.regenerate_params(proposal, system_cls)

    #: Assert if the new parameters are correct
    np.testing.assert_allclose(
        new_params,
        np.array(
            [
                [3.50816173e+06, 2.49928167e-01],
                [4.87521971e+06, 3.09866017e-01],
                [2.57761751e+06, 3.26589170e-01],
                [5.39169338e+06, 2.52627014e-01]
            ]
        ),
        rtol=0.001,
    )

    #: Make a new dummy proposal distribution
    proposal = np.array([0.0, 0.5, 0.4, 0.1])

    #: Initialize again a Gaussian Mixture Model object with slice sampling activated
    gmm_cls = GaussianMixtureModel(max_num_components=2, covariance_type="tied", init_params="k-means++",
                                   expand_factor=10, slice_sampling=True)

    #: Generate the initial parameter samples
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)

    #: Regenerate the parameter samples
    new_params = gmm_cls.regenerate_params(proposal, system_cls)

    #: Assert if the new parameters are correct
    np.testing.assert_allclose(
        new_params,
        np.array(
            [
                [5.50000000e+06, 2.93333333e-01],
                [3.25000000e+06, 3.96666667e-01],
                [5.07812500e+06, 3.20123457e-01],
                [4.55078125e+06, 3.30329218e-01]
            ]
        ),
        rtol=0.001,
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
    np.testing.assert_allclose(
        new_params,
        np.array(
            [
                [3.72038687e+06, 2.50311001e-01],
                [5.31282617e+06, 3.30634332e-01],
                [6.04092450e+06, 2.61269286e-01],
                [5.21447138e+06, 2.84730873e-01],
                [3.60796701e+06, 2.81671695e-01]
            ]
        ),
        rtol=0.001,
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


def test_save_and_load_gmm():
    """Test if the Gaussian Mixture Model is saved correctly to a file"""
    #: Make a dummy proposal distribution
    proposal = np.array([0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

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

    #: Save the Gaussian Mixture Model to a file
    gmm_cls.save_gmm_to_file("test_gmm.pkl")

    #: Load the Gaussian Mixture Model from a file
    gmm_cls_new = GaussianMixtureModel(max_num_components=2, random_state=100, expand_factor=2)
    gmm_cls_new.load_gmm_from_file("test_gmm.pkl")

    #: Check if the loaded Gaussian Mixture Model is the same as the original one
    np.testing.assert_equal(gmm_cls_new.gmm.__dict__, gmm_cls.gmm.__dict__)

    #: Check if the loaded max_params is the same as the original one
    np.testing.assert_equal(gmm_cls_new.max_params, gmm_cls_new.max_params)

    #: Delete the file
    remove("test_gmm.pkl")


def test_generate_params_qmc():
    """Test the Parameters class if the generated halton sequence is between mins and maxs"""
    system_cls = DynamicSystem.from_dict(
        {
            "param_min": [1, 2],
            "param_max": [3, 4],
            "obs_data": [2, 4, 6, 7],
            "ctrl_data": [1, 2, 3, 4],
            "num_samples": 5,
            "callback": None,
        }
    )

    #: Check if the halton sequence is generated correctly
    np.testing.assert_array_almost_equal(
        generate_params_qmc(system_cls, system_cls.num_samples),
        np.array(
            [
                [1., 2.],
                [2., 2.66666667],
                [1.5, 3.33333333],
                [2.5, 2.22222222],
                [1.25, 2.88888889]
            ]
        ),
        0.0001,
    )

    #: Check if the sobol sequence is generated correctly
    np.testing.assert_array_almost_equal(
        generate_params_qmc(system_cls, system_cls.num_samples, "sobol", 1),
        np.array(
            [
                [1.31093064, 3.17749465],
                [2.67676943, 2.20812031],
                [2.22669116, 3.71838415],
                [1.84694732, 2.68699761]
            ]
        ),
        0.0001,
    )

    #: Check if the Latin Hypercube is generated correctly
    np.testing.assert_array_almost_equal(
        generate_params_qmc(system_cls, system_cls.num_samples, "LH", 1),
        np.array(
            [
                [1.59527135, 3.61981452],
                [2.54233615, 3.22054022],
                [1.27526742, 2.63066942],
                [1.86891896, 2.23632035],
                [2.78016252, 3.18897635]
            ]
        ),
        0.0001,
    )
