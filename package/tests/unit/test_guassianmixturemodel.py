#%%
import numpy as np

from grainlearning import GaussianMixtureModel, Model, Parameters


def test_init():
    gmm_cls = GaussianMixtureModel(max_num_components=5)

    assert isinstance(gmm_cls, GaussianMixtureModel)

    gmm_dct = GaussianMixtureModel.from_dict({"max_num_components": 5})

    np.testing.assert_equal(gmm_dct.__dict__, gmm_cls.__dict__)

    assert gmm_cls.prior_weight == 1.0 / 5


#%%
def test_expand_proposal_to_normalized_params():
    proposal = np.array([0.25, 0.25, 0.25, 0.25])

    mymodel = Model()

    mymodel.num_samples = 4

    mymodel.parameters = Parameters(names=["E", "pois"], mins=[1e6, 0.19], maxs=[1e7, 0.5])

    mymodel.parameters.generate_halton(num_samples=mymodel.num_samples)

    gmm_cls = GaussianMixtureModel(max_num_components=5, expand_weight=2)

    expanded_parms, max_params = gmm_cls.expand_weighted_parameters(proposal, mymodel)
    # expanded_parms
    np.testing.assert_almost_equal(np.amax(expanded_parms, axis=0), np.ones(2))

    np.testing.assert_array_almost_equal(
        expanded_parms,
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
    proposal = np.array([0.25, 0.25, 0.25, 0.25])

    mymodel = Model()

    mymodel.num_samples = 4

    mymodel.parameters = Parameters(names=["E", "pois"], mins=[1e6, 0.19], maxs=[1e7, 0.5])

    mymodel.parameters.generate_halton(num_samples=mymodel.num_samples)

    gmm_cls = GaussianMixtureModel(max_num_components=2, expand_weight=2)

    expanded_parms, max_params = gmm_cls.expand_weighted_parameters(proposal, mymodel)

    np.testing.assert_almost_equal(np.amax(expanded_parms, axis=0), np.ones(2))

    np.testing.assert_array_almost_equal(
        expanded_parms,
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

    proposal = np.array([0.25, 0.25, 0.25, 0.25])

    new_params = gmm_cls.regenerate_params(proposal,mymodel)

    assert np.all(new_params > np.array(mymodel.parameters.mins))
    assert np.all(new_params < np.array(mymodel.parameters.maxs))


# %%
