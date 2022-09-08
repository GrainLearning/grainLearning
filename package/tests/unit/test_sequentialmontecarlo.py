#%%
import numpy as np

from grainlearning import SequentialMonteCarlo, Observations, Model, Parameters


def test_smc_init():
    """Test initialization of Sequential Monte Carlo class.
    Also check to test if the inverse normalized sigma matrix is calculated correctly
    """
    smc_cls = SequentialMonteCarlo(
        ess_target=0.1, inv_obs_weight=[0.5, 0.25], scale_cov_with_max=True
    )
    assert isinstance(smc_cls, SequentialMonteCarlo)

    smc_dct = SequentialMonteCarlo.from_dict(
        {"ess_target": 0.1, "inv_obs_weight": [0.5, 0.25], "scale_cov_with_max": True}
    )

    np.testing.assert_equal(smc_dct.__dict__, smc_cls.__dict__)

    np.testing.assert_array_almost_equal(
        smc_dct._inv_normalized_sigma,
        [
            [
                1.41421356,
                0.0,
            ],
            [0.0, 0.70710678],
        ],
        0.00001,
    )


def test_get_covariance_matrix():
    """Test to see if covariance matrix is generated as expected"""
    smc_cls = SequentialMonteCarlo(
        ess_target=0.1, inv_obs_weight=[1, 1], scale_cov_with_max=True
    )

    obs = Observations(
        data=[[100, 200, 300], [30, 10, 5]],
        ctrl=[-0.1, -0.2, -0.3],
        names=["F", "G"],
        ctrl_name="x",
    )

    mdl = Model()
    mdl.observations = obs

    cov_matrices = smc_cls.get_covariance_matrices(100, mdl)

    #: assert shape is (num_steps,num_obs,num_obs)
    assert cov_matrices.shape == (3, 2, 2)

    np.testing.assert_array_almost_equal(
        cov_matrices,
        [
            [[30000.0, 0.0], [0.0, 3000.0]],
            [[30000.0, 0.0], [0.0, 3000.0]],
            [[30000.0, 0.0], [0.0, 3000.0]],
        ],
    )


# def test_get_likelihood():
"""Test to see if likelihood is generated as expected"""
smc_cls = SequentialMonteCarlo(
    ess_target=0.1, inv_obs_weight=[1, 1], scale_cov_with_max=True
)

obs = Observations(
    data=[[100, 200, 300], [30, 10, 5]],
    ctrl=[-0.1, -0.2, -0.3],
    names=["F", "G"],
    ctrl_name="x",
)

mdl = Model()
mdl.observations = obs
mdl.num_samples = 5
data = []
for _ in range(5):
    data.append(np.random.rand(2, 3))
mdl.data = np.array(data)

cov_matrices = np.repeat([np.identity(2)], 3, axis=0) * 100

likelihoods = smc_cls.get_likelihoods(mdl, cov_matrices)

assert likelihoods.shape == (3, 5)

#%%

def test_get_posterior():
    """Test to see if posterior is generated as expected"""
    smc_cls = SequentialMonteCarlo(
        ess_target=0.1, inv_obs_weight=[1, 1], scale_cov_with_max=True
    )

    obs = Observations(
        data=[[100, 200, 300], [30, 10, 5]],
        ctrl=[-0.1, -0.2, -0.3],
        names=["F", "G"],
        ctrl_name="x",
    )

    mdl = Model()
    mdl.observations = obs
    mdl.num_samples = 5
    likelihoods = np.ones((3, 5)) * 0.5

    proposal_prev = np.ones([mdl.num_samples]) / mdl.num_samples

    posteriors = smc_cls.get_posterors(
        model=mdl, likelihoods=likelihoods, proposal_prev=proposal_prev
    )

    np.testing.assert_array_almost_equal(
        posteriors,
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
    )


def test_ips_covs():
    """Test to see if ips is generated as expected."""

    smc_cls = SequentialMonteCarlo(
        ess_target=0.1, inv_obs_weight=[1, 1], scale_cov_with_max=True
    )

    obs = Observations(
        data=[[100, 200, 300], [30, 10, 5]],
        ctrl=[-0.1, -0.2, -0.3],
        names=["F", "G"],
        ctrl_name="x",
    )

    mdl = Model()
    mdl.observations = obs
    mdl.num_samples = 5
    mdl.parameters = Parameters(names=["A", "B"], mins=[2, 2], maxs=[10, 10])
    mdl.parameters.generate_halton(5)

    posteriors = np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    ips, covs = smc_cls.get_ensamble_ips_covs(model=mdl, posteriors=posteriors)

    assert ips.shape == (3, 2)
    assert covs.shape == (3, 2)

# %%
