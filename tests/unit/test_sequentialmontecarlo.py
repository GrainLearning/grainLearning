#%%
import numpy as np

from grainlearning import SequentialMonteCarlo, Model


def test_smc_init():
    """Test initialization of Sequential Monte Carlo class."""
    smc_cls = SequentialMonteCarlo(ess_target=0.1, scale_cov_with_max=True)
    assert isinstance(smc_cls, SequentialMonteCarlo)

    smc_dct = SequentialMonteCarlo.from_dict(
        {"ess_target": 0.1, "scale_cov_with_max": True}
    )

    np.testing.assert_equal(smc_dct.__dict__, smc_cls.__dict__)


def test_get_covariance_matrix():
    """Test to see if covariance matrix is generated as expected"""

    model_cls = Model(
        param_mins=[1, 2],
        param_maxs=[3, 4],
        obs_data=[[12, 3, 4], [12, 4, 5]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=3,
    )

    smc_cls = SequentialMonteCarlo(ess_target=0.1, scale_cov_with_max=True)

    cov_matrices = smc_cls.get_covariance_matrices(100, model_cls)

    #: assert shape is (num_steps,num_obs,num_obs)
    assert cov_matrices.shape == (3, 2, 2)

    np.testing.assert_array_almost_equal(
        cov_matrices,
        [
            [[1200.0, 0.0], [0.0, 1200.0]],
            [[1200.0, 0.0], [0.0, 1200.0]],
            [[1200.0, 0.0], [0.0, 1200.0]],
        ],
    )

    smc_cls = SequentialMonteCarlo(ess_target=0.1, scale_cov_with_max=False)

    cov_matrices = smc_cls.get_covariance_matrices(100, model_cls)

    np.testing.assert_array_almost_equal(
        cov_matrices,
        [
            [[14400.0, 0.0], [0.0, 14400.0]],
            [[900.0, 0.0], [0.0, 1600.0]],
            [[1600.0, 0.0], [0.0, 2500.0]],
        ],
    )


def test_get_likelihood():
    """Test to see if likelihood is generated as expected"""
    smc_cls = SequentialMonteCarlo(
        ess_target=0.1, scale_cov_with_max=True
    )


    model_cls = Model(
        param_mins=[1, 2],
        param_maxs=[3, 4],
        obs_data=[[100, 200, 300], [30, 10, 5]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=5,
    )


    sim_data = []
    for _ in range(model_cls.num_samples):
        sim_data.append(np.random.rand(2, 3))
    model_cls.sim_data = np.array(sim_data)

    cov_matrices = np.repeat([np.identity(2)], 3, axis=0) * 100

    likelihoods = smc_cls.get_likelihoods(model_cls, cov_matrices)

    assert likelihoods.shape == (3, 5)

#%%

def test_get_posterior():
    """Test to see if posterior is generated as expected"""
    smc_cls = SequentialMonteCarlo(
        ess_target=0.1, scale_cov_with_max=True
    )

    model_cls = Model(
        param_mins=[1, 2],
        param_maxs=[3, 4],
        obs_data=[[100, 200, 300], [30, 10, 5]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=5,
        inv_obs_weight=[1, 1],
    )

    likelihoods = np.ones((3, 5)) * 0.5

    posteriors = smc_cls.get_posterors(
        model=model_cls, likelihoods=likelihoods, proposal_ibf= None
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
        ess_target=0.1, scale_cov_with_max=True
    )


    model_cls = Model(
        param_mins=[2, 2],
        param_maxs=[10, 10],
        obs_data=[[100, 200, 300], [30, 10, 5]],
        num_samples=5,
    )

    posteriors = np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    ips, covs = smc_cls.get_ensamble_ips_covs(model=model_cls, posteriors=posteriors)

    assert ips.shape == (3, 2)
    assert covs.shape == (3, 2)