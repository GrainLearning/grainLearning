import numpy as np
from grainlearning import SMC, DynamicSystem, GaussianMixtureModel, generate_params_qmc

def test_smc_init():
    """Test initialization of Sequential Monte Carlo class."""
    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)
    assert isinstance(smc_cls, SMC)

    smc_dct = SMC.from_dict(
        {"ess_target": 0.1, "scale_cov_with_max": True}
    )

    np.testing.assert_equal(smc_dct.__dict__, smc_cls.__dict__)


def test_get_covariance_matrix():
    """Test to see if covariance matrix is generated as expected"""

    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=[[12, 3, 4], [12, 4, 5]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=3,
    )

    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)

    cov_matrices = smc_cls.get_covariance_matrices(100, system_cls)

    #: assert shape is (num_steps,num_obs,num_obs)
    assert cov_matrices.shape == (3, 2, 2)

    np.testing.assert_array_almost_equal(
        cov_matrices,
        [
            [[14400.0, 0.0], [0.0, 14400.0]],
            [[14400.0, 0.0], [0.0, 14400.0]],
            [[14400.0, 0.0], [0.0, 14400.0]],
        ],
    )

    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=False)
    cov_matrices = smc_cls.get_covariance_matrices(100, system_cls)
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
    smc_cls = SMC(
        ess_target=0.1, scale_cov_with_max=True
    )

    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=[[100, 200, 300], [30, 10, 5]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=5,
    )

    sim_data = []
    for _ in range(system_cls.num_samples):
        sim_data.append(np.random.rand(2, 3))

    system_cls.sim_data = np.array(sim_data)
    cov_matrices = np.repeat([np.identity(2)], 3, axis=0) * 100
    likelihoods = smc_cls.get_likelihoods(system_cls, cov_matrices)

    assert likelihoods.shape == (3, 5)


# %%

def test_get_posterior():
    """Test to see if posterior is generated as expected"""
    smc_cls = SMC(
        ess_target=0.1, scale_cov_with_max=True
    )

    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=[[100, 200, 300], [30, 10, 5]],
        ctrl_data=[1, 2, 3, 4],
        num_samples=5,
        inv_obs_weight=[1, 1],
    )

    likelihoods = np.ones((3, 5)) * 0.5

    posteriors = smc_cls.get_posterors(
        system=system_cls, likelihoods=likelihoods, proposal_ibf=None
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

    smc_cls = SMC(
        ess_target=0.1, scale_cov_with_max=True
    )

    system_cls = DynamicSystem(
        param_min=[2, 2],
        param_max=[10, 10],
        obs_data=[[100, 200, 300], [30, 10, 5]],
        num_samples=5,
    )

    gmm_cls = GaussianMixtureModel(max_num_components=1)
    system_cls.param_data = generate_params_qmc(system_cls, system_cls.num_samples)
    posteriors = np.array(
        [
            [0.1, 0.2, 0.3, 0.2, 0.2],
            [0.2, 0.1, 0.2, 0.3, 0.1],
            [0.3, 0.2, 0.2, 0.1, 0.2],
        ]
    )
    ips, covs = smc_cls.get_ensemble_ips_covs(system=system_cls, posteriors=posteriors)

    np.testing.assert_array_almost_equal(
        ips,
        [
            [4.8, 5.02222222],
            [4.5, 3.75555556],
            [4., 4.4],
        ],
    )

    np.testing.assert_array_almost_equal(
        covs,
        [
            [0.4145781, 0.3729435],
            [0.51759176, 0.51966342],
            [0.48733972, 0.45218241],
        ],
    )
