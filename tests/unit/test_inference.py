"""Test the inference module."""
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
        obs_data=np.array([[12, 3, 4], [12, 4, 5]]),
        ctrl_data=np.array([1, 2, 3, 4]),
        num_samples=3,
    )

    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)

    cov_matrices = smc_cls.get_covariance_matrices(100, system_cls)

    #: assert shape is (num_steps,num_obs,num_obs)
    assert cov_matrices.shape == (3, 2, 2)

    #: assert that the covariance matrix is scaled with the maxima of the observations
    np.testing.assert_array_almost_equal(
        cov_matrices,
        [
            [[14400.0, 0.0], [0.0, 14400.0]],
            [[14400.0, 0.0], [0.0, 14400.0]],
            [[14400.0, 0.0], [0.0, 14400.0]],
        ],
    )

    #: assert that the covariance matrix is scaled with the observation sequence
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
    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)

    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=np.array([[1, 2, 3], [3, 2.5, 2]]),
        ctrl_data=np.array([1, 2, 3]),
        num_samples=5,
    )

    #: generate dummy simulation data
    sim_data = []
    for i in range(system_cls.num_samples):
        sim_data.append(np.arange(i, i + 6).reshape(2, 3))
    system_cls.set_sim_data(sim_data)

    #: generate dummy covariance matrices
    cov_matrices = np.repeat([np.diag([1, 2])], 3, axis=0) * 100

    #: get likelihoods
    likelihoods = smc_cls.get_likelihoods(system_cls, cov_matrices)

    #: assert shape is (num_steps,num_samples)
    assert likelihoods.shape == (3, 5)

    #: assert that the likelihood is calculated correctly
    np.testing.assert_array_almost_equal(
        likelihoods,
        [
            [0.20496576, 0.20547881, 0.20292631, 0.19742187, 0.18920724],
            [0.2079901, 0.20695275, 0.20285481, 0.1958777, 0.18632463],
            [0.2110356, 0.20841408, 0.20276078, 0.194324, 0.18346554]
        ],
        0.00001,
    )


def test_get_posterior():
    """Test to see if posterior is generated as expected"""
    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)

    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=np.array([[1, 3, 2, 4], [7, 6, 5, 8]]),
        ctrl_data=np.array([1, 2, 3, 4]),
        num_samples=5,
    )

    #: generate dummy simulation data
    sim_data = []
    for i in range(system_cls.num_samples):
        sim_data.append(np.arange(i, i + 8).reshape(2, 4))
    system_cls.set_sim_data(sim_data)

    #: generate dummy covariance matrices
    cov_matrices = np.repeat([np.diag([1, 2])], 4, axis=0) * 5

    #: get likelihoods
    likelihoods = smc_cls.get_likelihoods(system_cls, cov_matrices)

    #: get posteriors
    posteriors = smc_cls.get_posteriors(system=system_cls, likelihoods=likelihoods, proposal=None)

    #: assert shape is (num_steps,num_samples)
    assert posteriors.shape == (4, 5)

    #: assert that the posterior is calculated correctly with no proposal distribution
    np.testing.assert_array_almost_equal(
        posteriors,
        [
            [0.17412361, 0.24709317, 0.2597619, 0.20230277, 0.11671855],
            [0.1420761, 0.28610613, 0.31619618, 0.19178268, 0.06383891],
            [0.25941171, 0.40683854, 0.25941171, 0.06724996, 0.00708809],
            [0.25027233, 0.45602592, 0.25027233, 0.04136974, 0.00205968]
        ],
        0.00001,
    )

    #: assert that the posterior is calculated correctly with a proposal distribution
    proposal = np.array([1, 2, 3, 2, 1], dtype=float)
    proposal /= proposal.sum()
    posteriors = smc_cls.get_posteriors(system=system_cls, likelihoods=likelihoods, proposal=proposal)
    np.testing.assert_array_almost_equal(
        posteriors,
        [
            [0.28918067, 0.20518345, 0.14380229, 0.16799, 0.1938436],
            [0.258199, 0.25997447, 0.19154415, 0.1742661, 0.11601629],
            [0.43966996, 0.34476988, 0.14655665, 0.05699008, 0.01201342],
            [0.42821566, 0.3901299, 0.14273855, 0.03539179, 0.00352411]
        ],
        0.00001,
    )


def test_compute_effective_sample_size():
    """Test to see if effective sample size is computed as expected"""
    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)

    system_cls = DynamicSystem(
        param_min=[1, 2],
        param_max=[3, 4],
        obs_data=np.array([[1.2, 3.3, 2.5, 4.5], [7.6, 6.7, 5.8, 8.9]]),
        ctrl_data=np.array([1, 2, 3, 4]),
        num_samples=10,
    )

    #: generate dummy simulation data
    sim_data = []
    for i in range(system_cls.num_samples):
        sim_data.append(np.arange(i, i + 8).reshape(2, 4))
    system_cls.set_sim_data(sim_data)

    #: generate dummy covariance matrices
    cov_matrices = np.repeat([np.diag([1, 2])], 4, axis=0) * 5

    #: get likelihoods
    likelihoods = smc_cls.get_likelihoods(system_cls, cov_matrices)

    #: get posteriors
    smc_cls.posteriors = smc_cls.get_posteriors(system=system_cls, likelihoods=likelihoods, proposal=None)

    #: compute effective sample size
    smc_cls.compute_effective_sample_size()

    #: assert that the effective sample size is computed correctly
    np.testing.assert_array_almost_equal(
        smc_cls.ess,
        [
            [0.55760381],
            [0.43902564],
            [0.35468171],
            [0.31708364]
        ],
        0.00001,
    )


def test_estimated_params():
    """Test to see if estimated_params is generated as expected."""

    smc_cls = SMC(ess_target=0.1, scale_cov_with_max=True)

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
    system_cls.compute_estimated_params(posteriors)

    np.testing.assert_array_almost_equal(
        system_cls.estimated_params,
        [
            [4.8, 5.02222222],
            [4.5, 3.75555556],
            [4., 4.4],
        ],
    )

    np.testing.assert_array_almost_equal(
        system_cls.estimated_params_cv,
        [
            [0.4145781, 0.3729435],
            [0.51759176, 0.51966342],
            [0.48733972, 0.45218241],
        ],
    )
