"""Test the iterative bayesian filter class."""
import os
from shutil import rmtree
import numpy as np
from grainlearning import (
    SMC,
    IterativeBayesianFilter,
    GaussianMixtureModel,
    DynamicSystem,
    IODynamicSystem,
)

PATH = os.path.abspath(os.path.dirname(__file__))


def test_init():
    """Test if the iterative bayesian filter is initialized correctly"""
    #: Create the members of the iterative bayesian filter
    smc_cls = SMC(ess_target=0.1)
    gmm_cls = GaussianMixtureModel(max_num_components=5)
    #: Create the iterative bayesian filter
    ibf_cls = IterativeBayesianFilter(
        Bayes_filter=smc_cls,
        sampling=gmm_cls,
        initial_sampling='halton',
        proposal=np.ones(10),
        proposal_data_file='dumpy_file',
    )

    #: Create the iterative bayesian filter from a dictionary
    ibf_dct = IterativeBayesianFilter.from_dict(
        {
            "Bayes_filter": {"ess_target": 0.1},
            "sampling": {"max_num_components": 5},
            "initial_sampling": 'halton',
            "proposal": np.ones(10),
            "proposal_data_file": 'dumpy_file',
        }
    )

    #: Assert that the object is of the correct type
    assert isinstance(ibf_dct, IterativeBayesianFilter)
    assert isinstance(ibf_dct.sampling, GaussianMixtureModel)
    assert isinstance(ibf_dct.Bayes_filter, SMC)

    raw_ibf_dct = ibf_dct.__dict__
    raw_ibf_cls = ibf_cls.__dict__

    # TODO @Retief: why do we have to remove these member objects for the assert to work?
    raw_ibf_dct.pop("Bayes_filter")
    raw_ibf_dct.pop("sampling")
    raw_ibf_cls.pop("sampling")
    raw_ibf_cls.pop("Bayes_filter")

    #: Assert that the two iterative Bayesian filter objects are equal
    np.testing.assert_equal(raw_ibf_dct, raw_ibf_cls)


def test_initialize():
    """Test if the initial parameter samples are generated correctly"""
    #: Create the iterative bayesian filter from a dictionary
    ibf_dct = IterativeBayesianFilter.from_dict(
        {
            "Bayes_filter": {"ess_target": 0.1},
            "sampling": {"max_num_components": 5},
            "initial_sampling": 'halton'
        }
    )

    system_cls = DynamicSystem.from_dict({
        "param_min": [0, 10],
        "param_max": [10, 100],
        "obs_data": np.random.uniform(0, 100, (3, 4)),
        "num_samples": 10,
        "callback": None,
    })

    #: Generate the initial parameter samples
    ibf_dct.initialize(system_cls)

    #: Assert that the initial parameter samples are generated correctly
    np.testing.assert_array_almost_equal(
        ibf_dct.param_data_list[-1],
        np.array(
            [
                [0., 10.],
                [5., 40.],
                [2.5, 70.],
                [7.5, 20.],
                [1.25, 50.],
                [6.25, 80.],
                [3.75, 30.],
                [8.75, 60.],
                [0.625, 90.],
                [5.625, 13.33333333]
            ]
        ),
    )


def test_run_inference():
    """Test if the inference runs"""
    #: Create a dynamic system from a dictionary.
    # Observed data generated using the following code: np.random.uniform(0, 100, (3, 4)
    obs_data = np.array([[89.38748749, 20.54419146, 57.39546832, 83.0982845],
                         [68.41695997, 30.3873664, 31.68803094, 98.31786969],
                         [76.44687489, 70.60453217, 15.61866939, 71.03609807]])
    system_cls = DynamicSystem.from_dict({
        "param_min": [0, 10],
        "param_max": [10, 100],
        "obs_data": obs_data,
        "num_samples": 3,
        "callback": None,
    })

    #: Assign the simulated data to the dynamic system.
    # Simulation data generated using the following code: np.random.uniform(0, 100, (3, 3, 4))
    system_cls.set_sim_data(np.array([[[8.05930103e+01, 4.26938401e+00, 4.43094496e+01, 1.46923752e+01],
                                       [6.46472684e-01, 1.73509784e+01, 1.22798597e-01, 1.20403173e+01],
                                       [4.24170187e+01, 6.74455857e+01, 2.39582486e+01, 2.70144154e+00]],

                                      [[9.10500197e+01, 4.82299568e+01, 2.15940875e+01, 8.44022906e+01],
                                       [8.18422577e+01, 3.26844757e+01, 5.63683869e+01, 3.06096618e+01],
                                       [1.03812199e+01, 2.57135437e+01, 8.41059191e+01, 8.48553647e+01]],

                                      [[6.51376570e+01, 8.44703569e-01, 9.87061586e+00, 3.61795636e-02],
                                       [1.74846306e+01, 7.97853362e+01, 5.57097840e+01, 7.77250290e+01],
                                       [8.52725537e+01, 5.52691132e+01, 3.15013590e+01, 6.46202944e+01]]]))

    #: Create the iterative bayesian filter from a dictionary
    ibf_cls = IterativeBayesianFilter.from_dict(
        {
            "Bayes_filter": {"ess_target": 0.5, "scale_cov_with_max": True},
            "sampling": {"max_num_components": 5},
            "initial_sampling": 'halton',
        }
    )

    #: Generate the initial parameter samples
    ibf_cls.initialize(system=system_cls)

    #: Run the inference for one iteration
    ibf_cls.run_inference(system=system_cls)

    #: Assert that the inference runs correctly
    np.testing.assert_array_almost_equal(
        ibf_cls.posterior,
        np.array([0.065373, 0.131263, 0.803364]),
    )

    #: Assert that the inference runs correctly if a proposal density is provided
    ibf_cls = IterativeBayesianFilter.from_dict(
        {
            "Bayes_filter": {"ess_target": 0.5, "scale_cov_with_max": True},
            "sampling": {"max_num_components": 5},
            "initial_sampling": 'halton',
            "proposal": np.array([0.5, 0.2, 0.3])
        }
    )

    #: Run the assertion again
    ibf_cls.initialize(system=system_cls)
    ibf_cls.run_inference(system=system_cls)
    #: Assert that the inference runs correctly
    np.testing.assert_array_almost_equal(
        ibf_cls.posterior,
        np.array([0.032105, 0.170057, 0.797837])
    )


def test_save_and_load_proposal():
    from scipy.stats import multivariate_normal
    """Test if the proposal density can be loaded from a file"""
    #: Initialize a system object (note the observed data is not used in this test)
    system_cls = IODynamicSystem(sim_name='test_ibf', sim_data_dir=PATH + '/sim_data/', sim_data_file_ext='.txt',
                                 obs_data_file=os.path.abspath(
                                     os.path.join(__file__, "../..")) + '/data/linear_sim_data/linear_obs.dat',
                                 obs_names=['f'], ctrl_name='u', num_samples=1000, param_min=[6, 0.2],
                                 param_max=[7, 0.5], obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]], ctrl_data=[1, 2, 3, 4],
                                 param_names=['a', 'b'])

    #: Assert that the inference runs correctly if a proposal density is provided
    ibf_cls = IterativeBayesianFilter.from_dict(
        {
            "Bayes_filter": {"ess_target": 1.0},
            "sampling": {"max_num_components": 1, "expand_factor": 100},
            "initial_sampling": "halton",
            "proposal_data_file": "test_proposal.pkl"
        }
    )

    #: Generate the initial parameter samples
    ibf_cls.initialize(system=system_cls)

    #: Set a dummy proposal density
    mean = np.mean(system_cls.param_data, axis=0)
    cov = np.array([[1, 0], [0, 1]]) * mean
    rv = multivariate_normal(mean, cov, allow_singular=True)
    dummy_proposal = rv.pdf(system_cls.param_data)
    dummy_proposal /= np.sum(dummy_proposal)

    #: Train the proposal density
    ibf_cls.sampling.train(dummy_proposal, system_cls)

    #: Correct the scores
    covs_ref = dummy_proposal.dot((system_cls.param_data - dummy_proposal.dot(system_cls.param_data))**2)
    ibf_cls.sampling.scores = ibf_cls.sampling.correct_scores(covs_ref, system_cls, tol=1e-6)

    #: Assert that empirical covariance and the covariance esimated by GMM are the same for one Gaussian component
    empirical_cov = np.cov(ibf_cls.sampling.expanded_normalized_params.T)
    gmm_cov = ibf_cls.sampling.gmm.covariances_
    np.testing.assert_array_almost_equal(empirical_cov, gmm_cov, 0.0001)

    #: Assert if the dummy proposal density is the same as the one trained
    trained_proposal = np.exp(ibf_cls.sampling.scores)
    trained_proposal /= trained_proposal.sum()
    np.testing.assert_allclose(trained_proposal, dummy_proposal, 0.01)

    #: Save the Gaussian Mixture Model to a file
    sim_data_sub_dir = f'{system_cls.sim_data_dir}/iter{-1}'
    if not os.path.exists(sim_data_sub_dir):
        os.makedirs(sim_data_sub_dir)
    ibf_cls.save_proposal_to_file(system_cls)

    #: Load the Gaussian Mixture Model from a file
    ibf_cls.load_proposal_from_file(system_cls)

    #: Check if the loaded proposal density is the same as the original one
    np.testing.assert_allclose(ibf_cls.proposal, dummy_proposal, 0.01)

    #: Delete the temporary file and directory
    rmtree(PATH + '/sim_data/')
