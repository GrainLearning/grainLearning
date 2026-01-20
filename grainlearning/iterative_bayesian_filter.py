"""
This module contains various Bayesian filtering classes by mixing the inference and sampling methods
from the :mod:`.inference` and :mod:`.sampling` modules.
"""
from typing import Type, List
from pickle import load
import numpy as np
from scipy import optimize
from grainlearning.dynamic_systems import DynamicSystem, IODynamicSystem
from grainlearning.inference import SMC
from grainlearning.sampling import GaussianMixtureModel, generate_params_qmc
from grainlearning.tools import voronoi_vols


class IterativeBayesianFilter:
    """This is the Iterative Bayesian Filter class.

    The idea is to solve an inverse problem repeatedly by refining the previous knowledge of the dynamical system.
    An `iterative Bayesian filter <https://doi.org/10.1016/j.cma.2019.01.027>`_ is essentially
    a sequential Monte Carlo filter that iteratively samples the parameter space, using a proposal distribution
    trained with non-Gaussian mixture models.
    It consists of the inference class and the (re)sampling class.
    The inference class is a sequential Monte Carlo filter, and the resampling class is a Gaussian mixture model.
    The Gaussian mixture model is trained with the previous ensemble (i.e., samples and associated weights).


    The steps of the iterative Bayesian filter are as follows:

    1. :meth:`Initialize <.IterativeBayesianFilter.initialize>` the ensemble of model (including parameter) states using a low-discrepancy sequence.

    2. :meth:`Compute the posterior distribution <.IterativeBayesianFilter.run_inference>` of model states such that the target effective sample size is reached.

    3. :meth:`Generate new samples <.IterativeBayesianFilter.run_sampling>` from a proposal density that is trained with the previous ensemble (i.e., samples and associated weights).

    4. Run the model with the new samples and update the ensemble of model states using :meth:`.DynamicSystem.run` and :meth:`.BayesianCalibration.load_system`.

    5. Repeat steps 2-4 until convergence. The iterations are done in the Bayesian Calibration class by calling :meth:`.BayesianCalibration.run_one_iteration`.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        ibf_cls = IterativeBayesianFilter.from_dict(
            {
                "Bayes_filter":{
                    "ess_target": 0.3,
                    "scale_cov_with_max": True
                },
                "sampling":{
                    "max_num_components": 2
                }
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        ibf_cls = IterativeBayesianFilter(
                Bayes_filter = SMC(...),
                sampling = GaussianMixtureModel(...)
        )

    :param Bayes_filter: A Bayesian filtering algorithm. Currently, only the sequential Monte Carlo class (SMC) is available
    :param sampling: Gaussian Mixture Model class (GMM)
    :param initial_sampling: The initial sampling method, defaults to Halton
    :param ess_tol: Tolerance for the target effective sample size to converge, defaults to 1.0e-2
    :param proposal: A proposal distribution to sample the state-parameter space, defaults to None
    :param proposal_data_file: Pickle that stores the previously trained proposal distribution, defaults to None
    :param param_data_list: List of the parameter samples of shape (num_samples, num_params) generated in all iterations
    :param sigma_list: List of sigma values optimized to satisfy the target effective sample size in all iterations
    :param posterior: The posterior distribution of model states at the last time step
    """

    def __init__(
        self,
        Bayes_filter: Type["SMC"] = None,
        sampling: Type["GaussianMixtureModel"] = None,
        ess_tol: float = 1.0e-2,
        initial_sampling: str = 'halton',
        proposal: np.ndarray = None,
        proposal_data_file: str = None,
    ):
        """Initialize the Iterative Bayesian Filter."""

        self.Bayes_filter = Bayes_filter

        self.initial_sampling = initial_sampling

        self.sampling = sampling

        self.ess_tol = ess_tol

        self.proposal = proposal

        self.proposal_data_file = proposal_data_file

        self.param_data_list = []

        self.sigma_list = []

        self.posterior = None

    @classmethod
    def from_dict(cls: Type["IterativeBayesianFilter"], obj: dict):
        """Initialize the class using a dictionary style

        :param obj: a dictionary containing the keys and values to construct an iBF object
        :return: an iBF object
        """
        return cls(
            Bayes_filter=SMC.from_dict(obj["Bayes_filter"]),
            sampling=GaussianMixtureModel.from_dict(obj["sampling"]),
            ess_tol=obj.get("ess_tol", 1.0e-2),
            initial_sampling=obj.get("initial_sampling", "halton"),
            proposal=obj.get("proposal", None),
            proposal_data_file=obj.get("proposal_data_file", None),
        )

    def initialize(self, system: Type["DynamicSystem"]):
        """Initialize the ensemble of model (including parameter) states using a low-discrepancy sequence.

        :param system: Dynamic system class
        """
        system.param_data = generate_params_qmc(system, system.num_samples, self.initial_sampling)
        self.param_data_list.append(system.param_data)

    def run_inference(self, system: Type["DynamicSystem"]):
        """Compute the posterior distribution of model states such that the target effective sample size is reached.

        :param system: Dynamic system class
        """
        # if the name of proposal data file is given, make use of the proposal density during Bayesian updating
        if self.proposal_data_file is not None and self.proposal is None:
            self.load_proposal_from_file(system)

        result = optimize.minimize_scalar(
            self.Bayes_filter.data_assimilation_loop,
            args=(system, self.proposal),
            method="bounded",
            options={"xatol": 0.01},
            bounds=(system.sigma_lower_bound, system.sigma_max),
        )
        system.sigma_max = result.x

        # use the optimized sigma value to compute the posterior distribution
        if system.sigma_max > system.sigma_tol:
            self.Bayes_filter.data_assimilation_loop(system.sigma_max, system, self.proposal)
        else:
            self.Bayes_filter.data_assimilation_loop(system.sigma_tol, system, self.proposal)

        # get the posterior distribution at the last time step
        self.posterior = self.Bayes_filter.get_posterior_at_time(-1)

        # compute the estimated means and coefficient of variation from the posterior distribution
        system.compute_estimated_params(self.Bayes_filter.posteriors)

    def run_sampling(self, system: Type["DynamicSystem"]):
        """Generate new samples from a proposal density.

        The proposal density can be trained with the previous ensemble (i.e., samples and associated weights).

        :param system: Dynamic system class
        """
        self.param_data_list.append(self.sampling.regenerate_params(self.posterior, system))
        # self.param_data_list.append(self.sampling.regenerate_params_with_gmm(self.posterior, system))

    def solve(self, system: Type["DynamicSystem"]):
        """Run both inference and sampling for a dynamic system

        The iterations are done at the level of the Bayesian Calibration class
        by calling :meth:`.BayesianCalibration.run_one_iteration`.

        :param system: Dynamic system class
        """
        self.run_inference(system)
        self.run_sampling(system)

    def add_curr_param_data_to_list(self, param_data: np.ndarray):
        """Add the current parameter samples to the list of parameter samples.

        :param param_data: Current parameter samples of shape (num_samples, num_params)
        """
        self.param_data_list.append(param_data)

    def load_proposal_from_file(self, system: Type["IODynamicSystem"]):
        """Load the proposal density from a file.

        :param system: Dynamic system class
        """
        if system.param_data is None:
            raise RuntimeError("parameter samples not yet loaded...")

        if self.proposal_data_file is None:
            return

        # load the proposal density from a file
        self.sampling.load_gmm_from_file(f'{system.sim_data_dir}/iter{system.curr_iter-1}/{self.proposal_data_file}')

        samples = np.copy(system.param_data)

        # normalize the parameter samples
        samples_normalized = self.sampling.normalize(samples)

        self.proposal = np.exp(self.sampling.gmm.score_samples(samples_normalized))

        self.proposal /= self.proposal.sum()

        ## turn off the proposal density to probability mass correction because quasi-random sampling is used by default
        # proposal *= voronoi_vols(samples)

        # # assign the maximum vol to open regions (use a uniform proposal distribution if Voronoi fails)
        # if (proposal < 0.0).all():
        #     self.proposal = np.ones(proposal.shape) / system.num_samples
        # else:
        #     proposal[np.where(proposal < 0.0)] = min(proposal[np.where(proposal > 0.0)])
        #     self.proposal = proposal / sum(proposal)

    def save_proposal_to_file(self, system: Type["IODynamicSystem"]):
        """Save the proposal density to a file.

        :param system: Dynamic system class
        """
        self.sampling.save_gmm_to_file(f'{system.sim_data_dir}/iter{system.curr_iter-1}/{self.proposal_data_file}')
