import numpy as np
from typing import Type, List
from .dynamic_systems import DynamicSystem
from .inference import SMC
from .sampling import GaussianMixtureModel, generate_params_qmc
from scipy import optimize


class IterativeBayesianFilter:
    """Iterative Bayesian Filter class.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        ibf_cls = IterativeBayesianFilter.from_dict(
            {
                "inference":{
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

        system_cls = IterativeBayesianFilter(
                inference = SMC(...),
                sampling = GaussianMixtureModel(...)
        )

    :param inference: Sequential Monte Carlo class (SMC)
    :param sampling: Gaussian Mixture Model class (GMM)
    :param initial_sampling: the initial sampling method, defaults to Halton
    :param ess_tol: Tolerance for the effective sample size to converge, defaults to 1.0e-2
    :param proposal_ibf: User defined proposal distribution for the data assimilation loop, defaults to None
    :param proposal_data_file: Pickle that stores the previously trained posterior distribution
    """

    #: The inference class is a member variable of the particle filter which is used to generate the likelihood
    inference = Type["SMC"]

    #: The gaussian mixture model class is used to sample the parameters
    sampling = Type["GaussianMixtureModel"]

    #: list of parameter data of shape (num_samples, num_params) from all iterations
    param_data_list: List = []

    #: List of sigmas
    sigma_list: List = []

    #: This a tolerance to which the optimization algorithm converges.
    ess_tol: float = 1.0e-2

    #: The non-informative distribution to draw the initial samples
    initial_sampling: str = "halton"

    #: this is the current proposal distribution
    posterior_ibf: np.ndarray

    proposal_ibf: np.ndarray

    proposal_data_file: str

    def __init__(
        self,
        inference: Type["SMC"],
        sampling: Type["GaussianMixtureModel"],
        ess_tol: float = 1.0e-2,
        initial_sampling: str = 'halton',
        proposal_ibf: np.ndarray = None,
        proposal_data_file: str = None,
    ):
        """Initialize the Iterative Bayesian Filter."""
        self.inference = inference
        self.initial_sampling = initial_sampling
        self.sampling = sampling
        self.ess_tol = ess_tol
        self.proposal_ibf = proposal_ibf
        self.proposal_data_file = proposal_data_file

    @classmethod
    def from_dict(cls: Type["IterativeBayesianFilter"], obj: dict):
        """Initialize the class using a dictionary style"""
        return cls(
            inference=SMC.from_dict(obj["inference"]),
            sampling=GaussianMixtureModel.from_dict(obj["sampling"]),
            ess_tol=obj.get("ess_tol", 1.0e-2),
            initial_sampling=obj.get("initial_sampling", "halton"),
            proposal_ibf=obj.get("proposal_ibf", None),
            proposal_data_file=obj.get("proposal_data_file", None),
        )

    def run_inference(self, system: Type["DynamicSystem"]):
        """Run inference (e.g, data assimilation loop) using the Sequential Monte Carlo

        :param system: Dynamic system class
        """
        result = optimize.minimize_scalar(
            self.inference.data_assimilation_loop,
            args=(system, self.proposal_ibf),
            method="bounded",
            bounds=(system.sigma_min, system.sigma_max),
        )
        system.sigma_max = result.x

        # if the name of proposal data file is given, make use of the proposal density during Bayesian updating
        if self.proposal_data_file is not None and self.proposal_ibf is None:
            self.load_proposal_from_file(system)

        if system.sigma_max > system.sigma_tol:
            self.inference.data_assimilation_loop(system.sigma_max, system, self.proposal_ibf)
        else:
            self.inference.data_assimilation_loop(system.sigma_tol, system, self.proposal_ibf)

        self.posterior_ibf = self.inference.give_posterior()

    def initialize(self, system: Type["DynamicSystem"]):
        """Resample the parameters using the Gaussian mixture model

        :param system: Dynamic system class
        """
        system.param_data = generate_params_qmc(system, system.num_samples, self.initial_sampling)
        self.param_data_list.append(system.param_data)

    def run_sampling(self, system: Type["DynamicSystem"]):
        """Resample the parameters using the Gaussian mixture model

        :param system: Dynamic system class
        """
        self.param_data_list.append(self.sampling.regenerate_params(self.posterior_ibf, system))
        # self.param_data_list.append(self.sampling.regenerate_params_with_gmm(self.posterior_ibf, system))

    def solve(self, system: Type["DynamicSystem"]):
        """Run both inference and sampling for a dynamic system

        :param system: Dynamic system class
        """
        self.run_inference(system)
        self.run_sampling(system)

    def add_curr_param_data_to_list(self, param_data: np.ndarray):
        self.param_data_list.append(param_data)

    def load_proposal_from_file(self, system: Type["DynamicSystem"]):
        if system.param_data is None:
            RuntimeError("parameter samples not yet loaded...")

        if self.proposal_data_file is None: return

        from .tools import voronoi_vols
        from pickle import load
        param_max, gmm = load(open(system.sim_data_dir + '/' + self.proposal_data_file, 'rb'), encoding='latin1')
        samples = np.copy(system.param_data)
        samples /= param_max

        proposal = np.exp(gmm.score_samples(samples))
        proposal *= voronoi_vols(samples)
        # assign the maximum vol to open regions (use a uniform proposal distribution if Voronoi fails)
        if (proposal < 0.0).all():
            self.proposal_ibf = np.ones(proposal.shape) / system.num_samples
        else:
            proposal[np.where(proposal < 0.0)] = min(proposal[np.where(proposal > 0.0)])
            self.proposal_ibf = proposal / sum(proposal)