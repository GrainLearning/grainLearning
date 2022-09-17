# %%

import numpy as np
from typing import Type

from .models import Model

from .sequentialmontecarlo import SequentialMonteCarlo

from .gaussianmixturemodel import GaussianMixtureModel

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
        
        model_cls = IterativeBayesianFilter(
                inference = SequentialMonteCarlo(...),
                sampling = GaussianMixtureModel(...)
        )

    :param inference: Sequential Monte Carlo class (SMC)
    :param sampling: Gaussian Mixture Model class (GMM)
    :param num_samples: Number of samples within a user model
    :param ess_tol: Tolerance for the effective sample size to converge, defaults to 1.0e-2
    :param proposal_ibf: User defined proposal distribution for the data assimilation loop, defaults to None
    """

    #: The inference class is a member variable of the particle filter which is used to generate the likelihood
    inference = Type["SequentialMonteCarlo"]

    #: The gaussian mixture model class is used to sample the parameters
    sampling = Type["GaussianMixtureModel"]

    #: This a tolerance to which the optimization algorithm converges.
    ess_tol: float = 1.0e-2

    #: this is the current proposal distribution
    posterior_ibf: np.ndarray

    proposal_ibf: np.ndarray

    def __init__(
            self,
            inference: Type["SequentialMonteCarlo"],
            sampling: Type["GaussianMixtureModel"],
            ess_tol: float = 1.0e-2,
            proposal_ibf: np.ndarray = None,
    ):
        """Initialize the Iterative Bayesian Filter."""
        self.inference = inference
        self.sampling = sampling
        self.ess_tol = ess_tol
        self.proposal_ibf = proposal_ibf

    @classmethod
    def from_dict(cls: Type["IterativeBayesianFilter"], obj: dict):
        """Initialize the class using a dictionary style"""
        return cls(
            inference=SequentialMonteCarlo.from_dict(obj["inference"]),
            sampling=GaussianMixtureModel.from_dict(obj["sampling"]),
            ess_tol=obj.get("ess_tol", 1.0e-2),
            proposal_ibf=obj.get("ess_tol", None),
        )

    def run_inference(self, model: Type["Model"]):
        """Run inference (e.g, data assimilation loop) using the Sequential Monte Carlo

        :param model: Model class
        """
        result = optimize.minimize_scalar(
            self.inference.data_assimilation_loop,
            args=(model, self.proposal_ibf),
            method="bounded",
            bounds=(model.sigma_min, model.sigma_max),
        )
        model.sigma_max = result.x

        # make sure values are set
        self.inference.data_assimilation_loop(model.sigma_max, model, self.proposal_ibf)

        self.posterior_ibf = self.inference.give_posterior()

    def run_sampling(self, model: Type["Model"]):
        """Resample the parameters using the Gaussian mixture model

        :param model: Model class
        """
        model.param_data = self.sampling.regenerate_params(self.posterior_ibf, model)

    def solve(self, model: Type["Model"]):
        """Run both inference and sampling on a model

        :param model: Model class
        """
        self.run_inference(model)
        self.run_sampling(model)
