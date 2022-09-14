# %%

import numpy as np
from typing import Type

from .models import Model

from .sequentialmontecarlo import SequentialMonteCarlo

from .gaussianmixturemodel import GaussianMixtureModel


class IterativeBayesianFilter:
    """This class contains the iterative bayesian filter algorithm.

    Initialize the module like this:

    .. highlight:: python
    .. code-block:: python

        ibf = IterativeBayesianFilter(
            inference =SequentialMonteCarlo(
                ess_target=0.1, inv_obs_weight=[0.5, 0.25], scale_cov_with_max=True
            ),
            sampling = GaussianMixtureModel(max_num_components=5)
        )

        # after initialization we have to configure the algorithm to some model (see :class:`.Parameters`)
        # this will select an appropriate sigma min and sigma max
        ibf.configure(mymodel)

        # the data assimilation loop can be run like this
        ibf.run_inference(mymodel)



    :param inference: Sequential Monte Carlo class (SMC)
    :param sampling: Gaussian Mixture Model class (GMM)
    :param sigma_max: Initial sigma max (this value gets automatically adjusted), defaults to 1.0e6
    :param sigma_min: Initial sigma min (this value gets automatically adjusted), defaults to 1.0e-6
    :param ess_tol: Tolarance for the effective sample size to converge, defaults to 1.0e-2

    """

    #: The inference class is a member variable of the particle filter which is used to generate the likelihood
    inference = Type["SequentialMonteCarlo"]

    #: The gaussian mixture model class is used to sample the parameters
    sampling = Type["GaussianMixtureModel"]

    #: This a tolarance to which the optimization algorithm converges.
    ess_tol: float = 1.0e-2

    #: this is the current proposal distribution
    posterior_ibf: np.ndarray
    
    proposal_ibf: np.ndarray

    def __init__(
        self,
        inference: Type["SequentialMonteCarlo"],
        sampling: Type["GaussianMixtureModel"],
        num_samples: int,
        ess_tol: float = 1.0e-2,
        proposal_ibf: np.ndarray = None 
    ):
        """Initialize the Iterative Bayesian Filter class"""
        self.inference = inference
        self.sampling = sampling
        self.ess_tol = ess_tol
        self.proposal_ibf = proposal_ibf

    @classmethod
    def from_dict(cls: Type["IterativeBayesianFilter"], obj: dict):
        return cls(
            inference=SequentialMonteCarlo.from_dict(obj["inference"]),
            sampling=GaussianMixtureModel.from_dict(obj["sampling"]),
            num_samples=obj["num_samples"],
            ess_tol=obj.get("ess_tol", 1.0e-2),
            proposal_ibf = obj.get("ess_tol", None),
        )

    def run_inference(self, model: Type["Model"]):

        from scipy import optimize

        result = optimize.minimize_scalar(
            self.inference.data_assimilation_loop,
            args=(self.proposal_ibf, model),
            method="bounded",
            bounds=(model.sigma_min, model.sigma_max),
        )
        model.sigma_max = result.x

        # make sure values are set
        self.inference.data_assimilation_loop(model.sigma_max, self.proposal_ibf, model)
        
        self.posterior_ibf = self.inference.give_posterior()
        
                
        print(self.inference.eff,"eff")

    def run_sampling(self, model: Type["Model"]):
        model.param_data = self.sampling.regenerate_params(self.posterior_ibf, model)

        
    def solve(self, model: Type["Model"]) -> np.ndarray:
        self.run_inference(model)
        self.run_sampling(model)











    # def check_sigma_bounds(self, sigma_adjust: float, model: Type["Model"]):

    #     sigma_new = sigma_adjust

    #     while True:
    #         cov_matrices = self.inference.get_covariance_matrices(sigma_new, model)

    #         # get determinant of all covariant matricies
    #         det_all = np.linalg.det(cov_matrices)

    #         # if all is above threshold, decrease sigma
    #         if (det_all > 1e16).all():
    #             sigma_new *= 0.9
    #             continue

    #         # if all is below threshold, increase sigma
    #         if (det_all < 0.01).all():
    #             sigma_new *= 1.1
    #             continue

    #         break

    #     return sigma_new

    # self.sigma_min = self.check_sigma_bounds(
    #     sigma_adjust=self.sigma_min, model=model
    # )
    # self.sigma_max = self.check_sigma_bounds(
    #     sigma_adjust=self.sigma_max, model=model
    # )
