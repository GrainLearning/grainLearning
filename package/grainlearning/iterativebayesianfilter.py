# # %%

import numpy as np
from typing import Type


from .models import Model

from .sequentialmontecarlo import SequentialMonteCarlo

from .gaussianmixturemodel import GaussianMixtureModel

#  TODO add .from_dict class


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
    proposal_ibf: np.ndarray

    #: This is the minimum value of sigma which is automatically adjusted such that to covariance matrix is not too big.
    sigma_min: float = 1.0e-6

    #: This is the maximum value of sigma which is automatically adjusted such that the covariance matrix is not singular
    sigma_max: float = 1.0e6

    def __init__(
        self,
        inference: Type["SequentialMonteCarlo"],
        sampling: Type["GaussianMixtureModel"],
        sigma_max: float = 1.0e6,
        sigma_min: float = 1.0e-6,
        ess_tol: float = 1.0e-2,
    ):
        """Initialize the Iterative Bayesian Filter class


        """
        self.inference = inference
        self.sampling = sampling
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.ess_tol = ess_tol
        self.proposal_ibf = None

    @classmethod
    def from_dict(cls: Type["IterativeBayesianFilter"], obj: dict):
        return cls(
            inference=SequentialMonteCarlo.from_dict(obj["inference"]),
            sigma_max=obj.get("sigma_max", 1.0e6),
            sigma_min=obj.get("sigma_min", 1.0e-6),
            ess_tol=obj.get("ess_tol", 1.0e-2),
            sampling=GaussianMixtureModel.from_dict(obj["sampling"]),
        )

    def set_proposal(self, model: Type["Model"]):
        """set the proposal distribution iterative bayesian filter

        :param model: Calibration model.
        :param input_proposal: initial proposal distribution, defaults to None
        """
        self.proposal_ibf = np.ones([model.num_samples]) / model.num_samples

    def check_sigma_bounds(self, sigma_adjust: float, model: Type["Model"]):
        
        sigma_new = sigma_adjust

        while True:
            cov_matrices = self.inference.get_covariance_matrices(sigma_new, model)
            
            # get determinant of all covariant matricies
            det_all = np.linalg.det(cov_matrices)

            # if all is above threshold, decrease sigma
            if (det_all > 1e16).all():
                sigma_new *= 0.75
                continue

            # if all is below threshold, increase sigma
            if (det_all < 1e1).all():
                sigma_new *= 1.25
                continue

            break

        return sigma_new

    def configure(self, model: Type["Model"]):

        self.set_proposal(model=model)

        self.sigma_min = self.check_sigma_bounds(
            sigma_adjust=self.sigma_min, model=model
        )
        self.sigma_max = self.check_sigma_bounds(
            sigma_adjust=self.sigma_max, model=model
        )

    def run_inference(self, model: Type["Model"]):

        from scipy import optimize
        
        result = optimize.minimize_scalar(
            self.inference.data_assimilation_loop,
            args=(self.proposal_ibf, model),
            method="bounded",
            bounds=(self.sigma_min, self.sigma_max),
        )
        self.sigma_max = result.x

        # make sure values are set
        self.inference.data_assimilation_loop(
            result.x, self.proposal_ibf, model
        )
        self.proposal_ibf = self.inference.give_proposal()

        # print(self.sigma_min,self.sigma_max,self.proposal_ibf,result.x)


    def run_sampling(self, model: Type["Model"]):
        new_params = self.sampling.regenerate_params(self.proposal_ibf, model)
        return new_params

    def solve(
        self, model: Type["Model"]
    ) -> np.ndarray:
        self.run_inference(model)
        new_params = self.run_sampling(model)
        return new_params
