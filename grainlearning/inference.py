"""
This module contains various methods for performing statistical and Bayesian inference
"""
from typing import Type
import numpy as np
from scipy.stats import multivariate_normal
from grainlearning.dynamic_systems import DynamicSystem


class SMC:
    """This is the Sequential Monte Carlo class that recursively
    update model states based on sequential observations using the Bayes' theorem

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        system_cls = SMC.from_dict(
            {
                "ess_target": 0.3,
                "scale_cov_with_max": False
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        system_cls = SMC(
                ess_target = 0.3,
                scale_cov_with_max = False
        )

    :param ess_target: Target effective sample size (w_0 / sum_i w_i^2)
        where w_0 = 1 / N_p, defaults to 0.3
    :param scale_cov_with_max: True if the covariance matrix is scaled
        with the maxima of the observations, defaults to True
    :param cov_matrices: Covariance matrices of shape (num_steps, num_obs, num_obs),
        defaults to None, Optional
    :param likelihoods: Likelihood distributions of shape (num_steps, num_samples)
    :param posteriors: Posterior distributions of shape (num_steps, num_samples)
    :param ess: Time evolution of the effective sample size
    """
    def __init__(
        self,
        ess_target: float = 0.3,
        scale_cov_with_max: bool = False,
        cov_matrices: np.array = None,
        init_inference_step: int = 0,
    ):
        """Initialize the SMC class"""
        self.ess_target = ess_target

        self.scale_cov_with_max = scale_cov_with_max

        self.cov_matrices = cov_matrices

        self.init_inference_step = init_inference_step

        self.likelihoods = None

        self.posteriors = None

        self.ess = None

    @classmethod
    def from_dict(cls: Type["SMC"], obj: dict):
        """Initialize the class using a dictionary style

        :param obj: a dictionary containing the keys and values to construct an SMC object
        :return: an SMC object
        """
        return cls(
            ess_target=obj.get("ess_target", 0.3),
            scale_cov_with_max=obj.get("scale_cov_with_max", False),
            cov_matrices=obj.get("cov_matrices", None),
            init_inference_step=obj.get("init_inference_step", 0),
        )

    def get_covariance_matrices(self, sigma: float, system: Type["DynamicSystem"]):
        """Compute the (diagonal) covariance matrices from an uncertainty that is either
        assumed by the user or updated by SMC to satisfy the target effective sample size.

        This function is vectorized for all time steps

        :param sigma: Uncertainty
        :param system: Dynamic system
        :return: Covariance matrices for all time steps
        """
        cov_matrix = sigma * system.get_inv_normalized_sigma()

        # duplicated the covariant matrix in time
        cov_matrices = cov_matrix[None, :].repeat(system.num_steps, axis=0)

        if self.scale_cov_with_max:
            # scale with the maxima of the observations
            cov_matrices *= system.obs_data.max(axis=1)[:, None] ** 2
        else:
            # or element wise multiplication of covariant matrix with observables of all time steps
            cov_matrices *= system.obs_data.T[:, None] ** 2

        return cov_matrices

    def get_likelihoods(self, system: Type["DynamicSystem"], cov_matrices: np.array):
        """Compute the likelihood distributions of simulation data as a multivariate normal
        centered around the observation.

        This function is vectorized for all time steps

        :param system: Dynamic system class
        :param cov_matrices: Covariance matrices of shape (num_steps, num_obs, num_obs)
        :return: Likelihood matrices of shape (num_steps, num_samples) considering all time steps
        """
        likelihoods = np.ones((system.num_steps, system.num_samples)) / system.num_samples

        for stp_id in range(self.init_inference_step, system.num_steps):
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
            likelihood = multivariate_normal.pdf(
                system.sim_data[:, :, stp_id] * system.normalization_factor[stp_id],
                mean=system.obs_data[:, stp_id] * system.normalization_factor[stp_id],
                cov=cov_matrices[stp_id] * system.normalization_factor[stp_id] ** 2
            )
            likelihoods[stp_id, :] = likelihood / likelihood.sum()

        return likelihoods

    def get_posteriors(self, system: Type["DynamicSystem"], likelihoods: np.array, proposal: np.array = None):
        """Compute the posterior distributions from the likelihood for all the time steps

        This function is vectorized for all time steps

        :param system: Dynamic system class
        :param likelihoods: Likelihood matrices of shape (num_steps, num_samples)
        :param proposal: Proposal distribution at the initial time step, defaults to None, optional
        :return: Posterior matrices of shape (num_steps, num_samples) considering all time steps
        """
        posteriors = np.zeros((system.num_steps, system.num_samples))

        if proposal is None:
            proposal = np.ones([system.num_samples]) / system.num_samples

        posteriors[0, :] = likelihoods[0, :] / proposal
        posteriors[0, :] /= posteriors[0, :].sum()

        for stp_id in range(1, system.num_steps):
            posteriors[stp_id, :] = posteriors[stp_id - 1, :] * likelihoods[stp_id, :]
            posteriors[stp_id, :] /= posteriors[stp_id, :].sum()

        return posteriors

    def get_posterior_at_time(self, time_step: int = -1):
        """Get the posterior distribution at a certain time step

        :param time_step: input time step, defaults to -1 (the last step in time), optional
        :return: Posterior distribution at a certain time step
        """
        return self.posteriors[time_step, :]

    def data_assimilation_loop(self, sigma: float, system: Type["DynamicSystem"], proposal: np.ndarray = None):
        """Perform data assimilation loop

        :param sigma: Uncertainty
        :param proposal: Proposal distribution from which the samples are sampled from, defaults to None, optional
        :param system: Dynamic system class
        :return: Result of the objective function which converges to a user defined effective sample size
        """
        self.cov_matrices = self.get_covariance_matrices(
            sigma=sigma, system=system
        )
        self.likelihoods = self.get_likelihoods(
            system=system, cov_matrices=self.cov_matrices
        )

        self.posteriors = self.get_posteriors(system=system, likelihoods=self.likelihoods, proposal=proposal)

        self.compute_effective_sample_size()
        return (self.ess[-1] - self.ess_target) ** 2

    def set_posteriors(self, posteriors: np.ndarray = None):
        """Set the posterior distribution"""
        self.posteriors = posteriors

    def compute_effective_sample_size(self):
        """Compute the effective sample size"""
        # compute the effective sample size for every time step
        num_steps, num_samples = self.posteriors.shape
        ess = 1.0 / np.sum(self.posteriors ** 2, axis=1)
        ess /= num_samples
        ess = ess.reshape(num_steps, 1)
        self.ess = ess
