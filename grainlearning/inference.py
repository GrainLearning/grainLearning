from typing import Type
import numpy as np

from .models import Model

from scipy.stats import multivariate_normal


class SMC:
    """This is the Sequential Monte Carlo class that recursively
    update the model state and model parameters based on Bayes' theorem

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        model_cls = SMC.from_dict(
            {
                "ess_target": 0.3,
                "scale_cov_with_max": True
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        model_cls = SMC(
                ess_target = 0.3,
                scale_cov_with_max = True
        )

    :param ess_target: Target effective sample size (what we want)
    :param scale_cov_with_max: Flag if the covariance matrix should scale with the maximum accross the loading steps, defaults to True
    """

    #: Target effective sample size.
    ess_target: float

    #: Flag if the covariance matrix should be scaled with the maximum values of the observations
    scale_cov_with_max: bool = True

    #: Covariance matricies of shape (num_steps,num_obs,num_obs)
    cov_matrices: np.array

    #: Likelihoods of shape (num_steps, num_samples)
    likelihoods: np.array

    #: Posteriors of shape (num_steps, num_samples)
    posteriors: np.array

    #: Array containing ips of (num_steps, num_params)
    ips: np.array

    #: Array containing covs of (num_steps, num_params)
    covs: np.array

    #: Calculated effective sample size
    ess: float

    def __init__(
        self,
        ess_target: float,
        scale_cov_with_max: bool = True,
    ):
        """Initialize the variables."""
        self.ess_target = ess_target
        self.scale_cov_with_max = scale_cov_with_max

    @classmethod
    def from_dict(cls: Type["SMC"], obj: dict):
        """Initialize the class using a dictionary style"""
        return cls(
            ess_target=obj["ess_target"],
            scale_cov_with_max=obj.get("scale_cov_with_max", True),
        )

    def get_covariance_matrices(
        self, sigma_guess: float, model: Type["Model"]
    ) -> np.array:
        """Compute the diagonal covariance matrices from a given input sigma.

        This function is vectorized for all loading steps

        :param sigma_guess: input sigma
        :param model: Model class
        :return: Covariance matrices for all loading steps
        """
        cov_matrix = sigma_guess * model._inv_normalized_sigma

        # duplicated covariant matrix to loading step
        cov_matrices = cov_matrix[None, :].repeat(model.num_steps, axis=0)

        # scale with the maximum of the loading steps
        if self.scale_cov_with_max:
            cov_matrices *= model.obs_data.max(axis=1)[:, None] ** 2
        else:
            # or element wise multiplication of covariant matrix with observables of all loading steps
            cov_matrices *= model.obs_data.T[:, None] ** 2

        return cov_matrices

    def get_likelihoods(self, model: Type["Model"], cov_matrices: np.array) -> np.array:
        """Compute the likelihoods as a multivariate normal of the simulation data centered around the observations.

        This function is vectorized for all loading steps

        :param model: Model class
        :param cov_matrices: covariance matricies
        :return: Likelihood matrices for all loading steps
        """
        likelihoods = np.zeros((model.num_steps, model.num_samples))

        for stp_id in range(model.num_steps):
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
            likelihood = multivariate_normal.pdf(
                model.sim_data[:, :, stp_id],
                mean=model.obs_data[:, stp_id],
                cov=cov_matrices[stp_id],
            )
            likelihoods[stp_id, :] = likelihood / likelihood.sum()

        return likelihoods

    def get_posterors(
        self, model: Type["Model"], likelihoods: np.array, proposal_ibf: np.array = None
    ) -> np.array:
        """Compute the posteriors for all the loading steps

        This function is vectorized for all loading steps

        :param model: Model class
        :param likelihoods: Likelihood matrices
        :param proposal_ibf: Optional input proposal
        :return: Posterior distributions for all loading steps
        """
        posteriors = np.zeros((model.num_steps, model.num_samples))

        if proposal_ibf is None:
            proposal = np.ones([model.num_samples]) / model.num_samples
        else:
            proposal = proposal_ibf

        posteriors[0, :] = likelihoods[0, :] / proposal
        posteriors[0, :] /= posteriors[0, :].sum()

        for stp_id in range(1, model.num_steps):
            posteriors[stp_id, :] = posteriors[stp_id - 1, :] * likelihoods[stp_id, :]
            posteriors[stp_id, :] /= posteriors[stp_id, :].sum()

        return posteriors

    def get_ensemble_ips_covs(
        self,
        model: Type["Model"],
        posteriors: np.array,
    ) -> np.array:
        """Compute the ensemble averages for parameters. (Used for post-processing)

        This function is vectorized for all loading steps

        :param model: Model class
        :param posteriors: Posterior distributions
        :return: Ensemble averages
        """
        ips = np.zeros((model.num_steps, model.num_params))
        covs = np.zeros((model.num_steps, model.num_params))

        for stp_id in range(model.num_steps):
            ips[stp_id, :] = posteriors[stp_id, :] @ model.param_data

            covs[stp_id, :] = (
                posteriors[stp_id, :] @ (ips[stp_id, :] - model.param_data) ** 2
            )

            covs[stp_id, :] = np.sqrt(covs[stp_id, :]) / ips[stp_id, :]

        return ips, covs

    def give_posterior(self, loading_step=-1):
        """Give posterior distribution of a loading step

        :param loading_step: Optional input loading step, defaults to -1 (last value)
        :return: Posterior distribution for a single step
        """
        return self.posteriors[loading_step, :]

    def data_assimilation_loop(
        self, sigma_guess: float, model: Type["Model"], proposal_ibf: np.ndarray = None
    ):
        """Perform data assimilation loop

        :param sigma_guess: Guess of sigma
        :param proposal_ibf: Input distribution
        :param model: Model class
        :return: Result of the objective function which converges to a user defined effective sample size
        """
        self.cov_matrices = self.get_covariance_matrices(
            sigma_guess=sigma_guess, model=model
        )
        self.likelihoods = self.get_likelihoods(
            model=model, cov_matrices=self.cov_matrices
        )

        self.posteriors = self.get_posterors(
            model=model, likelihoods=self.likelihoods, proposal_ibf=proposal_ibf
        )

        self.ips, self.covs = self.get_ensemble_ips_covs(
            model=model, posteriors=self.posteriors
        )

        # TODO: I (Hongyang) would save the whole effective sample size sequence in time.
        #  Examining the evolution of eff gives you a good idea how your filtering algorithm is doing.
        self.ess = 1.0 / np.sum(
            self.posteriors[-1, :] ** 2,
        )

        self.ess /= model.num_samples

        return (self.ess - self.ess_target) ** 2
