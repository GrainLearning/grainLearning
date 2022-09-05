#%%

from typing import Type, List
import numpy as np

from .models import Model

class SequentialMonteCarlo:
    """Sequential Monte Carlo (SMC) filter class.
    
        :param ess_target: _description_
        :param inv_obs_weight: _description_
        :param scale_cov_with_max: _description_, defaults to True
        
        Example usage:

        .. highlight:: python
        .. code-block:: python

            smc = SequentialMonteCarlo(
                ess_target=0.1, inv_obs_weight=[0.5, 0.25], scale_cov_with_max=True
            )

            # make sure sigma_guess is not very big or not very small. 
            # So that the determinant of covariance matrix should be sufficiently small and not zero.
            # i.e check sigma_guess for some initialized model.
            
            cov_matrices = smc_cls.get_covariance_matrices(sigma_guess, mymodel)
            
            # initialize proposal_prev or use values from previous iteration
            smc = smc.data_assimilation_loop(
                sigma_guess = sigma_guess, proposal_prev= someproposal, model mymodel
            )
            
    """

    #: Targete effective sample size.
    ess_target: float

    #: Inverse observation weights for the covariance matrix.
    inv_obs_weight: List[float]

    #: Flag if the covariance matrix should be scaled with the maximum values of the observations
    scale_cov_with_max: bool = True

    #: calculated normalized sigma to weigh the covariance matrix
    _inv_normalized_sigma: np.array

    #: Numpy array containing the covariance matricies of shape (num_steps,num_obs,num_obs)
    cov_matrices: np.array
    
    #: Numpy array containing data for the likelihoods of shape (num_steps, num_samples)
    likelihoods: np.array

    #: Numpy array containing data for the posteriors of shape (num_steps, num_samples)
    posteriors: np.array
    
    #: Numpy array containing ips of (num_steps, num_params)
    ips: np.array
    
    #: Numpy array containing covs of (num_steps, num_params)
    covs: np.array

    #: The calculated effective sample size
    eff: float


    def __init__(
        self,
        ess_target: float,
        inv_obs_weight: List,
        scale_cov_with_max: bool = True,
    ):
        """Initialize the Sequential Monte Carlo class
        """
        self.ess_target = ess_target
        self.scale_cov_with_max = scale_cov_with_max
        self.inv_obs_weight = inv_obs_weight

        inv_obs_mat = np.diagflat(self.inv_obs_weight)

        self._inv_normalized_sigma = inv_obs_mat * np.linalg.det(inv_obs_mat) ** (
            -1.0 / inv_obs_mat.shape[0]
        )

    @classmethod
    def from_dict(cls: Type["SequentialMonteCarlo"], obj: dict):
        """The class can also be initialized using a dictionary style.

        :param cls: The SequentialMonteCarlo class referenced to itself.
        :param obj: Dictionary containing the input to the object.
        :return: An initialized SequentialMonteCarlo object

        Example usage:

        .. highlight:: python
        .. code-block:: python

            smc = SequentialMonteCarlo.from_dict({
            "data": {"ess_target": 0.2,
                "inv_obs_weight": [0.3,0.7], # a weight per observable
                "scale_cov_with_max": False
            })
            
        """
        return cls(
            ess_target=obj["ess_target"],
            inv_obs_weight=obj.get("inv_obs_weight", None),
            scale_cov_with_max=obj.get("scale_cov_with_max", True),
        )

    def get_covariance_matrices(
        self, sigma_guess: float, model: Type["Model"]
    ) -> np.array:
        """Create a diagonal covariance matrix from a given input sigma.

        :param sigma_guess: input sigma
        :param observations: Observations class
        :param load_step: the load step of the simulation, defaults to 0
        :return: a covariance matrix of shape (num_observables, num_observables)
        """
        cov_matrix = sigma_guess * self._inv_normalized_sigma

        # duplicated covariant matrix to loading step
        cov_matrices = cov_matrix[None, :].repeat(model.observations.num_steps, axis=0)

        if self.scale_cov_with_max:
            cov_matrices *= model.observations.data.max(axis=1)[:, None]
        else:
            # element wise multiplication of covariant matrix with observables of all loading steps
            cov_matrices *= model.observations.data.T[:, None] ** 2

        return cov_matrices

    def get_likelihoods(
        self, model: Type["Model"], cov_matrices: np.array
    ) -> np.array:

        # (num_samples, num_observations,num_steps )
        rel_vectors = (
            np.repeat([model.observations.data], model.num_samples, axis=0) - model.data
        )

        inv_cov_matrices = np.linalg.inv(cov_matrices)

        likelihoods = np.zeros((model.observations.num_steps, model.num_samples))

        for sim_id in range(model.num_samples):
            # create reshape the column vectors into a stack so they can be multiplied by covariance matricies
            vec = rel_vectors[sim_id].reshape(
                model.observations.num_steps, model.observations.num_obs, 1
            )
            vec_t = rel_vectors[sim_id].reshape(
                model.observations.num_steps, 1, model.observations.num_obs
            )

            power = (vec_t @ inv_cov_matrices @ vec).flatten()

            likelihoods[:, sim_id] = np.exp(-0.5 * power)
           
        # regularize
        likelihoods = likelihoods / likelihoods.sum(axis=1)[:, None]

        return likelihoods

    def get_posterors(
        self, model: Type["Model"], likelihoods: np.array, proposal_prev: np.array
    ) -> np.array:

        posteriors = np.zeros((model.observations.num_steps, model.num_samples))

        posteriors[0, :] = likelihoods[0, :] / proposal_prev

        for stp_id in range(1, model.observations.num_steps):
            posteriors[stp_id, :] = posteriors[stp_id - 1, :] * likelihoods[stp_id, :]
            posteriors[stp_id, :] /= posteriors[stp_id, :].sum()
            
            #TODO normalization per step not after
        # posteriors = posteriors / posteriors.sum(axis=1)[:, None]

        return posteriors

    def get_ensamble_ips_covs(
        self,
        model: Type["Model"],
        posteriors: np.array,
    ) -> np.array:

        ips = np.zeros((model.observations.num_steps, model.parameters.num_params))
        covs = np.zeros((model.observations.num_steps, model.parameters.num_params))

        for stp_id in range(model.observations.num_steps):

            ips[stp_id, :] = posteriors[stp_id, :] @ model.parameters.data

            covs[stp_id, :] = (
                posteriors[stp_id, :] @ (ips[stp_id, :] - model.parameters.data) ** 2
            )

            covs[stp_id, :] = np.sqrt(covs[stp_id, :]) / ips[stp_id, :]

        return ips, covs


    def give_proposal(self):
        return self.posteriors[-1,:]
        
    def data_assimilation_loop(
        self, sigma_guess: float, proposal_prev: np.ndarray, model: Type["Model"]
    ):

        self.cov_matrices = self.get_covariance_matrices(
            sigma_guess=sigma_guess, model=model
        )
        self.likelihoods = self.get_likelihoods(
            model=model, cov_matrices=self.cov_matrices
        )

        self.posteriors = self.get_posterors(
            model=model, likelihoods=self.likelihoods, proposal_prev=proposal_prev
        )

        self.ips, self.covs = self.get_ensamble_ips_covs(
            model=model, posteriors=self.posteriors
        )

        eff_all_steps = 1.0 / sum(self.posteriors**2)
        
        self.eff = eff_all_steps[-1]
        
        return (self.eff - self.ess_target)**2