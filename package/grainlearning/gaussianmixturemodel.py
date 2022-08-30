# #%%

from typing import Tuple, Type
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

from .models import Model


class GaussianMixtureModel:
    """This class is used for variational inference (sampling) of parameters using a baysian gausian mixture model.

    See `BayesianGaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html>`_.
    """

    #: number of components for the Gaussian mixture model
    max_num_components: int = 0

    prior_weight: int = 0
    cov_type: str = "full"
    n_init: int = 100
    tol: float = 1.0e-3
    max_iter: int = 100000
    expand_weight: int = 10
    seed: int
    gmm: Type["BayesianGaussianMixture"]

    def __init__(
        self,
        max_num_components,
        prior_weight: int = None,
        cov_type: str = "full",
        n_init: int = 100,
        tol: float = 1.0e-5,
        max_iter: int = 100000,
        expand_weight: int = 10,
        seed: int = None,
    ):
        self.max_num_components = max_num_components
        self.cov_type = cov_type
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.expand_weight = expand_weight

        if prior_weight is None:
            self.prior_weight = 1.0 / max_num_components
        else:
            self.prior_weight = prior_weight

    @classmethod
    def from_dict(cls: Type["GaussianMixtureModel"], obj: dict):
        return cls(
            max_num_components=obj["max_num_components"],
            prior_weight=obj.get("prior_weight", None),
            cov_type=obj.get("cov_type", "full"),
            n_init=obj.get("n_init", 100),
            tol=obj.get("tol", 1.0e-5),
            max_iter=obj.get("max_iter", 100000),
            seed=obj.get("seed", None),
            expand_weight=obj.get("expand_weight", 10),
        )

    def expand_weighted_parameters(
        self, proposal_weight: np.ndarray, model: Type["Model"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_copies = (
            np.floor(
                self.expand_weight
                * model.num_samples
                * np.asarray(proposal_weight)
            )
        ).astype(int)

        indices = np.repeat(np.arange(model.num_samples), num_copies).astype(int)

        expanded_parameters = model.parameters.data[indices]

        max_params = np.amax(expanded_parameters, axis=0)  # find max along axis

        normalized_parameters = (
            expanded_parameters / max_params
        )  #  and do array broadcasting to divide by max
        return normalized_parameters, max_params


    def regenerate_params(
        self, proposal_weight: np.ndarray, simulations: Type["Model"]
    ) -> np.ndarray:
        expanded_normalized_params, max_params = self.expand_weighted_parameters(
            proposal_weight, simulations
        )

        self.gmm = BayesianGaussianMixture(
            n_components=self.max_num_components,
            weight_concentration_prior=self.prior_weight,
            covariance_type=self.cov_type,
            tol=self.tol,
            max_iter=int(self.max_iter),
            n_init=self.n_init,
            random_state=self.seed,
        )
        self.gmm.fit(expanded_normalized_params)
        new_params, _ = self.gmm.sample(simulations.num_samples)
        new_params *= max_params
        # resample until all parameters are within min and max bounds, is there a better way to do this?
        # while True:
        #     print("resample")
        #     new_params, _ = self.gmm.sample(simulations.num_samples)
        #     new_params *= max_params

        #     params_above_min = new_params > np.array(simulations.parameters.mins)
        #     params_below_max = new_params < np.array(simulations.parameters.maxs)

        #     if params_above_min.all() & params_below_max.all():
        #         break
        return new_params
