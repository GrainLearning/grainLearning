# #%%

from typing import Tuple, Type
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

from .models import Model
from .tools import regenerate_params_with_gmm, unweighted_resample

class GaussianMixtureModel:
    """This class is used for the inference (sampling) of parameters using a variational Gaussian mixture model.

    See `BayesianGaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture
    .BayesianGaussianMixture.html>`_.

    There are two ways of initializing the class.

    Method 1 - dictionary style
    
    .. highlight:: python
    .. code-block:: python
    
        model_cls = GaussianMixtureModel.from_dict(
            {
                "max_num_components": 2
            }
        )

    or

    Method 2 - class style
    
    .. highlight:: python
    .. code-block:: python
    
        model_cls = GaussianMixtureModel(
            max_num_components = 2
        )

    :param max_num_components: Maximum number of components
    :param prior_weight: Prior weight, defaults to None
    :param cov_type: Covariance type, defaults to "full"
    :param n_init: number of initial samples, defaults to 100
    :param tol: tolarance, defaults to 1.0e-5
    :param max_iter: maximum number of iterations, defaults to 100000
    :param expand_weight: weighted expansions, defaults to 10
    :param seed: random generation seed, defaults to None
    """
    max_num_components: int = 0

    prior_weight: float = 0.0

    cov_type: str = "full"

    n_init: int = 1

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
            n_init: int = 1,
            tol: float = 1.0e-5,
            max_iter: int = 100000,
            expand_weight: int = 10,
            seed: int = None,
    ):
        """ Initialize the gaussian mixture model class"""
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

        self.gmm = BayesianGaussianMixture(
            n_components=self.max_num_components,
            weight_concentration_prior=self.prior_weight,
            covariance_type=self.cov_type,
            tol=self.tol,
            max_iter=int(self.max_iter),
            n_init=self.n_init,
            random_state=self.seed,
        )

    @classmethod
    # TODO: with this class method, GaussianMixtureModel has only one argument, can we use **kwargs to allow more user input?
    def from_dict(cls: Type["GaussianMixtureModel"], obj: dict):
        """Initialize the class using a dictionary style"""
        return cls(
            max_num_components=obj["max_num_components"],
            prior_weight=obj.get("prior_weight", None),
            cov_type=obj.get("cov_type", "full"),
            n_init=obj.get("n_init", 1),
            tol=obj.get("tol", 1.0e-5),
            max_iter=obj.get("max_iter", 100000),
            seed=obj.get("seed", None),
            expand_weight=obj.get("expand_weight", 10),
        )

    # TODO: review below with Retief (originally in resample.py)
    def expand_weighted_parameters(
            self, posterior_weight: np.ndarray, model: Type["Model"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expand or duplicate the parameters for the gaussian mixture model. If the weights are higher, more parameters are assigned to that value

        :param posterior_weight: Posterior found by the data assimulation
        :param model: Model class
        :return: Expanded parameters
        """
        num_copies = (
            np.floor(
                self.expand_weight * model.num_samples * np.asarray(posterior_weight)
            )
        ).astype(int)

        indices = np.repeat(np.arange(model.num_samples), num_copies).astype(int)

        expanded_parameters = model.param_data[indices]

        max_params = np.amax(expanded_parameters, axis=0)  # find max along axis

        normalized_parameters = (
                expanded_parameters / max_params
        )  # and do array broadcasting to divide by max

        return normalized_parameters, max_params

    # TODO: this should go to the sampling class
    def generate_params_halton(self, model: Type["Model"]) -> np.ndarray:
        """Generate a Halton table of the parameters"""

        from scipy.stats import qmc

        halton_sampler = qmc.Halton(model.num_params, scramble=False)
        param_table = halton_sampler.random(n=model.num_samples)

        for param_i in range(model.num_params):
            for sim_i in range(model.num_samples):
                mean = 0.5 * (model.param_maxs[param_i] + model.param_mins[param_i])
                std = 0.5 * (model.param_maxs[param_i] - model.param_mins[param_i])
                param_table[sim_i][param_i] = (
                        mean + (param_table[sim_i][param_i] - 0.5) * 2 * std
                )

        model.param_data = np.array(param_table, ndmin=2)

        return model.param_data

    def regenerate_params(
            self, posterior_weight: np.ndarray, model: Type["Model"]
    ) -> np.ndarray:
        """Regenerate the parameters by fitting the Gaussian Mixture model

        :param posterior_weight: Posterior found by the data assimulation
        :param model: Model class
        :return: Expanded parameters
        """
        expanded_normalized_params, max_params = self.expand_weighted_parameters(
            posterior_weight, model
        )

        self.gmm.fit(expanded_normalized_params)
        minimum_num_samples = model.num_samples

        new_params = self.get_samples_within_bounds(model, max_params)

        # resample until all parameters are within the upper and lower bounds
        while (model.param_mins and model.param_maxs and len(new_params) < minimum_num_samples):
            model.num_samples *= 1.1
            new_params = self.get_samples_within_bounds(model, max_params)

        model.num_samples = new_params.shape[0]

        return new_params

    def get_samples_within_bounds(
            self, model: Type["Model"], max_params
    ) -> np.ndarray:
        new_params, _ = self.gmm.sample(model.num_samples)
        new_params *= max_params
        
        if model.param_mins and model.param_maxs:
            params_above_min = new_params > np.array(model.param_mins)
            params_below_max = new_params < np.array(model.param_maxs)
            bool_array = params_above_min & params_below_max
            indices = bool_array[:,0]
            for i in range(model.num_params-1):
                indices = np.logical_and(indices, bool_array[:,i+1])
            return new_params[indices]
        else:
            return new_params

    def regenerate_params_with_gmm(
            self, posterior_weight: np.ndarray, model: Type["Model"]
    ) -> np.ndarray:
        """Regenerate the parameters by fitting the Gaussian Mixture model (for testing against the old approach)

        :param posterior_weight: Posterior found by the data assimulation
        :param model: Model class
        :return: Expanded parameters
        """

        new_params, self.gmm = regenerate_params_with_gmm(
            posterior_weight,
            model.param_data,
            model.num_samples,
            self.max_num_components,
            self.prior_weight,
            self.cov_type,
            unweighted_resample,
            model.param_mins,
            model.param_maxs,
            self.n_init,
            self.tol,
            self.max_iter,
            self.seed,
        )

        return new_params
