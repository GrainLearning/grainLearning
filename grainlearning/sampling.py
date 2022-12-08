# #%%

from typing import Tuple, Type
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats.qmc import Sobol, Halton, LatinHypercube

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
    :param cov_type: Covariance type, defaults to "tied"
    :param n_init: number of initial samples, defaults to 100
    :param tol: tolarance, defaults to 1.0e-5
    :param max_iter: maximum number of iterations, defaults to 100000
    :param expand_weight: weighted expansions, defaults to 10
    :param seed: random generation seed, defaults to None
    :param slice_sampling: flag to use slice sampling, defaults to False
    """
    max_num_components: int = 0

    prior_weight: float = 0.0

    cov_type: str = "tied"

    n_init: int = 1

    tol: float = 1.0e-3

    max_iter: int = 100000

    expand_weight: int = 10

    seed: int

    slice_sampling: False

    gmm: Type["BayesianGaussianMixture"]

    def __init__(
        self,
        max_num_components,
        prior_weight: int = None,
        cov_type: str = "tied",
        n_init: int = 1,
        tol: float = 1.0e-5,
        max_iter: int = 100000,
        expand_weight: int = 10,
        seed: int = None,
        slice_sampling: bool = False,
    ):
        """ Initialize the gaussian mixture model class"""
        self.max_params = None
        self.expanded_normalized_params = None
        self.max_num_components = max_num_components
        self.cov_type = cov_type
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.slice_sampling = slice_sampling
        self.expand_weight = expand_weight

        if prior_weight is None:
            self.prior_weight = 1.0 / max_num_components
        else:
            self.prior_weight = prior_weight

    @classmethod
    def from_dict(cls: Type["GaussianMixtureModel"], obj: dict):
        """Initialize the class using a dictionary style"""
        return cls(
            max_num_components=obj["max_num_components"],
            prior_weight=obj.get("prior_weight", None),
            cov_type=obj.get("cov_type", "tied"),
            n_init=obj.get("n_init", 1),
            tol=obj.get("tol", 1.0e-5),
            max_iter=obj.get("max_iter", 100000),
            seed=obj.get("seed", None),
            slice_sampling=obj.get("slice_sampling", False),
            expand_weight=obj.get("expand_weight", 10),
        )

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

        self.expanded_normalized_params = normalized_parameters
        self.max_params = max_params

    def regenerate_params(
        self, posterior_weight: np.ndarray, model: Type["Model"],
    ) -> np.ndarray:
        """Regenerate the parameters by fitting the Gaussian Mixture model

        :param posterior_weight: Posterior found by the data assimulation
        :param model: Model class
        :return: Expanded parameters
        """
        self.expand_weighted_parameters(posterior_weight, model)

        self.gmm = BayesianGaussianMixture(
            n_components=self.max_num_components,
            weight_concentration_prior=self.prior_weight,
            covariance_type=self.cov_type,
            tol=self.tol,
            max_iter=int(self.max_iter),
            n_init=self.n_init,
            random_state=self.seed,
        )

        self.gmm.fit(self.expanded_normalized_params)
        minimum_num_samples = model.num_samples

        new_params = self.get_samples_within_bounds(model, model.num_samples)

        # resample until all parameters are within the upper and lower bounds
        test_num = model.num_samples
        while model.param_mins and model.param_maxs and len(new_params) < minimum_num_samples:
            test_num = int(np.ceil(1.1 * test_num))
            new_params = self.get_samples_within_bounds(model, test_num)

        return new_params

    def get_samples_within_bounds(
        self, model: Type["Model"], num: int) -> np.ndarray:

        if not self.slice_sampling:
            new_params, _ = self.gmm.sample(num)

        # use slice sampling scheme for resampling
        else:
            # define the mininum of score_samples as the threshold for slice sampling
            new_params = generate_params_qmc(model, num)
            new_params /= self.max_params

            scores = self.gmm.score_samples(self.expanded_normalized_params)
            new_params = new_params[np.where(
                self.gmm.score_samples(new_params) > scores.mean() - 2 * scores.std())]

        new_params *= self.max_params

        if model.param_mins and model.param_maxs:
            params_above_min = new_params > np.array(model.param_mins)
            params_below_max = new_params < np.array(model.param_maxs)
            bool_array = params_above_min & params_below_max
            indices = bool_array[:, 0]
            for i in range(model.num_params - 1):
                indices = np.logical_and(indices, bool_array[:, i + 1])
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


def generate_params_qmc(model: Type["Model"], num_samples: int, method: str = "halton") -> np.ndarray:
    """This is the class to uniformly draw samples in n-dimensional space from
    a low-discrepancy sequence or a Latin hypercube.

    See `Quasi-Monte Carlo <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>`_.

    :param max_iter: maximum number of iterations, defaults to 100000
    :param expand_weight: weighted expansions, defaults to 10
    :param seed: random generation seed, defaults to None
    """

    if method == "halton":
        sampler = Halton(model.num_params, scramble=False)

    elif method == "sobol":
        sampler = Sobol(model.num_params)
        random_base = round(np.log2(num_samples))
        num_samples = 2 ** random_base

    elif method == "LH":
        sampler = LatinHypercube(model.num_params)

    param_table = sampler.random(n=num_samples)

    for param_i in range(model.num_params):
        for sim_i in range(num_samples):
            mean = 0.5 * (model.param_maxs[param_i] + model.param_mins[param_i])
            std = 0.5 * (model.param_maxs[param_i] - model.param_mins[param_i])
            param_table[sim_i][param_i] = (
                mean + (param_table[sim_i][param_i] - 0.5) * 2 * std
            )

    return np.array(param_table, ndmin=2)
