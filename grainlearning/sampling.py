"""
This module contains various methods to sample the state-parameter space of a dynamic system.
"""
from typing import Type
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats.qmc import Sobol, Halton, LatinHypercube
from grainlearning.dynamic_systems import DynamicSystem
# from grainlearning.tools import regenerate_params_with_gmm, unweighted_resample#


class GaussianMixtureModel:
    """This is the wrapper class of the `Bayesian Gaussian Mixture <https://scikit-learn.org/stable/modules/generated/
    sklearn.mixture.BayesianGaussianMixture.html>`_. from scikit-learn.

    The `BayesianGaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture
    .BayesianGaussianMixture.html>`_. class is extended to include the following functionalities:

    -   :meth:`.GaussianMixtureModel.expand_and_normalize_weighted_samples`:
        Converting an ensemble of samples with importance weights into an ensemble of samples with equal weights
    -   :meth:`.GaussianMixtureModel.regenerate_params`:
        Regenerating new samples by training and sampling from a Gaussian mixture model.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        system_cls = GaussianMixtureModel.from_dict(
            {
                "max_num_components": 2,
                "covariance_type": "full",
                "random_state": 0,
                slice_sampling: False,
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        system_cls = GaussianMixtureModel(
            max_num_components = 2,
            covariance_type = "full",
            random_state = 0,
            slice_sampling = False,
        )

    :param max_num_components: Maximum number of components
    :param weight_concentration_prior: The dirichlet concentration of each component on the weight distribution \
    (Dirichlet), default to None, optional
        This is commonly called gamma in the literature. The higher concentration puts more mass in the center and will
        lead to more components being active, while a lower concentration parameter will lead to more mass at the edge
        of the mixture weights simplex. The value of the parameter must be greater than 0. If it is None,
        it’s set to `1. / n_components`.
    :param covariance_type: {‘full’, ‘tied’, ‘diag’, ‘spherical’}, defaults to "full", optional
        String describing the type of covariance parameters to use. Must be one of:
        :samp:`'full'`: each component has its own general covariance matrix.
        :samp:`'tied'`: all components share the same general covariance matrix.
        :samp:`'diag'`: each component has its own diagonal covariance matrix.
        :samp:`'spherical'`: each component has its own single variance.
    :param n_init: number of initialization to perform, defaults to 1, optional
        The result with the highest lower bound value on the likelihood is kept.
    :param tol: tolerance threshold, defaults to 1.0e-3, optional
        EM iterations will stop when the lower bound average gain on the likelihood (of the training data with respect
        to the model) is below this threshold.
    :param max_iter: maximum number of EM iterations to perform, defaults to 100, optional
    :param random_state: random seed given to the method chosen to initialize the parameters, defaults to None, optional
    :param init_params: {‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}, default=’kmeans’, optional
        The method used to initialize the weights, the means and the covariances. String must be one of:
        :samp:`'kmeans'`: responsibilities are initialized using kmeans.
        :samp:`k-means++`: responsibilities are initialized using a k-means++ algorithm.
        :samp:`'random'`: responsibilities are initialized randomly.
        :samp:`'random_from_data'`: responsibilities are initialized randomly from the data.
    :param warm_start: flag to use warm start, defaults to False, optional
        If ‘warm_start’ is True, the solution of the last fitting is used as initialization for the next call of fit().
        This can speed up convergence when fit is called several times on similar problems. See the Glossary.
    :param expand_factor: factor used when converting the ensemble from weighted to unweighted, defaults to 10, optional
    :param slice_sampling: flag to use slice sampling, defaults to False, optional
    """
    #: Maximum number of components
    max_num_components: int = 0

    #: The dirichlet concentration of each component on the weight distribution (Dirichlet), default to None.
    weight_concentration_prior: float = 0.0

    #: String describing the type of covariance parameters to use.
    covariance_type: str = "full"

    #: number of initialization to perform, defaults to 1.
    n_init: int = 1

    #: tolerance threshold, defaults to 1.0e-3.
    tol: float = 1.0e-3

    #: maximum number of EM iterations to perform, defaults to 100
    max_iter: int = 100

    #: random seed given to the method chosen to initialize the weights, the means and the covariances.
    random_state: int

    #: The method used to initialize the weights, the means and the covariances.
    init_params: str = "kmeans"

    #: flag to use warm start, defaults to False.
    warm_start: bool = False

    #: the factor used when converting and populating the ensemble from weighted to unweighted, defaults to 10.
    expand_factor: int = 10

    #: flag to use slice sampling, defaults to False.
    slice_sampling: False

    gmm: Type["BayesianGaussianMixture"]

    def __init__(
        self,
        max_num_components,
        weight_concentration_prior: float = None,
        covariance_type: str = "tied",
        n_init: int = 1,
        tol: float = 1.0e-5,
        max_iter: int = 100,
        random_state: int = None,
        init_params: str = "kmeans",
        warm_start: bool = False,
        expand_factor: int = 10,
        slice_sampling: bool = False,
    ):
        """ Initialize the Gaussian Mixture Model class"""
        self.max_num_components = max_num_components

        if weight_concentration_prior is None:
            self.weight_concentration_prior = 1.0 / max_num_components
        else:
            self.weight_concentration_prior = weight_concentration_prior

        self.covariance_type = covariance_type

        self.n_init = n_init

        self.tol = tol

        self.max_iter = max_iter

        self.random_state = random_state

        self.init_params = init_params

        self.warm_start = warm_start

        self.expand_factor = expand_factor

        self.slice_sampling = slice_sampling

        self.max_params = None

        self.expanded_normalized_params = None

        self.gmm = Type["BayesianGaussianMixture"]

    @classmethod
    def from_dict(cls: Type["GaussianMixtureModel"], obj: dict):
        """Initialize the class using a dictionary style

        :param obj: a dictionary containing the keys and values to construct a GMM object
        :return: a GMM object
        """
        return cls(
            max_num_components=obj["max_num_components"],
            weight_concentration_prior=obj.get("weight_concentration_prior", None),
            covariance_type=obj.get("covariance_type", "tied"),
            n_init=obj.get("n_init", 1),
            tol=obj.get("tol", 1.0e-5),
            max_iter=obj.get("max_iter", 100),
            random_state=obj.get("random_state", None),
            init_params=obj.get("init_params", "kmeans"),
            warm_start=obj.get("warm_start", False),
            expand_factor=obj.get("expand_factor", 10),
            slice_sampling=obj.get("slice_sampling", False),
        )

    def expand_and_normalize_weighted_samples(self, weights: np.ndarray, system: Type["DynamicSystem"]):
        """Converting an ensemble of samples with importance weights into an ensemble of samples with equal weights.

        For instance, a sample is duplicated a few times depending on the importance weight associated to it.

        :param weights: Importance weights associated with the ensemble
        :param system: Dynamic system class
        :return: Expanded unweighted samples
        """
        num_copies = (
            np.floor(
                self.expand_factor * system.num_samples * np.asarray(weights)
            )
        ).astype(int)

        indices = np.repeat(np.arange(system.num_samples), num_copies).astype(int)

        expanded_parameters = system.param_data[indices]

        max_params = np.amax(expanded_parameters, axis=0)  # find max along axis

        normalized_parameters = (
            expanded_parameters / max_params
        )  # and do array broadcasting to divide by max

        self.expanded_normalized_params = normalized_parameters
        self.max_params = max_params

    def regenerate_params(self, weight: np.ndarray, system: Type["DynamicSystem"]):
        """Regenerating new samples by training and sampling from a Gaussian mixture model.

        :param weight: Posterior found by the data assimilation
        :param system: Dynamic system class
        :return: Resampled parameter data
        """
        self.expand_and_normalize_weighted_samples(weight, system)

        self.gmm = BayesianGaussianMixture(
            n_components=self.max_num_components,
            weight_concentration_prior=self.weight_concentration_prior,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            tol=self.tol,
            max_iter=int(self.max_iter),
            random_state=self.random_state,
            init_params=self.init_params,
            warm_start=self.warm_start,
        )

        self.gmm.fit(self.expanded_normalized_params)
        minimum_num_samples = system.num_samples

        new_params = self.draw_samples_within_bounds(system, system.num_samples)

        # resample until all parameters are within the upper and lower bounds
        test_num = system.num_samples
        while system.param_min and system.param_max and new_params.shape[0] < minimum_num_samples:
            test_num = int(np.ceil(1.1 * test_num))
            new_params = self.draw_samples_within_bounds(system, test_num)

        return new_params

    def draw_samples_within_bounds(self, system: Type["DynamicSystem"], num: int = 1):
        """Draw new parameter samples within the user-defined upper and lower bounds

        :param system: Dynamic system class
        :param num: Number of samples to draw
        :return: New parameter samples
        """
        if not self.slice_sampling:
            new_params, _ = self.gmm.sample(num)

        # use the slice sampling scheme for resampling
        else:
            # compute the minimum of score_samples as the threshold for slice sampling
            new_params = generate_params_qmc(system, num)
            new_params /= self.max_params

            scores = self.gmm.score_samples(self.expanded_normalized_params)
            new_params = new_params[np.where(
                self.gmm.score_samples(new_params) > scores.mean() - 2 * scores.std())]

        new_params *= self.max_params

        if system.param_min and system.param_max:
            params_above_min = new_params > np.array(system.param_min)
            params_below_max = new_params < np.array(system.param_max)
            bool_array = params_above_min & params_below_max
            indices = bool_array[:, 0]
            for i in range(system.num_params - 1):
                indices = np.logical_and(indices, bool_array[:, i + 1])
            return new_params[indices]
        else:
            return new_params

    # def regenerate_params_with_gmm(
    #     self, posterior_weight: np.ndarray, system: Type["DynamicSystem"]
    # ) -> np.ndarray:
    #     """Regenerate the parameters by fitting the Gaussian Mixture model (for testing against the old approach)
    #
    #     :param posterior_weight: Posterior found by the data assimilation
    #     :param system: Dynamic system class
    #     :return: Expanded parameters
    #     """
    #
    #     new_params, self.gmm = regenerate_params_with_gmm(
    #         posterior_weight,
    #         system.param_data,
    #         system.num_samples,
    #         self.max_num_components,
    #         self.weight_concentration_prior,
    #         self.covariance_type,
    #         unweighted_resample,
    #         system.param_min,
    #         system.param_max,
    #         self.n_init,
    #         self.tol,
    #         self.max_iter,
    #         self.random_state,
    #     )
    #
    #     return new_params


def generate_params_qmc(system: Type["DynamicSystem"], num_samples: int, method: str = "halton"):
    """This is the class to uniformly draw samples in n-dimensional space from
    a low-discrepancy sequence or a Latin hypercube.

    See `Quasi-Monte Carlo <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>`_.

    :param system: Dynamic system class
    :param num_samples: Number of samples to draw
    :param method: Method to use for Quasi-Monte Carlo sampling. Options are "halton", "sobol", and "LH"
    """

    sampler = Halton(system.num_params, scramble=False)

    if method == "sobol":
        sampler = Sobol(system.num_params)
        random_base = round(np.log2(num_samples))
        num_samples = 2 ** random_base

    elif method == "LH":
        sampler = LatinHypercube(system.num_params)

    param_table = sampler.random(n=num_samples)

    for param_i in range(system.num_params):
        for sim_i in range(num_samples):
            mean = 0.5 * (system.param_max[param_i] + system.param_min[param_i])
            std = 0.5 * (system.param_max[param_i] - system.param_min[param_i])
            param_table[sim_i][param_i] = (
                mean + (param_table[sim_i][param_i] - 0.5) * 2 * std
            )

    return np.array(param_table, ndmin=2)
