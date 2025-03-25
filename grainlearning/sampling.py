"""
This module contains various methods to sample the state-parameter space of a dynamic system.
"""
from typing import Type
from pickle import dump, load
import numpy as np
import warnings
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats.qmc import Sobol, Halton, LatinHypercube
from scipy.optimize import root_scalar
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
    :param deviation_factor: factor used to determine the threshold,
        scores.mean() - deviation_factor * scores.std(), for slice sampling, defaults to 0.0, optional
    :param gmm: The class of the Gaussian Mixture Model
    :param max_params: Current maximum values of the parameters
    :param min_params: Current minimum values of the parameters
    :param expanded_normalized_params: Normalized parameters expanded from the importance weights
    :param scores: Scores of the Gaussian Mixture Model
    """

    def __init__(
        self,
        max_num_components: int = 1,
        weight_concentration_prior: float = 0.0,
        covariance_type: str = "tied",
        n_init: int = 1,
        tol: float = 1.0e-5,
        max_iter: int = 100,
        random_state: int = None,
        init_params: str = "k-means++",
        warm_start: bool = False,
        expand_factor: int = 10,
        slice_sampling: bool = False,
        deviation_factor: float = 0.0,
    ):
        """ Initialize the Gaussian Mixture Model class"""
        self.max_num_components = max_num_components

        if not weight_concentration_prior:
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

        self.deviation_factor = deviation_factor

        self.max_params = None

        self.min_params = None

        self.expanded_normalized_params = None

        self.gmm = None

        self.scores = None

    @classmethod
    def from_dict(cls: Type["GaussianMixtureModel"], obj: dict):
        """Initialize the class using a dictionary style

        :param obj: a dictionary containing the keys and values to construct a GMM object
        :return: a GMM object
        """
        return cls(
            max_num_components=obj.get("max_num_components", 1),
            weight_concentration_prior=obj.get("weight_concentration_prior", None),
            covariance_type=obj.get("covariance_type", "tied"),
            n_init=obj.get("n_init", 10),
            tol=obj.get("tol", 1.0e-5),
            max_iter=obj.get("max_iter", 100),
            random_state=obj.get("random_state", None),
            init_params=obj.get("init_params", "k-means++"),
            warm_start=obj.get("warm_start", False),
            expand_factor=obj.get("expand_factor", 10),
            slice_sampling=obj.get("slice_sampling", False),
            deviation_factor=obj.get("deviation_factor", 0.0),
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
        min_params = np.amin(expanded_parameters, axis=0)  # find min along axis

        # Normalize the parameters from 0 to 1
        self.max_params = max_params
        self.min_params = min_params
        self.expanded_normalized_params = self.normalize(expanded_parameters)

    def normalize(self, params: np.ndarray):
        """Normalize the parameters from 0 to 1.

        :param params: Parameters to normalize
        :return: Normalized parameters
        """
        return (params - self.min_params) / (self.max_params - self.min_params)

    def train(self, weight: np.ndarray, system: Type["DynamicSystem"]):
        """Train the Gaussian mixture model.

        :param weight: Posterior found by the data assimilation
        :param system: Dynamic system class
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

        self.scores = self.gmm.score_samples(np.unique(self.expanded_normalized_params, axis=0))
        # FIXME: Gaussian mixture model introduces bias leading to covariances resulting from the trained probability density not matching the sample covariances
        # means_ref = weight.dot(system.param_data)
        # covs_ref = weight.dot((system.param_data - means_ref)**2)
        # self.scores = self.correct_scores(covs_ref, system)

    def correct_scores(self, covs_ref: float, system: Type["DynamicSystem"], tol: float = 1e-3, num_points: int = 10):
        """Correct the covariance (preceision) matrices and the scores of the Gaussian Mixture Model.

        :param covs_ref: Reference covariance from the current ensemble
        :param system: Dynamic system class that contains the parameter samples
        :param tol: Tolerance threshold on the difference between the sample covariance and the covariance estimated by GMM , defaults to 0.1, optional
        """
        # raise an error is the GMM is not initialized or not converged
        if self.gmm is None or not self.gmm.converged_:
            raise RuntimeError("The Gaussian mixture model is not initialized or not trained to convergence.")
        else:
            # normalize the parameters to the range [0, 1]
            params_normalized = self.normalize(system.param_data)
            # update the covariance matrices until the ratio of the norms of the reference and new covariances is less than 1 + tol
            ratio_history = []
            max_diff = 1e10
            while True:
                weight_new = np.exp(self.gmm.score_samples(params_normalized))
                weight_new /= weight_new.sum()
                means_new = weight_new.dot(system.param_data)
                covs_new = weight_new.dot((system.param_data - means_new)**2)
                ratio = np.linalg.norm(covs_ref)/np.linalg.norm(covs_new)
                ratio_history.append(ratio)
                # define a list of ratio values to check convergence
                if len(ratio_history) > num_points:
                    diffs = [np.abs(ratio_history[-i] - ratio_history[-i-1]) for i in range(1, num_points)]
                    max_diff = np.max(diffs, axis=0)
                # update the scores
                if abs(ratio - 1) < tol or np.all(max_diff < tol):
                    print(f"Converged at a ratio of {ratio}")
                    return self.gmm.score_samples(params_normalized)
                else:
                    # update the covariance matrices
                    self.gmm.covariances_ *= ratio
                    self.gmm.precisions_ /= ratio
                    self.gmm.precisions_cholesky_ /= ratio

    def regenerate_params(self, weight: np.ndarray, system: Type["DynamicSystem"]):
        """Regenerating new samples by training and sampling from a Gaussian mixture model.

        :param weight: Posterior found by the data assimilation
        :param system: Dynamic system class
        :return: Resampled parameter data
        """
        # Train the model using the provided weight and parameters from the dynamic system
        self.train(weight, system)

        # Minimum number of samples required
        minimum_num_samples = system.num_samples

        last_samples = None

        # Update the upper bound x1 for the root scalar optimization
        x1 = 100 * system.num_samples_max
        while self.draw_samples_within_bounds(system, x1).shape[0] < minimum_num_samples:
            x1 *= 10

        def sample_count_obj(num_samples):
            """Objective function to minimize the number of samples drawn."""
            nonlocal last_samples
            # Ensure we have an integer number of samples
            num_samples = max(int(np.ceil(num_samples)), minimum_num_samples)
            samples = self.draw_samples_within_bounds(system, num_samples)
            valid_samples_count = samples.shape[0]
            last_samples = samples

            if valid_samples_count < minimum_num_samples:
                # Strong negative penalty to ensure root_scalar sees a negative value
                return -1e3 - (minimum_num_samples - valid_samples_count)
            else:
                # Root is when this becomes zero
                return valid_samples_count - minimum_num_samples

        # Perform optimization to find the minimum number of samples needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = root_scalar(
                sample_count_obj,
                x0=system.num_samples_max,
                x1=x1,
                method='secant',
                xtol=1.0,
                maxiter=int(1e5)
            )

        system.num_samples_max = int(np.ceil(result.root))

        # Low discrepancy should be preserved after the truncation
        return last_samples[:minimum_num_samples]

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
            new_params = self.scale_qmc_to_gmm_range(generate_qmc(system, num))
            new_params = new_params[np.where(
                self.gmm.score_samples(new_params) >= self.scores.mean() - self.deviation_factor * self.scores.std())]

        # Un-normalize the parameters
        new_params = new_params * (self.max_params - self.min_params) + self.min_params

        # Ensure the parameters are within the user-defined bounds
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

    def scale_qmc_to_gmm_range(self, qmc_samples: np.ndarray):
        """Scale the quasi-random number samples to the range of data generated by the Gaussian Mixture Model."""

        # Generate trial samples to estimate the maximum and minimum samples generated by gmm
        samples, _ = self.gmm.sample(100 * self.expanded_normalized_params.shape[0])
        max_gmm_samples = np.amax(samples, axis=0)  # find max along axis
        min_gmm_samples = np.amin(samples, axis=0)  # find min along axis

        # Scale the quasi-random number samples to the range of data generated by the sampleGaussian Mixture Model
        qmc_samples = qmc_samples * (max_gmm_samples - min_gmm_samples) + min_gmm_samples
        return qmc_samples

    def save_gmm_to_file(self, proposal_data_file: str = "proposal_density.pkl"):
        """Save the Gaussian mixture model to a file."""
        with open(proposal_data_file, "wb") as f:
            dump((self.max_params, self.gmm), f)

    def load_gmm_from_file(self, proposal_data_file: str = "proposal_density.pkl"):
        """Load the Gaussian mixture model from a file.

        :param proposal_data_file: Name of the file that stores the trained Gaussian mixture model
        """
        with open(proposal_data_file, "rb") as f:
            self.max_params, self.gmm = load(f)


def generate_qmc(system: Type["DynamicSystem"], num_samples: int, method: str = "halton", seed=None):
    """This is the class to uniformly draw samples in n-dimensional space from
    a low-discrepancy sequence or a Latin hypercube.

    See `Quasi-Monte Carlo <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>`_.

    :param system: Dynamic system class
    :param num_samples: Number of samples to draw
    :param method: Method to use for Quasi-Monte Carlo sampling. Options are "halton", "sobol", and "LH"
    :param seed: Seed for the random number generator
    """

    sampler = Halton(system.num_params, scramble=False)

    if method == "sobol":
        sampler = Sobol(system.num_params, seed=seed)
        random_base = int(np.ceil(np.log2(num_samples)))
        num_samples = 2 ** random_base

    elif method == "LH":
        sampler = LatinHypercube(system.num_params, seed=seed)

    param_table = sampler.random(n=num_samples)
    return np.array(param_table, ndmin=2)


def generate_params_qmc(system: Type["DynamicSystem"], num_samples: int, method: str = "halton", seed=None):
    """This is the class to uniformly draw samples in n-dimensional space from
    a low-discrepancy sequence or a Latin hypercube.

    See `Quasi-Monte Carlo <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>`_.

    :param system: Dynamic system class
    :param num_samples: Number of samples to draw
    :param method: Method to use for Quasi-Monte Carlo sampling. Options are "halton", "sobol", and "LH"
    :param seed: Seed for the random number generator
    """

    param_table = generate_qmc(system, num_samples, method, seed)

    for param_i in range(system.num_params):
        for sim_i in range(num_samples):
            mean = 0.5 * (system.param_max[param_i] + system.param_min[param_i])
            std = 0.5 * (system.param_max[param_i] - system.param_min[param_i])
            param_table[sim_i][param_i] = (
                mean + (param_table[sim_i][param_i] - 0.5) * 2 * std
            )

    return np.array(param_table, ndmin=2)
